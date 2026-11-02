import os
import sys
import random
import glob
import datetime
import time
import functools
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # plots are only saved; LSB debug plots are drawn in the postproc thread
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d, median_filter

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

from mea1k_modules.mea1k_config_utils import create_stim_sine_sequence, get_headstage_stim_settings
from mea1k_modules.mea1k_config_utils import start_saving, stop_saving, shift_DAC
from mea1k_modules.mea1k_config_utils import turn_on_stimulation_units, turn_off_stimulation_units
from mea1k_modules.mea1k_config_utils import setup_stim_unit_characterization, find_stim_unit_amplifier
from mea1k_modules.mea1k_config_utils import get_maxlab_saving

from mea1k_connectivity_scripts.signal_helpers import extract_amplitude
from mea1k_modules.mea1k_raw_preproc import read_stim_DAC, read_raw_data
from mea1k_modules.live_postproc import LivePostproc, save_live_fig
import mea1k_modules.stim_unit_char_plots as plots

from PicoRecorder import PicoRecorder, mv_to_current_uA, DEFAULT_SENSE_GAIN

# ============================================================================================
# 1) ZERO-CURRENT DAC CODE
# Sweep the DAC code, connect the stim unit to an amplifier at each code and read
# the peak of the parasitic current transient: it changes sign at the zero-current code.
# ============================================================================================

def _warmup_connect_cycle(array, ampl_id, dac_code, break_time):
    """Unrecorded connect/disconnect of the stim unit at dac_code: the ringnode download leaves
    a transient that otherwise lands in the baseline of the first recording."""
    print(f"Warm-up cycle at DAC {dac_code} (not recorded)...")
    shift_DAC(dac_code)
    time.sleep(break_time/4)
    array.connect_amplifier_to_stimulation(ampl_id)
    array.download()
    time.sleep(break_time*2)
    array.disconnect_amplifier_from_stimulation(ampl_id)
    array.download()
    time.sleep(break_time*2)
    shift_DAC(512) # reset
    time.sleep(break_time*4)

def _sweep_DAC(dirname, array, stim_unit, ampl_id, set_id, DAC_values, stim_mode,
               debug, pico_rec=None, break_time = 0.01):
    s = get_maxlab_saving()

    print(f"Turning on stimulation unit {stim_unit}...")
    turn_on_stimulation_units([stim_unit], mode=stim_mode)
    print(f"Connecting amplifier {ampl_id} to ringnode...")
    array.connect_amplifier_to_ringnode(int(ampl_id))
    # by default connects automatically to stimluation as well, but want to to this controled in the loop
    array.disconnect_amplifier_from_stimulation(ampl_id)
    array.download()
    time.sleep(0.2)
    _warmup_connect_cycle(array, ampl_id, DAC_values[0], break_time)

    for DAC_val in (DAC_values):
        fname = f"config_StimUnit{int(stim_unit):02d}_Ampl{ampl_id:04d}_Set{set_id}_DAC{DAC_val}"
        channels_args = {'channels': [ampl_id]} if not debug else {'legacy': True}
        start_saving(s, dir_name=dirname, fname=fname, **channels_args)
        # everything up to stop_saving in try/finally: MaxWell keeps recording until saving is
        # stopped, an exception or Ctrl+C in between would leave it running (server crashes)
        try:
            # start external pico recorder if provided
            pico_fullfname = os.path.join(dirname, f"{fname}.npz")
            if pico_rec:
                pico_rec.start(pico_fullfname)
            print(f"Sweep DAC to {DAC_val}, {fname}")

            # has not effect yet, because StimUnit and ringnode are disconnected
            shift_DAC(DAC_val)
            time.sleep(break_time/4)

            print(f"Connecting StimUnit, now probing parasitic current on PI->MA pin...")
            array.connect_amplifier_to_stimulation(ampl_id)
            array.download()
            time.sleep(break_time*2)

            print(f"Disconnecting StimUnit...")
            array.disconnect_amplifier_from_stimulation(ampl_id)
            array.download()
            time.sleep(break_time*2)
            shift_DAC(512) # reset
            time.sleep(break_time*4)
        finally:
            stop_saving(s)
            if pico_rec:
                time.sleep(.3)
                pico_rec.stop()
        print('=================================')

    array.disconnect_amplifier_from_ringnode(ampl_id)
    array.download()
    turn_off_stimulation_units([stim_unit])

def _get_DAC_candidate_values(centered_around, delta, n_values=32):
    # emforce max 32 values
    resolution = max(1, int((2*delta) / n_values))
    DAC_candidates = np.arange(centered_around-delta, centered_around+delta, resolution)
    # ensure between 0 and 1023
    if DAC_candidates[0] < 0 or DAC_candidates[-1] > 1023:
        print("Warning! DAC values out of range, shifting into range")
        shift_lower_end_by, shift_upper_end_by = 0, 0
        for i in range(len(DAC_candidates)):
            if DAC_candidates[i] < 0:
                DAC_candidates[i] += 0 + shift_lower_end_by
                shift_lower_end_by += 32
            if DAC_candidates[i] > 1023:
                DAC_candidates[i] = 1023 - shift_upper_end_by
                shift_upper_end_by += 32
    return DAC_candidates.astype(int).tolist()

def _connect_edge(x, start, stop, w=40, frac=0.5):
    """Index of the stim unit connect in x[start:stop]: the first step larger than frac of the
    largest step there. A step is the difference between the mean of the next w and the
    previous w samples, robust to single noise spikes. None if x is flat there."""
    c = np.r_[0, np.cumsum(x)]
    t = np.arange(max(start, w), min(stop, len(x) - w))
    step = np.abs((c[t + w] - c[t]) / w - (c[t] - c[t - w]) / w)
    above = step > frac * step.max()
    if not above.any():   # flat trace, no edge
        return None
    first = int(np.argmax(above))
    # the edge is the largest step within this first run above the threshold
    run_end = first + int(np.argmin(above[first:])) if not above[first:].all() else len(above)
    return int(t[first + np.argmax(step[first:run_end])])

def _extract_DAC_transient_set(dirname, ampl_id, stim_unit, set_id, gain, debug=False,
                               median_win=21, current_smooth_win=25, baseline_n=500,
                               connect_search_to=1200):
    """Peak of the amplifier transient at the stim unit connect, for each recording of one DAC
    sweep set. gain: set on the chip, used for the uV conversion instead of the gain in the file.
    baseline_n: first samples, before the connect, for baseline and noise. The connect is the
    first large edge between baseline_n and connect_search_to: connects fall at ~550-1100
    (varies by ~200 samples between recordings), disconnects at >~1400, which is excluded so
    that a tiny connect step near the zero-current code isn't mistaken for the disconnect.
    median_win: samples (~1 ms) of the median filter on the amplifier trace, keeps the edge and
    peak height, removes the noise burst after the connect (0: unfiltered).
    current_smooth_win: samples (1.25 ms) of the moving average of the pico current."""
    prefix = f"config_StimUnit{stim_unit:02d}_Ampl{ampl_id:04d}_Set{set_id}_DAC"
    fnames = sorted(f for f in os.listdir(dirname) if f.startswith(prefix) and f.endswith(".raw.h5"))
    dac_transients_res, debug_traces = [], []
    for fname in fnames:
        DAC_val = int(re.search(r"_DAC(\d+)", fname).group(1))
        print(f"\nProcessing {fname}")
        data = read_raw_data(dirname, fname, convert2uV=True,
                             subtract_dc_offset=False, override_gain=gain)
        if data.shape[1]  < 1300:
            print("Warning: File was too short, skipping")
            continue
        # normal mode records only this amplifier (compressed, no DAC), debug mode all channels (legacy)
        all_channels = data.shape[0] > 1
        dac_trace = read_stim_DAC(dirname, fname) if all_channels else None
        amplifier_trace = (data[ampl_id] if all_channels else data[0]).astype(float)
        if median_win:
            amplifier_trace = median_filter(amplifier_trace, size=median_win, mode='nearest')
        baseline = np.median(amplifier_trace[:baseline_n])
        amplifier_trace -= baseline
        noise = amplifier_trace[:baseline_n].std()

        rise_at = _connect_edge(amplifier_trace, start=baseline_n, stop=connect_search_to)
        # readout window: up to 350 samples (17.5 ms) after the rise, clear of the disconnect
        if rise_at is None:
            # completely flat: no current at this DAC code (the zero-current code), plateau ~0
            print(f"No connect edge in {fname}, flat trace, reading the plateau after the baseline")
            rise_at = baseline_n
        if rise_at + 350 > len(amplifier_trace):
            print(f"Warning: connect at {rise_at}, too late for the readout window, skipping")
            continue
        # Peak in the direction of the plateau; an overshoot only counts if it clearly exceeds
        # the baseline noise, otherwise (small steps) a noise maximum would be picked -> read
        # the plateau at a fixed delay instead
        plateau = np.median(amplifier_trace[rise_at + 150: rise_at + 300])
        win = amplifier_trace[rise_at: rise_at + 350] * np.sign(plateau)
        is_overshoot = win.max() - abs(plateau) > 5 * noise
        peak_at = rise_at + (np.argmax(win) if is_overshoot else 225)
        peak = amplifier_trace[peak_at]
        print(f"Baseline: {baseline:.1f}, Peak: {peak:.1f}, DAC_val: {DAC_val}, connect at {rise_at}, peak at {peak_at}")

        # try to read external picoscope recording (full trace, not windowed)
        pico_fullfname = os.path.join(dirname, fname.replace('.raw.h5', '.npz'))
        if os.path.exists(pico_fullfname):
            pico_data = np.load(pico_fullfname)
            sensed_current_uA = mv_to_current_uA(pico_data['mv'], DEFAULT_SENSE_GAIN)
        else:
            sensed_current_uA = np.full_like(amplifier_trace, np.nan)
        # window-averaged current, the amplitude is read from this
        sensed_current_smooth_uA = uniform_filter1d(sensed_current_uA, current_smooth_win, mode='nearest')
        # pico starts/stops with a lag and saves whole 500-sample chunks, its trace can be
        # shorter than the MEA1K trace
        if peak_at < len(sensed_current_uA):
            current_at_peak = sensed_current_smooth_uA[peak_at]
        else:
            current_at_peak = np.nan
            print(f"Warning: Pico trace too short ({len(sensed_current_uA)} < {peak_at}) for {fname}")

        dac_transients_res.append(pd.Series({
            "stimunit_id": stim_unit,
            "ampl_id": ampl_id,
            "set_id": set_id,
            "gain": gain,
            "DAC_val": DAC_val,
            "peak_uV": peak,
            "peak_at": peak_at,
            "connect_at": rise_at,
            "baseline": baseline,
            "sensed_current_uA_at_peak": current_at_peak,
        }))
        if debug:
            debug_traces.append(dict(DAC_val=DAC_val, amplifier_trace=amplifier_trace, dac_trace=dac_trace,
                                     peak=peak, peak_at=peak_at, connect_at=rise_at,
                                     is_overshoot=is_overshoot, current_uA=sensed_current_uA,
                                     current_smooth_uA=sensed_current_smooth_uA,
                                     current_at_peak=current_at_peak))
    if debug_traces:
        fig = plots.dac_transients(debug_traces, f"Raw data of {prefix}*", baseline_n, connect_search_to)
        save_live_fig(fig, 'parasitic_DACcur_live', dpi=400)
        plt.close(fig)

    dac_transients_res = pd.DataFrame(dac_transients_res)
    fullfname = os.path.join(dirname, f"results_StimUnit{stim_unit:02d}_Ampl{ampl_id:04d}_Set{set_id}.csv")
    dac_transients_res.to_csv(fullfname, index=False)
    return dac_transients_res

def _sigmoid(x, L, x0, k, b):
    # curve_fit tries steep slopes, exp overflows to inf there and L/inf = 0 is the right limit
    with np.errstate(over='ignore'):
        return L / (1 + np.exp(-k * (x - x0))) + b

def _fit_zero_current_dac(df):
    """Sigmoid fit of peak_uV over DAC_val. None if there are too few points or the fit fails,
    otherwise: the zero-current DAC code (where the fit crosses 0, else the measured point
    closest to 0; None if outside the swept DAC range) and the fit's prediction there, R², the
    measured point closest to 0 and the fit curve (for plotting)."""
    df = df.dropna(subset=["DAC_val", "peak_uV"])
    if len(df) < 4 or df["DAC_val"].nunique() < 4:
        return None
    x = df["DAC_val"].to_numpy(dtype=float)
    y = df["peak_uV"].to_numpy(dtype=float)

    p0 = [np.max(y) - np.min(y), np.median(x),
          0.01 if (y[-1] > y[0]) == (x[-1] > x[0]) else -0.01, np.min(y)]
    try:
        popt, _ = curve_fit(_sigmoid, x, y, p0=p0, maxfev=10000)
    except Exception as e:
        print(f"Sigmoid fit failed: {e}")
        return None
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = np.nan if ss_tot == 0 else 1 - np.sum((y - _sigmoid(x, *popt)) ** 2) / ss_tot
    i_closest = np.argmin(np.abs(y))

    L, x0, k, b = popt
    zero_current_dac = zero_current_peak = None
    if k != 0:
        val = -L / b - 1
        x_cross = x0 - (1 / k) * np.log(val) if val > 0 else x[i_closest]
        zero_current_dac = int(np.clip(round(x_cross), 0, 1023))
        zero_current_peak = _sigmoid(zero_current_dac, *popt)
        if not x.min() <= zero_current_dac <= x.max():
            print(f"Warning: fit crosses 0 at DAC {zero_current_dac}, outside the swept "
                  f"{x.min():.0f}-{x.max():.0f}, not used")
            zero_current_dac = zero_current_peak = None
    x_line = np.linspace(x.min(), x.max(), 100)
    return dict(zero_current_dac=zero_current_dac, zero_current_peak=zero_current_peak,
                r_squared=r_squared, emp_dac=x[i_closest], emp_peak=y[i_closest],
                curve=(x_line, _sigmoid(x_line, *popt)))

def _update_characterization_csv(dirname, stimunit_id, zero_current_dac):
    """Set the stim unit's zero-current DAC code in processed/StimUnits_characterization.csv"""
    csv_path = os.path.join(dirname, "processed", "StimUnits_characterization.csv")
    if os.path.exists(csv_path):
        df_char = pd.read_csv(csv_path)
    else:
        df_char = pd.DataFrame(columns=["stimunit_id", "zero_current_DAC_code"])
    if stimunit_id in df_char["stimunit_id"].values:
        df_char.loc[df_char["stimunit_id"] == stimunit_id, "zero_current_DAC_code"] = zero_current_dac
    else:
        new_row = pd.DataFrame([{"stimunit_id": stimunit_id, "zero_current_DAC_code": zero_current_dac}])
        df_char = pd.concat([df_char, new_row], ignore_index=True)
    df_char = df_char.astype(int).sort_values("stimunit_id")
    df_char.to_csv(csv_path, index=False)

def eval_stim_unit_results(dirname, stim_unit_results, debug=False):
    """Zero-current DAC code of a stim unit from all its sets so far (sigmoid fit), written to
    processed/StimUnits_characterization.csv, plus its characterization plot (also live)."""
    if not stim_unit_results:
        return
    stimunit_id = int(stim_unit_results[0]["stimunit_id"].iloc[0])
    fit_all = _fit_zero_current_dac(pd.concat(stim_unit_results, axis=0, ignore_index=True))
    fit_last = _fit_zero_current_dac(stim_unit_results[-1])

    fig = plots.stim_unit_characterization(stim_unit_results, fit_all, fit_last, stimunit_id)
    os.makedirs(os.path.join(dirname, "processed"), exist_ok=True)
    save_live_fig(fig, 'stim_unit_characterization_live', dpi=300)
    if debug:
        save_live_fig(fig, 'StimUnit_characterization_debug', dpi=300)
    fig.savefig(os.path.join(dirname, "processed", f"StimUnit{stimunit_id:02d}_characterization.png"), dpi=300)
    plt.close(fig)

    if fit_all is not None and fit_all["zero_current_dac"] is not None:
        _update_characterization_csv(dirname, stimunit_id, fit_all["zero_current_dac"])

def _next_DAC_center(results, stim_unit_results, delta, centered_around):
    """Center of the next set: zero crossing of peak_uV between neighbouring DAC values.
    Without a sign change the whole set sits on a saturated plateau, then the next set
    is placed directly next to it, on the side where the sign flips.
    Returns (center, found_crossing)."""
    r = results.dropna(subset=["DAC_val", "peak_uV"]).sort_values("DAC_val")
    if r.empty:
        return centered_around, False
    dac, peak = r.DAC_val.to_numpy(dtype=float), r.peak_uV.to_numpy(dtype=float)
    flips = np.flatnonzero(np.sign(peak[:-1]) != np.sign(peak[1:]))
    if len(flips):
        # several flips means outliers, the real crossing is the largest jump
        i = flips[np.argmax(np.abs(np.diff(peak)[flips]))]
        center = dac[i] + (dac[i+1]-dac[i]) * peak[i] / (peak[i]-peak[i+1])
        return int(round(center)), True

    # slope from all sets of this stim unit once both signs were seen, otherwise the usual:
    # positive peaks at low DAC values
    slope = -1
    all_res = pd.concat(stim_unit_results)
    if (all_res.peak_uV > 0).any() and (all_res.peak_uV < 0).any():
        slope = np.sign(np.corrcoef(all_res.DAC_val, all_res.peak_uV)[0, 1]) or -1
    go_up = np.sign(np.median(peak)) != slope
    center = dac.max() + delta if go_up else dac.min() - delta
    return int(np.clip(center, delta, 1024 - delta)), False

def char_stim_units(dirname, stim_mode, n_amplifiers=2, stim_units=list(range(32)), debug=False,
                    gain=7, start_centered_around=512, delta=256, set_size=32, max_iterations=5,
                    down_step_size=4, rec_current_externally=False):
    """Zero-current DAC code per stim unit, by sweeping the DAC around a start value.
    start_centered_around: one DAC code for all stim units, or per stim unit (dict/Series,
    e.g. the headstage's current codes) for a quick recalibration with a small delta. With
    per-unit start codes, all_StimUnit_DAC_sweeps.png compares old and new codes."""
    # create processed directory if not exists
    os.makedirs(os.path.join(dirname, "processed"), exist_ok=True)
    array = setup_stim_unit_characterization(dirname, gain=gain) 
        
    if rec_current_externally:
        pico_rec = PicoRecorder(voltage_range="5V", resolution="16BIT", sample_interval_us=50)
    else:
        pico_rec = None
    
    per_unit_start = not np.isscalar(start_centered_around)
    for stim_unit in stim_units:
        plt.close('all')
        start = int(start_centered_around[stim_unit] if per_unit_start else start_centered_around)
        stim_unit_results = []
        best_center = None
        used_ampl_ids = []
        for which_amplifier in range(n_amplifiers):
            # iterate random mea1k el until we find the right stim unit + ampliifier
            ampl_id = find_stim_unit_amplifier(array, stim_unit=stim_unit,
                                               exclude=used_ampl_ids)
            used_ampl_ids.append(ampl_id)

            set_id = 0
            if best_center is None:
                centered_around, used_delta = start, delta
            else:
                # zero-current code belongs to the stim unit, new amplifier starts around the
                # previous amplifier's crossing
                centered_around, used_delta = best_center, max(16, delta//down_step_size)
            while True:
                print(f"\n==========={centered_around}+-{used_delta}================")
                DAC_values = _get_DAC_candidate_values(centered_around, used_delta, 
                                                       n_values=set_size)
                _sweep_DAC(dirname, array, stim_unit, ampl_id, set_id, DAC_values, 
                           stim_mode, debug=debug, pico_rec=pico_rec)
                results = _extract_DAC_transient_set(dirname, ampl_id, stim_unit, 
                                                    set_id, gain, debug=debug)
                stim_unit_results.append(results)
                
                # Pass dirname so it can save the CSV at every execution reliably
                eval_stim_unit_results(dirname, stim_unit_results, debug=debug)

                centered_around, found_crossing = _next_DAC_center(results, stim_unit_results,
                                                                    used_delta, centered_around)
                if found_crossing:
                    best_center = centered_around
                    print(f"Stim unit {stim_unit} amplifier {ampl_id} set {set_id}: "
                          f"sign change at DAC ~{centered_around}")
                    if used_delta == 16:
                        print("Delta is already 16, stopping here.\n\n\n")
                        break
                    used_delta = max(16, used_delta//down_step_size) # should get smaaller while closing in on the best value
                else:
                    print(f"Stim unit {stim_unit} amplifier {ampl_id} set {set_id}: no sign "
                          f"change, all points saturated, moving towards the crossing")
                print(f"Recentering around {centered_around}, new delta={used_delta}")
                set_id += 1
                if set_id >= max_iterations:
                    print(f"Reached maximum iterations ({max_iterations}), stopping here.\n\n\n")
                    break
    if per_unit_start:
        _plot_recalibration(dirname, stim_units, pd.Series(start_centered_around))

def _plot_recalibration(dirname, stim_units, old_codes):
    """All DAC sweeps with the old (start) and new zero-current DAC codes per stim unit."""
    new_codes = pd.read_csv(os.path.join(dirname, "processed", "StimUnits_characterization.csv"),
                            index_col=0).zero_current_DAC_code
    results = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(dirname, "results_StimUnit*.csv"))
                         if os.path.getsize(f) > 1])   # sets without any valid transient are empty
    fig = plots.all_dac_sweeps(results, stim_units, old_codes=old_codes, new_codes=new_codes)
    fig.suptitle(os.path.basename(dirname), fontsize=16, fontweight="bold")
    plot_fullfname = os.path.join(dirname, "processed", "all_StimUnit_DAC_sweeps.png")
    fig.savefig(plot_fullfname, dpi=300)
    plt.close(fig)
    print(plot_fullfname)



# ============================================================================================
# 2) CURRENT LSB
# A sine around each stim unit's zero-current code through a known resistor R; the
# amplifier amplitude gives the current per DAC unit. Evaluated in a 2nd thread.
# ============================================================================================

def _record_sine(dirname, array, stim_unit, ampl_id, fname, dac_code, sine_seq, seq_duration,
                 stim_mode, pico_rec=None, break_time=0.01, settle_time=0.1):
    """Chip sequence mirrors _sweep_DAC: stim unit powered and ringnode connected before the
    recording, unrecorded warm-up connect/disconnect, then a single recorded connection
    during which the sine is played around dac_code."""
    s = get_maxlab_saving()

    print(f"Turning on stimulation unit {stim_unit}...")
    turn_on_stimulation_units([stim_unit], mode=stim_mode)
    print(f"Connecting amplifier {ampl_id} to ringnode...")
    array.connect_amplifier_to_ringnode(int(ampl_id))
    # by default connects automatically to stimluation as well, but want to to this controled below
    array.disconnect_amplifier_from_stimulation(ampl_id)
    array.download()
    time.sleep(0.2)

    _warmup_connect_cycle(array, ampl_id, dac_code, break_time)

    start_saving(s, dir_name=dirname, fname=fname, legacy=True)
    # everything up to stop_saving in try/finally: MaxWell keeps writing (~40 MB/s in
    # legacy mode) until saving is stopped, so an exception in between fills the disk
    try:
        # start external pico recorder if provided
        pico_fullfname = os.path.join(dirname, f"{fname}.npz")
        if pico_rec:
            pico_rec.start(pico_fullfname)

        # has not effect yet, because StimUnit and ringnode are disconnected
        shift_DAC(dac_code)
        time.sleep(break_time/4)

        print(f"Connecting StimUnit at zero current DAC {dac_code}...")
        array.connect_amplifier_to_stimulation(ampl_id)
        array.download()
        time.sleep(settle_time)

        print(f"Stimulating ~~~~~~~ ({seq_duration:.1f}s)")
        sine_seq.send()
        time.sleep(seq_duration)

        print(f"Disconnecting StimUnit...")
        array.disconnect_amplifier_from_stimulation(ampl_id)
        array.download()
        time.sleep(break_time*2)
        shift_DAC(512) # reset
        time.sleep(break_time*4)
    finally:
        stop_saving(s)
        if pico_rec:
            time.sleep(.3)
            try:
                pico_rec.stop()
            except Exception as exc:
                print(f"WARNING! Could not stop the pico recorder: {exc}")

    array.disconnect_amplifier_from_ringnode(ampl_id)
    array.download()
    turn_off_stimulation_units([stim_unit])
    print('=================================')

def _remove_existing_lsb_files(dirname, tag):
    """MaxWell appends _0, _1, .. instead of overwriting, so earlier recordings of this kind
    (h5, pico npz, marker csv) are deleted before re-recording."""
    existing = [f for f in os.listdir(dirname) if re.search(re.escape(tag) + r"(_\d+)?\.(raw\.h5|npz|csv)$", f)]
    for f in existing:
        os.remove(os.path.join(dirname, f))
    if existing:
        print(f"Overwriting: deleted {len(existing)} '*{tag}' files in {dirname}")

def _char_current_lsb_loop(dirname, array, dac_settings, stim_units, n_amplifiers, tag,
                           sine_ampl_DAC_units, freq, ncycles, stim_mode, dac_code_offset,
                           gain, pico_rec, R, pp, settle_time=0.1, ampl_cache=None):
    """ampl_cache: optional {stim_unit: [ampl_id, ..]}, filled on first use and reused on later
    calls, so every condition of a sweep is measured on the same amplifiers."""
    for stim_unit in stim_units:
        print(dac_settings.loc[dac_settings.stimunit_id==stim_unit])
        dac_code = int(dac_settings.loc[dac_settings.stimunit_id==stim_unit].zero_current_DAC_code)
        if dac_code_offset is not None:
            dac_code += dac_code_offset
        print("Creating sine wave sequence...", end=" ", flush=True)
        sine_seq = create_stim_sine_sequence(dac_id=0, amplitude=sine_ampl_DAC_units,
                                                f=freq, ncycles=ncycles,
                                                center_around=dac_code)
        print("Done.")

        seq_duration = (ncycles / freq) + .5  # add .5s buffer
        used_ampl_ids = []
        for i in range(n_amplifiers):
            cached = ampl_cache.get(stim_unit, []) if ampl_cache is not None else []
            if i < len(cached):
                ampl_id = cached[i]
            else:
                ampl_id = find_stim_unit_amplifier(array, stim_unit=stim_unit, exclude=used_ampl_ids)
                if ampl_cache is not None:
                    ampl_cache.setdefault(stim_unit, []).append(ampl_id)
            used_ampl_ids.append(ampl_id)

            print(f"\n\nStim unit {stim_unit} amplifier {ampl_id} ({i+1}/ "
                  f"{n_amplifiers}), zero current DAC code: {dac_code}")
            print(f"[postproc] {pp.status()}", flush=True)

            fname = f"config_StimUnit{int(stim_unit):02d}_Ampl{ampl_id:04d}{tag}"
            _record_sine(dirname, array, stim_unit, ampl_id, fname, dac_code, sine_seq,
                         seq_duration, stim_mode, pico_rec=pico_rec, settle_time=settle_time)
            # written after stop_saving + pico stop -> "recording finished" signal for postproc
            pd.Series({
                "stimunit_id": stim_unit, "ampl_id": ampl_id, "dac_code": dac_code,
                "dac_code_offset": dac_code_offset or 0, "gain": gain,
                "sine_amplitude_DAC_units": sine_ampl_DAC_units, "freq": freq,
                "ncycles": ncycles, "stim_mode": stim_mode, "resistance": R,
            }).to_frame().T.to_csv(os.path.join(dirname, f"{fname}.csv"), index=False)
            pp.add(f"{fname}.raw.h5")

def make_lsb_conditions(gains, freqs, sine_ampl_DAC_units, ncycles_by_freq, ref_gain=7):
    """Grid of (gain x freq) LSB measurements. sine_ampl_DAC_units is the amplitude at ref_gain,
    scaled by ref_gain/gain (min. 1 DAC unit) because the recording full scale shrinks
    with gain. Pass a {gain: amplitude} dict to set them explicitly."""
    conditions = []
    for gain in gains:
        if isinstance(sine_ampl_DAC_units, dict):
            ampl = sine_ampl_DAC_units[gain]
        else:
            ampl = max(1, round(sine_ampl_DAC_units * ref_gain / gain))
        for freq in freqs:
            conditions.append(dict(gain=gain, freq=freq, ncycles=ncycles_by_freq[freq],
                                   sine_ampl_DAC_units=ampl))
    return conditions

def char_current_lsb(dirname, R, stim_units, n_amplifiers, stim_mode, conditions,
                     subfolder=None, rec_current_externally=False, skip_stimulation=False,
                     reprocess=False, fallback_gain=7, fallback_sine_ampl_DAC_units=None,
                     debug=False, plot_ylims='default'):
    """Current LSB of the stim units: a sine around each unit's zero current DAC code (from
    dirname/processed/StimUnits_characterization.csv, char_stim_units) through R.
    conditions: list of dict(gain, freq, ncycles, sine_ampl_DAC_units[, dac_code_offset]),
    one condition for the standard LSB, several for a sweep (see make_lsb_conditions).
    The same amplifiers are used in every condition. Recordings and processed/ go into
    dirname/subfolder (or dirname). Each recording is evaluated in a 2nd thread, the
    aggregated csv is updated after every stim unit. skip_stimulation: only (re)evaluate
    the recordings there; fallback_* are for old recordings without gain/amplitude info."""
    rec_dir = os.path.join(dirname, subfolder) if subfolder else dirname
    os.makedirs(os.path.join(rec_dir, "processed"), exist_ok=True)
    if plot_ylims == 'default':
        plot_ylims = LSB_PLOT_YLIMS.get(stim_mode)

    process_fn = functools.partial(process_lsb_recording, R=R, stim_mode=stim_mode,
                                   fallback_gain=fallback_gain,
                                   fallback_sine_ampl_DAC_units=fallback_sine_ampl_DAC_units,
                                   plot_ylims=plot_ylims, debug=debug)
    summarize_fn = functools.partial(summarize_lsb, stim_mode=stim_mode)
    pp = LivePostproc(rec_dir, process_fn, summarize_fn, name="postproc_char_current_lsb",
                      reprocess=reprocess, match="_CurrentLSB")
    pico_rec = None
    if not skip_stimulation:
        for c in conditions:
            offset = c.get('dac_code_offset')
            c['tag'] = (f"_f{c['freq']:04d}Hz_Gain{c['gain']}_Sine{c['sine_ampl_DAC_units']}"
                        + (f"_DACoffset{offset}" if offset is not None else "") + "_CurrentLSB")
            _remove_existing_lsb_files(rec_dir, c['tag'])
        dac_settings = pd.read_csv(os.path.join(dirname, "processed", "StimUnits_characterization.csv"))
        if rec_current_externally:
            pico_rec = PicoRecorder(voltage_range="5V", resolution="16BIT", sample_interval_us=50)

    pp.start()
    try:
        ampl_cache, array, cur_gain = {}, None, None
        for i, c in enumerate(conditions if not skip_stimulation else []):
            print(f"\n######## Condition {i+1}/{len(conditions)}: gain {c['gain']}, {c['freq']} Hz, "
                  f"{c['ncycles']} cycles, sine ±{c['sine_ampl_DAC_units']} DAC units")
            if c['gain'] != cur_gain:
                array = setup_stim_unit_characterization(rec_dir, gain=c['gain'])
                cur_gain = c['gain']
            _char_current_lsb_loop(rec_dir, array, dac_settings, stim_units, n_amplifiers, c['tag'],
                                   c['sine_ampl_DAC_units'], c['freq'], c['ncycles'], stim_mode,
                                   c.get('dac_code_offset'), c['gain'], pico_rec, R, pp,
                                   settle_time=max(0.1, 2/c['freq']), ampl_cache=ampl_cache)
            # aggregated csv, in the postproc thread after this condition's recordings
            pp.run_in_thread(lambda: _save_lsb_results(rec_dir, pp.results_df(), stim_mode,
                                                       sweep=len(conditions) > 1 or None))
    finally:
        results = pp.finish()
        if pico_rec is not None:
            pico_rec.close()
    if len(results):
        result_fullfname = _save_lsb_results(rec_dir, results, stim_mode)
        aggr = pd.read_csv(result_fullfname)
        if aggr.groupby(['gain', 'freq']).ngroups > 1:
            fig = plots.lsb_sweep_summary(aggr, R, stim_mode)
        else:
            fig = plots.lsb_summary(aggr, R, sorted(aggr.stimunit_id.unique()), stim_mode)
        fig.savefig(result_fullfname.replace(".csv", ".png"), dpi=300)
        plt.close(fig)
    return results

# fixed y-limits of the live debug plots, per stim mode (None -> auto-scaled)
# v: amplifier voltage [uV], i: picoscope current through R [uA]; *_raw: unprocessed panel
LSB_PLOT_YLIMS = {
    'large_current': dict(v=(-200_000, 200_000), v_raw=(-300_000, 300_000),
                          i=(-0.25, 0.25), i_raw=(-0.3, 0.3)),
    'small_current': None,
}

def _sine_onset_offset(dac, center, sine_ampl_DAC_units, max_gap=20):
    """First/last DAC step of the sine. Only steps that stay within center ± sine amplitude
    and have another step within max_gap samples (one 1 kHz period) count, so isolated
    shift_DAC jumps (512 -> center, center -> 512) are ignored even when center is near 512."""
    in_sine = np.abs(dac.astype(int) - center) <= sine_ampl_DAC_units
    steps = np.nonzero((np.diff(dac) != 0) & in_sine[:-1] & in_sine[1:])[0]
    gaps = np.diff(steps)
    dense = np.r_[False, gaps <= max_gap] | np.r_[gaps <= max_gap, False]
    steps = steps[dense]
    return int(steps[0]), int(steps[-1])

def _eval_lsb_file(dirname, fname, stimunit_id, ampl_id, freq, sine_amplitude_DAC_units, gain,
                   dac_code_offset, R, stim_mode, debug=False, plot_ylims=None, n_zoom_cycles=5,
                   log=print):
    """Lock-in amplitude of the amplifier voltage and (if recorded) the picoscope current
    for one sine LSB recording. Returns one result row. gain: set on the chip during the
    recording, used instead of the gain in the file."""
    plot_ylims = plot_ylims or {}
    file_freq, sine_ampl_DAC_units = freq, sine_amplitude_DAC_units
    log(f"Processing {fname} ampl={ampl_id} stimunit={stimunit_id}")
    data = read_raw_data(dirname, fname, convert2uV=True, row_slice=[ampl_id], override_gain=gain)
    dac = read_stim_DAC(dirname, fname)

    zero_current_DAC_code = int(np.median(dac)) # sine is symmetric around the center code
    # DAC steps of a slow sine are further apart: allow up to one period between steps
    onset, offset = _sine_onset_offset(dac, zero_current_DAC_code, sine_ampl_DAC_units,
                                       max_gap=max(20, 20_000//file_freq))
    rng = max(1, file_freq//100) # allow 1% frequency deviation
    # analysis window = sine onset/offset from the DAC; notch + demean on by default
    mean_ampl, phase_shift, _, _ = extract_amplitude(
        data[0].astype(float),
        sampling_rate=20_000,
        highpass=min(100, file_freq/5), # 100 Hz at 1 kHz, must stay below the stim frequency
        notch=True,
        min_band=file_freq-rng, max_band=file_freq+rng,
        zoom_from=-150, zoom_to=150,
        dac=dac.astype(float),
        estimate_amplitude_from=onset,
        estimate_amplitude_to=offset,
        name=fname.replace('.raw.h5', ''),
        plot_fname='debug_amplifier',
        ylim=plot_ylims.get('v'), raw_ylim=plot_ylims.get('v_raw'), unit='uV',
        debug=debug,
    )

    # picoscope current through R, recorded when rec_current_externally=True
    # MaxWell appends _0, _1, .. on name clashes, the pico .npz is overwritten instead:
    # strip the suffix, and only accept the .npz if it was written right after this h5
    pico_fullfname = os.path.join(dirname, re.sub(r'(_\d+)?\.raw\.h5$', '.npz', fname))
    if os.path.exists(pico_fullfname) and abs(os.path.getmtime(pico_fullfname) -
                                              os.path.getmtime(os.path.join(dirname, fname))) < 10:
        pico_data = np.load(pico_fullfname)
        pico_current_uA = mv_to_current_uA(pico_data['mv'], DEFAULT_SENSE_GAIN)
        # pico and MaxWell recordings are not sample-synchronized: use the DAC sine
        # window from the h5, trimmed by 50 ms on each side to absorb the start latency
        ext_current_uA, _, ext_baseline_dc, ext_sine_dc = extract_amplitude(
            pico_current_uA,
            sampling_rate=20_000,
            min_band=file_freq-rng, max_band=file_freq+rng,
            estimate_amplitude_from=onset + 1_000,
            estimate_amplitude_to=offset - 1_000,
            lowpass=3000.0,
            # zoom on a few single cycles at the start of the analysis window
            zoom_from=0, zoom_to=n_zoom_cycles * 20_000 // file_freq,
            name=f'StimUnit={stimunit_id}, DAC={zero_current_DAC_code}, ampl={ampl_id}',
            ylim=plot_ylims.get('i'), raw_ylim=plot_ylims.get('i_raw'), unit='uA',
            debug=debug,
            plot_fname='debug_picoscope_signal',
            save_to_dir=os.path.join(dirname, "processed"),
        )
    else:
        ext_current_uA, ext_baseline_dc, ext_sine_dc = np.nan, np.nan, np.nan

    return {
        "stimunit_id": stimunit_id,
        "ampl_id": ampl_id,
        "gain": gain,
        "mean_amplitude_uV": mean_ampl,
        "dac_code_offset": dac_code_offset,
        f"{stim_mode}_zero_current_DAC": zero_current_DAC_code,
        "resistance": R,
        "sine_amplitude_DAC_units": sine_ampl_DAC_units,
        f"{stim_mode}_LSB_nA": (mean_ampl / R) / (sine_ampl_DAC_units / 1_000),  # in nA
        "phase_shift_deg": phase_shift,
        "freq": file_freq,
        "external_current_uA": ext_current_uA,
        f"{stim_mode}_external_LSB_nA": ext_current_uA * 1_000 / sine_ampl_DAC_units,
        "external_current_pre_DC_offset": ext_baseline_dc,
        "external_current_DC_offset": ext_sine_dc,
    }

def _lsb_params_from_fname(fname):
    """Recording parameters from the file name, for recordings without a marker csv."""
    return {
        "stimunit_id": int(fname.split("_")[1].replace("StimUnit", "")),
        "ampl_id": int(fname.split("_")[2].replace("Ampl", "")),
        "freq": int(re.search(r'f(\d{4})Hz', fname).group(1)),
        "dac_code_offset": int(m.group(1)) if (m := re.search(r'DACoffset(-?\d+)', fname)) else 0,
        "gain": float(m.group(1)) if (m := re.search(r'_Gain(\d+)', fname)) else None,
        "sine_amplitude_DAC_units": int(m.group(1)) if (m := re.search(r'_Sine(\d+)', fname)) else None,
    }

def process_lsb_recording(dirname, fname, log, R, stim_mode, fallback_gain=None,
                          fallback_sine_ampl_DAC_units=None, plot_ylims=None, debug=False):
    """process_fn for LivePostproc: one sine LSB recording -> one result row. Parameters come
    from the marker csv; recordings without one (or without gain in it) from the file name
    and the fallback_* values."""
    marker_fullfname = os.path.join(dirname, fname.replace(".raw.h5", ".csv"))
    params = _lsb_params_from_fname(fname)
    if os.path.exists(marker_fullfname):
        marker = pd.read_csv(marker_fullfname).iloc[0]
        params.update({k: marker[k] for k in params if k in marker and pd.notna(marker[k])})
    elif params["gain"] is not None:
        # recordings with the gain in their name always get a marker csv, unless interrupted
        log(f"{fname}: no marker csv, recording was interrupted?")
        return None
    if params["gain"] is None:
        params["gain"] = fallback_gain
    if params["sine_amplitude_DAC_units"] is None:
        params["sine_amplitude_DAC_units"] = fallback_sine_ampl_DAC_units
    if params["gain"] is None or params["sine_amplitude_DAC_units"] is None:
        log(f"{fname}: gain or sine amplitude unknown, pass fallback_gain/fallback_sine_ampl_DAC_units")
        return None
    row = _eval_lsb_file(dirname, fname, R=R, stim_mode=stim_mode, plot_ylims=plot_ylims,
                         debug=debug, log=log, **params)
    return pd.DataFrame([row])

def summarize_lsb(results, last_result, stim_mode):
    if last_result is None:
        return ""
    last = last_result.iloc[0]
    return (f"last StimUnit{int(last.stimunit_id):02d} Ampl{int(last.ampl_id)}: "
            f"LSB {last[f'{stim_mode}_LSB_nA']:.2f} nA")

def _save_lsb_results(dirname, results, stim_mode, sweep=None):
    """Aggregate per-recording results (median per stim unit and condition) and write the LSB
    csv: {stim_mode}_lsb_characterization.csv, or _lsb_sweep.csv for several conditions
    (sweep=None: decided from the results)."""
    aggr = pd.DataFrame(results)
    group_by = ['stimunit_id', 'gain', 'freq', 'sine_amplitude_DAC_units', 'dac_code_offset']
    aggr['median_amplitude_uV'] = aggr.groupby(group_by)['mean_amplitude_uV'].transform('median')
    aggr[f'{stim_mode}_median_LSB_nA'] = aggr.groupby(group_by)[f"{stim_mode}_LSB_nA"].transform('median')
    if sweep is None:
        sweep = aggr.groupby(['gain', 'freq']).ngroups > 1
    name = "lsb_sweep" if sweep else "lsb_characterization"
    result_fullfname = os.path.join(dirname, "processed", f"{stim_mode}_{name}.csv")
    os.makedirs(os.path.dirname(result_fullfname), exist_ok=True)
    aggr.to_csv(result_fullfname, index=False)
    return result_fullfname



# ============================================================================================
# 3) HELPERS
# Copy results to the headstage's stim unit characterization csv.
# ============================================================================================

def transfer_stimunit_characterization(dirname, stim_mode, headstage_name, incl_LSB=False):
    # read in final characterization CSV (falls back to drift-check output, no LSB there)
    result_fullfname = os.path.join(dirname, "processed", f"{stim_mode}_lsb_characterization.csv")
    if not os.path.exists(result_fullfname):
        result_fullfname = os.path.join(dirname, "processed", "StimUnits_characterization.csv")
    if not os.path.exists(result_fullfname):
        print(f"Final characterization CSV not found: {result_fullfname}")
        return
    df_final = pd.read_csv(result_fullfname).rename(columns={'zero_current_DAC_code': f"{stim_mode}_zero_current_DAC"})
    df_final = df_final[['stimunit_id', f"{stim_mode}_zero_current_DAC"] + ([f"{stim_mode}_median_LSB_nA"] if incl_LSB else [])].drop_duplicates('stimunit_id').sort_values('stimunit_id').reset_index(drop=True)
    df_final = df_final.set_index('stimunit_id')

    csv_fullfname = os.path.join(device_paths()[0], "devices/headstage_devices", headstage_name,
                                 f"{headstage_name}_stimunit_characterization.csv")
    if os.path.exists(csv_fullfname):
        stim_settings = get_headstage_stim_settings(headstage_name)
        for col in df_final.columns:
            if col in stim_settings.columns:
                print(f"Transferring new {col} values to {headstage_name} stimunit "
                      f"characterization CSV. Diff:\n{df_final[col] - stim_settings[col]}")
            stim_settings.loc[df_final.index, col] = df_final[col]
    else:
        # no csv for this headstage yet: hard copy
        print(f"No stimunit characterization CSV for {headstage_name} yet, creating {csv_fullfname}")
        stim_settings = df_final
        os.makedirs(os.path.dirname(csv_fullfname), exist_ok=True)

    stim_settings.to_csv(csv_fullfname)



def main():
    random.seed(42)
    debug = False
    L = Logger()
    L.init_logger(None, None, "WARNING")
    
    STIM_MODE = 'large_current'  # 'small_current' or 'large_current'
    # STIM_MODE = 'small_current'  # 'small_current' or 'large_current'
    headstage_name = "MEA1K27"
    nas_dir = device_paths()[0]
    # nas_dir = '/home/houmanjava/nas_imitation/'
    # device_dir = "devices/well_devices/4988/recordings"
    device_dir = f"devices/headstage_devices/{headstage_name}/recordings"
    
    R = 1_000_000  # 1 MOhm
    # R = 1_024_000  # 1 MOhm
    sine_ampl_DAC_units = 3 if STIM_MODE == 'large_current' else 30
    gain = 7
    rec_current_externally = False
    stim_units = list(range(32))
    # stim_units = [1,2,3,4]
    skip_stimulation = False  # LSB: if True, only (re)evaluate the recordings in full_path
    reprocess = False         # LSB: if True, also redo recordings that already have a processed/ output
    

    t = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
    rec_dir = f"{t}_{R=}_CharStim_{STIM_MODE}"
    
    rec_dir = f"2026-11-02_15.07_R=1000000_CharStim_ShowCase2_gain1_large_current"

    full_path = os.path.join(nas_dir, device_dir, rec_dir)
    
    # ======== 1) zero current DAC code per stim unit ========
    # full search: sweep the DAC values, writes the best DAC fit to processed/StimUnits_characterization.csv
    # n_amplifiers = 2
    # char_stim_units(full_path, stim_mode=STIM_MODE, n_amplifiers=n_amplifiers, gain=gain,
    #                 # start_centered_around=512, delta=256, set_size=16, max_iterations=4, 
    #                 start_centered_around=512, delta=32, set_size=16, max_iterations=4, 
    #                 stim_units=stim_units, debug=debug, rec_current_externally=rec_current_externally)
    
    # recalibration (was check_zero_current_DAC_drift): start at the headstage's current codes,
    # small windows; processed/all_StimUnit_DAC_sweeps.png compares old and new codes
    # rec_dir = f"{t}_{STIM_MODE}_check_zero_current_DAC_drift"
    # prior_codes = get_headstage_stim_settings(headstage_name)[f"{STIM_MODE}_zero_current_DAC"]
    # char_stim_units(full_path, stim_mode=STIM_MODE, n_amplifiers=3, gain=7,
    #                 start_centered_around=prior_codes, delta=8, down_step_size=2,
    #                 set_size=16, max_iterations=2, stim_units=stim_units, debug=debug)
    
    # exit()

    # ======== 2) current LSB, at the zero current DAC codes from 1) ========
    # standard LSB: one condition, recordings in full_path
    # conditions = [dict(gain=7, freq=1000, ncycles=400, sine_ampl_DAC_units=sine_ampl_DAC_units)]
    # char_current_lsb(full_path, R=R, stim_units=stim_units, n_amplifiers=6, stim_mode=STIM_MODE,
    #                  conditions=conditions, rec_current_externally=rec_current_externally,
    #                  skip_stimulation=skip_stimulation, reprocess=reprocess, debug=debug)
    
    # LSB sweep at several recording gains and frequencies, goes to full_path/LSB_sweep.
    # Sine amplitude is scaled down with gain, as the recording full scale shrinks.
    conditions = make_lsb_conditions(gains=[7, 112, 512], freqs=[1000, 100, 10],
                                     sine_ampl_DAC_units=sine_ampl_DAC_units,
                                     ncycles_by_freq={1000: 400,  10: 10, 100: 40})
    char_current_lsb(full_path, R=R, stim_units=stim_units, n_amplifiers=3, stim_mode=STIM_MODE,
                     conditions=conditions, subfolder="LSB_sweep", 
                     rec_current_externally=rec_current_externally,
                     skip_stimulation=skip_stimulation, reprocess=reprocess, debug=debug)

    # # copy over the new values to final csv
    # transfer_stimunit_characterization(full_path, stim_mode=STIM_MODE, 
    #                                    headstage_name=headstage_name)

if __name__ == "__main__":
    main()