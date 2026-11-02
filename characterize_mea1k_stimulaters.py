import os
import sys
import random
import glob
import datetime
import time
import threading
import re
import logging

import maxlab
import numpy as np
import pandas as pd
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

from PicoRecorder import PicoRecorder, mv_to_current_uA, DEFAULT_SENSE_GAIN

# testing options of _extract_DAC_transient_set
MEASURE_TRUE_PEAK = True  # True: largest |value| right after the connect rise instead of the settled value 10 ms later
MEDIAN_WIN = 21            # samples (~1 ms) of the median filter on the amplifier trace, 0: unfiltered. Keeps the edge
                           # and peak height, removes the noise burst after the connect (replaces the 2 kHz low-pass)
CURRENT_SMOOTH_WIN = 25    # samples (1.25 ms) of the moving average of the pico current, read out at the peak

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

    # unrecorded warm-up cycle: the ringnode download above leaves a transient that otherwise
    # lands in the baseline of the first file, now every recorded file follows a connect/disconnect
    print(f"Warm-up cycle at DAC {DAC_values[0]} (not recorded)...")
    shift_DAC(DAC_values[0])
    time.sleep(break_time/4)
    array.connect_amplifier_to_stimulation(ampl_id)
    array.download()
    time.sleep(break_time*2)
    array.disconnect_amplifier_from_stimulation(ampl_id)
    array.download()
    time.sleep(break_time*2)
    shift_DAC(512) # reset
    time.sleep(break_time*4)

    for DAC_val in (DAC_values):
        fname = f"config_StimUnit{int(stim_unit):02d}_Ampl{ampl_id:04d}_Set{set_id}_DAC{DAC_val}"
        channels_args = {'channels': [ampl_id]} if not debug else {'legacy': True}
        start_saving(s, dir_name=dirname, fname=fname, **channels_args)
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

        stop_saving(s)
        if pico_rec:
            time.sleep(.3)
            pico_rec.stop()
        print('=================================')
        
    array.disconnect_amplifier_from_ringnode(ampl_id)
    array.download()
    turn_off_stimulation_units([stim_unit])
            # break
        

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

def _extract_fname_info(fname):
    # config_StimUnit00_Ampl0828_Set0_DAC480.raw.h5
    stimunit_id = int(fname.split("_")[1].replace("StimUnit", ""))
    ampl_id = int(fname.split("_")[2].replace("Ampl", ""))
    set_id = int(fname.split("_")[3].replace("Set", ""))
    DAC_code = int(fname.split("_")[4].replace("DAC", "").replace(".raw.h5", ""))
    return stimunit_id, ampl_id, set_id, DAC_code

def _step_edge(x, w=40):
    """Index of the largest step in x: difference between the mean of the next w and the previous w
    samples. Averaging makes it robust to the noise spikes that win a plain argmax(|diff|)."""
    c = np.r_[0, np.cumsum(x)]
    t = np.arange(w, len(x) - w)
    step = (c[t + w] - c[t]) / w - (c[t] - c[t - w]) / w
    return int(t[np.argmax(np.abs(step))])

def _extract_DAC_transient_set(dirname, ampl_id, stim_unit, set_id, debug=False):
    if debug:
        fig, ax = plt.subplots(figsize=(20, 10), nrows=3, sharex=True)
    
    # go through all files in the directory, skip those that do not match the set_id, ampl_id, stim_unit
    dac_transients_res = []
    fnames = [f for f in sorted(os.listdir(dirname)) if f.endswith(".raw.h5")]
    for i,fname in enumerate(fnames):
        if (f"Set{set_id}_" not in fname) or ("DAC" not in fname):
            # print("Skipping, set_id or DAC missing", (f"Set{set_id}_" not in fname), ("DAC" not in fname), fname)
            continue
        if f"Ampl{ampl_id:04d}_" not in fname:
            # print("Skipping, ampl_id mismatch", ampl_id,  f"Ampl{ampl_id:04d}_", fname)
            continue
        if f"StimUnit{stim_unit:02d}_" not in fname:
            # print("Skipping, stim_unit mismatch")
            continue
        
        # extract dac_val
        _, ampl_id, _, DAC_val = _extract_fname_info(fname)
        print(f"\nProcessing i={i}: {fname} ", ampl_id, stim_unit)
        data = read_raw_data(dirname, fname, convert2uV=True,
                             subtract_dc_offset=False,)
        dac_trace = read_stim_DAC(dirname, fname)
        if i == 0:
            # for some reason first measuremnt has delay in  amplifier traces + DAC
            print("First file: shifting data and DAC to remove initial delay")
            data = data[:, 2600:]
            if dac_trace is not None:
                dac_trace = dac_trace[2600:]
        
        if data.shape[1]  < 1300:
            print("Warning: File was too short, skipping")
            continue
            
        if dac_trace is None: 
            # just one row in the file, when using compression
            amplifier_trace = data[0].astype(float)
        else:
            # when run in legacy mode
            amplifier_trace = data[ampl_id].astype(float)
            
        # edge-preserving removal of the noise burst after the connect
        if MEDIAN_WIN:
            amplifier_trace = median_filter(amplifier_trace, size=MEDIAN_WIN, mode='nearest')
        
        baseline = np.median(amplifier_trace[:500])
        amplifier_trace -= baseline

        from_t, to_t = 700, 1500 # interval contains the transient
        interval_trace = amplifier_trace[from_t:to_t]
        if len(interval_trace) == 0:
            print(f"Skipping empty interval for {fname}")
            continue

        # calculate derivative only inside the region of interest
        diff_amplifier_trace = np.diff(interval_trace)
        fast_rise_at = np.argmax(np.abs(diff_amplifier_trace))
        if MEASURE_TRUE_PEAK:
            # single-sample derivative is dominated by noise spikes for small steps
            fast_rise_at = _step_edge(interval_trace)
        # latest rise whose peak/plateau readout (rise + up to 219, or the fixed +225) is still inside the interval
        max_rise_at = len(interval_trace) - (226 if MEASURE_TRUE_PEAK else 220)
        if fast_rise_at > max_rise_at:
            print(f"Warning: Fast rise at {fast_rise_at}, outside expected range, skipping")
            continue
        # shift post noise that we see after peak, traces converge twoards middle
        if MEASURE_TRUE_PEAK:
            # up to 350 samples (17.5 ms) after the rise, stays clear of the disconnect transient.
            # Peak in the direction of the plateau; an overshoot only counts if it clearly exceeds
            # the baseline noise, otherwise (small steps) a noise maximum would be picked -> read
            # the plateau at a fixed delay instead
            plateau = np.median(interval_trace[fast_rise_at + 150: fast_rise_at + 300])
            win = interval_trace[fast_rise_at: fast_rise_at + 350] * np.sign(plateau)
            noise = amplifier_trace[:500].std()
            peak_at = fast_rise_at + (np.argmax(win) if win.max() - abs(plateau) > 5 * noise else 225)
        else:
            peak_at = fast_rise_at+200 + np.argmax(np.abs(interval_trace[fast_rise_at+200: fast_rise_at + 220]))
        peak = interval_trace[peak_at]
        peak_at_plot = from_t + peak_at
        print(f"Baseline: {baseline:.1f}, Peak: {peak:.1f}, DAC_val: {DAC_val}, at {peak_at_plot}")
        
        # try to read external picoscope recording (full trace, not windowed)
        pico_fullfname = os.path.join(dirname, fname.replace('.raw.h5', '.npz'))
        if os.path.exists(pico_fullfname):
            pico_data = np.load(pico_fullfname)
            sensed_current_uA = mv_to_current_uA(pico_data['mv'], DEFAULT_SENSE_GAIN)
        else:
            sensed_current_uA = np.full_like(amplifier_trace, np.nan)
            print(f"Warning: Pico recording not found for {fname}, skipping current measurement")
        # window-averaged current, the amplitude is read from this
        sensed_current_smooth_uA = uniform_filter1d(sensed_current_uA, CURRENT_SMOOTH_WIN, mode='nearest')
        # pico starts/stops with a lag and saves whole 500-sample chunks, its trace can be
        # shorter than the MEA1K trace
        if peak_at_plot < len(sensed_current_uA):
            current_at_peak = sensed_current_smooth_uA[peak_at_plot]
        else:
            current_at_peak = np.nan
            print(f"Warning: Pico trace too short ({len(sensed_current_uA)} < {peak_at_plot}) for {fname}")

        dac_transients_res.append(pd.Series({
            "stimunit_id": stim_unit,
            "ampl_id": ampl_id,
            "set_id": set_id,
            "DAC_val": DAC_val,
            "peak_uV": peak,
            "peak_at": peak_at_plot,
            "baseline": baseline,
            "sensed_current_uA_at_peak": current_at_peak,
        }))
        
        if debug:
            ax[1].plot(amplifier_trace, linewidth=1, alpha=.7)
            col = ax[1].lines[-1].get_color()
            
            ax[1].axvspan(from_t, to_t, color='0.9', alpha=0.18, zorder=0)
            ax[1].vlines(peak_at_plot, np.min(amplifier_trace), np.max(amplifier_trace), alpha=0.4, linestyle='--', color=col)
            ax[1].scatter([peak_at_plot], [peak], color=col, s=50, edgecolor=col, 
                          marker='o', zorder=10, label=f'DAC {DAC_val}, Peak {peak/1000:.0f} mV')
            ax[1].set_xlabel("Sample index")
            
            if dac_trace is not None:
                dac_full = dac_trace.astype(float)
                ax[0].plot(dac_full, linewidth=1, color=col, 
                        alpha=0.4, label=f'DAC {DAC_val}')
                ax[0].set_ylabel("DAC value")
                ax[2].plot(sensed_current_uA[:3000], linewidth=0.6, alpha=0.25, color=col)
                ax[2].plot(sensed_current_smooth_uA[:3000], linewidth=1.5, alpha=0.9, color=col,
                           linestyle='--', label='Pico current [uA]')
                # annotate
                ax[2].scatter([peak_at_plot], [current_at_peak], color=col, s=50, edgecolor=col,
                              marker='o', zorder=10, label=f'DAC {DAC_val}, Current {current_at_peak:.2f} uA')
                ax[2].set_ylabel("Current [uA]")
                ax[2].set_xlabel("Sample index")
    
            ax[0].set_title(f"Raw data from {fname}")
            ax[0].legend(fontsize=6, ncol=2)
            ax[1].legend(fontsize=6, ncol=2)
            ax[2].legend(fontsize=6, ncol=2)
            plt.tight_layout()
            # save as live figure
            plt.savefig('./live_figures/parasitic_DACcur_live.png', dpi=400)
    
    dac_transients_res = pd.DataFrame(dac_transients_res)
    fullfname = os.path.join(dirname, f"results_StimUnit{stim_unit:02d}_Ampl{ampl_id:04d}_Set{set_id}.csv")
    dac_transients_res.to_csv(fullfname, index=False)        
    return dac_transients_res

def _sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

def _fit_single_regression(df):
    """Fits a sigmoid and calculates R^2 and the residual nearest to y=0."""
    df = df.dropna(subset=["DAC_val", "peak_uV"])
    if len(df) < 4 or df["DAC_val"].nunique() < 4:
        return None

    x = df["DAC_val"].to_numpy(dtype=float)
    y = df["peak_uV"].to_numpy(dtype=float)

    # 1. Fit sigmoid
    L_guess = np.max(y) - np.min(y)
    b_guess = np.min(y)
    x0_guess = np.median(x)
    k_guess = 0.01 if (y[-1] > y[0]) == (x[-1] > x[0]) else -0.01

    try:
        popt, pcov = curve_fit(_sigmoid, x, y, p0=[L_guess, x0_guess, k_guess, b_guess], maxfev=10000)
    except Exception as e:
        print(f"Sigmoid fit failed: {e}")
        return None

    y_pred = _sigmoid(x, *popt)

    # 2. Calculate general goodness of fit (R^2)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)

    # 3. Find the empirical point closest to y = 0
    idx_closest_to_zero = np.argmin(np.abs(y))
    x_zero_pt = x[idx_closest_to_zero]
    y_zero_pt_actual = y[idx_closest_to_zero]
    y_zero_pt_pred = y_pred[idx_closest_to_zero]
    
    # Calculate the residual at that point
    error_at_zero = abs(y_zero_pt_actual - y_zero_pt_pred)

    return popt, r_squared, x_zero_pt, y_zero_pt_actual, y_zero_pt_pred, error_at_zero































def eval_stim_unit_results(dirname, stim_unit_results, debug=False):
    if not stim_unit_results:
        return

    # Extract ID for saving
    stimunit_id = int(stim_unit_results[0]["stimunit_id"].iloc[0])

    # Create figure with independent axes (sharex=False by default)
    fig, ax = plt.subplots(nrows=2, figsize=(14, 10))
    fig.suptitle(f"Characterization: Stimulation Unit {stimunit_id:02d}", fontsize=16, fontweight="bold")
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    y_formatter = plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}')

    # --- Top Plot: All Points (colored by set) ---
    all_results = pd.concat(stim_unit_results, axis=0, ignore_index=True)
    
    # Target Box
    ax[0].axhspan(-25000, 25000, color='gray', alpha=0.15, zorder=0, label="±25k uV Target Range")
    
    for i, res_df in enumerate(stim_unit_results):
        if not res_df.empty:
            amp_id = res_df["ampl_id"].iloc[0]
            set_id = res_df["set_id"].iloc[0]
            c = colors[i % len(colors)]
            ax[0].scatter(res_df["DAC_val"], res_df["peak_uV"], s=30, alpha=0.3, color=c, 
                          label=f"Ampl {amp_id} | Set {set_id}")
            
    ax[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax[0].yaxis.set_major_formatter(y_formatter)

    stats_all = _fit_single_regression(all_results)
    best_dac = None  # To save to CSV later

    if stats_all is not None:
        popt, r_sq, x_0, y_0, y_pred_0, err_0 = stats_all
        L, x0_p, k, b = popt
        
        x_line = np.linspace(all_results["DAC_val"].min(), all_results["DAC_val"].max(), 100)
        y_line = _sigmoid(x_line, *popt)
        
        ax[0].plot(x_line, y_line, color="green", linestyle="--", 
                   label=f"Fit All (R² = {r_sq:.3f})")
                   
        ax[0].axvline(x_0, color='red', linestyle='--', alpha=0.7, 
                      label=f"Empirical Min (DAC: {x_0:.0f}, |Peak|: {abs(y_0):,.1f} uV)")
        ax[0].scatter([x_0], [y_0], facecolors='none', edgecolors='red', s=200, linewidths=2, zorder=6)

        if k != 0:
            val = -L / b - 1
            if val > 0:
                x_cross_0 = x0_p - (1 / k) * np.log(val)
            else:
                x_cross_0 = x_0
                
            best_dac_top = int(round(x_cross_0))
            best_dac_top = max(0, min(1023, best_dac_top))

            y_pred_best = _sigmoid(best_dac_top, *popt)
            delta_dac = abs(x_0 - best_dac_top)

            ax[0].axvline(best_dac_top, color='green', linestyle='-.', alpha=0.7, 
                          label=f"Fit 0-Cross (DAC: {best_dac_top}, Offset: {y_pred_best:,.1f} uV, Δ: {delta_dac:.0f})")
            ax[0].scatter([best_dac_top], [y_pred_best], facecolors='none', edgecolors='green', s=200, linewidths=2, zorder=6)
            best_dac = best_dac_top

    ax[0].set_title("All Measured Points")
    ax[0].set_ylabel("Peak [uV]")
    ax[0].grid(True, axis='y', alpha=0.3)
    ax[0].legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))

    # --- Bottom Plot: Current (Latest) Set Only ---
    current_set = stim_unit_results[-1]
    current_color = colors[(len(stim_unit_results) - 1) % len(colors)]

    # Target Box
    ax[1].axhspan(-25000, 25000, color='gray', alpha=0.15, zorder=0)

    ax[1].scatter(current_set["DAC_val"], current_set["peak_uV"], s=50, alpha=0.9, 
                  color=current_color, label="Current Set")
    ax[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax[1].yaxis.set_major_formatter(y_formatter)

    stats_curr = _fit_single_regression(current_set)
    if stats_curr is not None:
        popt, r_sq_c, x_0c, y_0c, y_pred_0c, err_0c = stats_curr
        L, x0, k, b = popt
        
        # Dense x for smooth curve
        x_line_c = np.linspace(current_set["DAC_val"].min(), current_set["DAC_val"].max(), 100)
        y_line_c = _sigmoid(x_line_c, *popt)
        
        # Regression Line (Green)
        ax[1].plot(x_line_c, y_line_c, color="green", linestyle="--", 
                   label=f"Fit Current (R² = {r_sq_c:.3f})")
        
        # Empirical Min Highlight (Red)
        ax[1].axvline(x_0c, color='red', linestyle='--', alpha=0.7, 
                      label=f"Empirical Min (DAC: {x_0c:.0f}, |Peak|: {abs(y_0c):,.1f} uV)")
        ax[1].scatter([x_0c], [y_0c], facecolors='none', edgecolors='red', s=200, linewidths=2, zorder=6)
        
        # Fit Best DAC Highlight (Green)
        best_dac_c = None
        if k != 0:
            val = -L / b - 1
            if val > 0:
                x_cross_0c = x0 - (1 / k) * np.log(val)
            else:
                x_cross_0c = x_0c
            
            best_dac_c = int(round(x_cross_0c))
            best_dac_c = max(0, min(1023, best_dac_c))

            y_pred_best_c = _sigmoid(best_dac_c, *popt)
            delta_dac_c = abs(x_0c - best_dac_c)

            ax[1].axvline(best_dac_c, color='green', linestyle='-.', alpha=0.7, 
                          label=f"Fit 0-Cross (DAC: {best_dac_c}, Offset: {y_pred_best_c:,.1f} uV, Δ: {delta_dac_c:.0f})")
            ax[1].scatter([best_dac_c], [y_pred_best_c], facecolors='none', edgecolors='green', s=200, linewidths=2, zorder=6)

    ax[1].set_title("Current Set Only")
    ax[1].set_xlabel("DAC value")
    ax[1].set_ylabel("Peak [uV]")

    # Bottom Plot Specific Formatting (No x grid, no x ticks/labels)
    ax[1].set_xticks([]) # Removes the ticks and labels completely
    ax[1].grid(False, axis='x')
    ax[1].grid(True, axis='y', alpha=0.3)
    
    ax[1].legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # adjust rect to make room for suptitle so it doesn't overlap with the top title
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Saving Routine ---
    os.makedirs(os.path.join(dirname, "processed"), exist_ok=True)
    
    # 1. Update/Save the live preview
    plt.savefig('./live_figures/stim_unit_characterization_live.png', dpi=300)
    
    # 2. Save the explicit final image for the given Stim Unit
    plot_path = os.path.join(dirname, "processed", f"StimUnit{stimunit_id:02d}_characterization.png")
    plt.savefig(plot_path, dpi=300)

    # 3. Aggreate standard DataFrame for downstream current_lsb evaluation
    if best_dac is not None:
        csv_path = os.path.join(dirname, "processed", "StimUnits_characterization.csv")
        if os.path.exists(csv_path):
            df_char = pd.read_csv(csv_path)
        else:
            df_char = pd.DataFrame(columns=["stimunit_id", "zero_current_DAC_code"])
        
        # Overwrite if exists, otherwise append
        if stimunit_id in df_char["stimunit_id"].values:
            df_char.loc[df_char["stimunit_id"] == stimunit_id, "zero_current_DAC_code"] = best_dac
        else:
            new_row = pd.DataFrame([{"stimunit_id": stimunit_id, "zero_current_DAC_code": best_dac}])
            df_char = pd.concat([df_char, new_row], ignore_index=True)
            
        df_char = df_char.astype(int) # Standardize back to clean integer codes
        df_char.sort_values("stimunit_id", inplace=True)
        df_char.to_csv(csv_path, index=False)

    if debug:
        live_dir = os.path.join('.', 'live_figures')
        os.makedirs(live_dir, exist_ok=True)
        debug_fname = os.path.join(live_dir, f"StimUnit_characterization_debug.png")
        plt.savefig(debug_fname, dpi=300)
        
    plt.close(fig)

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
                    down_step_size=4, rec_current_externally=False, fast_setup=False):
    # create processed directory if not exists
    os.makedirs(os.path.join(dirname, "processed"), exist_ok=True)
    if fast_setup:
        array = maxlab.chip.Array()
    else:
        array = setup_stim_unit_characterization(dirname, gain=gain) 
        
    if rec_current_externally:
        pico_rec = PicoRecorder(voltage_range="5V", resolution="16BIT", sample_interval_us=50)
    else:
        pico_rec = None
    
    for stim_unit in stim_units:
        plt.close('all')
        if stim_unit>30:
            debug=True
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
                centered_around, used_delta = start_centered_around, delta
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
                                                    set_id, debug=debug)
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
                
                # set_id

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

    # unrecorded warm-up cycle: the ringnode download above leaves a transient that otherwise
    # lands in the baseline of the recording, same as in _sweep_DAC
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

def _get_pp_logger(dirname):
    """Logger of the postprocess thread: writes only to a file (no console, no propagation to
    the main logger), so its messages don't intermingle with the main thread's output."""
    log_fullfname = os.path.join(dirname, "processed", "lsb_postproc.log")
    os.makedirs(os.path.dirname(log_fullfname), exist_ok=True)
    pp_logger = logging.getLogger("lsb_pp")
    pp_logger.setLevel(logging.INFO)
    pp_logger.propagate = False
    if not pp_logger.handlers:
        hdlr = logging.FileHandler(log_fullfname)
        hdlr.setFormatter(logging.Formatter("%(asctime)s|%(message)s", "%H:%M:%S"))
        pp_logger.addHandler(hdlr)
    return pp_logger, log_fullfname

def _postprocess_lsb_live(dirname, tag, eval_kwargs, rows, stop_event, poll_interval=2,
                          save_kwargs=None):
    """Run in a thread. Evaluates every recording whose marker csv (written by
    char_current_lsb after the recording finished) exists; one final sweep after stop.
    Logs to processed/lsb_postproc.log, see _get_pp_logger."""
    pp_logger, _ = _get_pp_logger(dirname)
    done = set()
    while True:
        stopped = stop_event.is_set()
        markers = sorted(f for f in os.listdir(dirname) if f.endswith(f"{tag}.csv"))
        new = [m for m in markers if m not in done]
        for marker in new:
            fname = marker.replace(".csv", ".raw.h5")
            rows.append(_eval_lsb_file(dirname, fname, log=pp_logger.info, **eval_kwargs))
            done.add(marker)
        if new:
            _save_lsb_results(dirname, rows, eval_kwargs['stim_mode'], **(save_kwargs or {}))
            pp_logger.info(f"Postprocessed {len(done)} LSB recordings")
        if stopped:
            break
        time.sleep(poll_interval)

def _run_postprocess_lsb_live_guarded(dirname, tag, eval_kwargs, rows, stop_event, save_kwargs=None):
    """Run _postprocess_lsb_live, but hard-exit the process if it fails unexpectedly
    (same as shortcut_stim), so stimulation doesn't continue without evaluation."""
    try:
        _postprocess_lsb_live(dirname, tag, eval_kwargs, rows, stop_event, save_kwargs=save_kwargs)
    except BaseException as exc:
        Logger().logger.critical("LSB postprocess thread crashed", exc_info=exc)
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    finally:
        if not stop_event.is_set():
            Logger().logger.critical("LSB postprocess thread exited unexpectedly")
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)

def _remove_existing_lsb_files(dirname, tag):
    """MaxWell appends _0, _1, .. instead of overwriting, so earlier recordings of this kind
    (h5, pico npz, marker csv) are deleted before re-recording."""
    existing = [f for f in os.listdir(dirname) if re.search(re.escape(tag) + r"(_\d+)?\.(raw\.h5|npz|csv)$", f)]
    for f in existing:
        os.remove(os.path.join(dirname, f))
    if existing:
        print(f"Overwriting: deleted {len(existing)} '*{tag}' files in {dirname}")

def char_current_lsb(dirname, R, sine_ampl_DAC_units, stim_units, n_amplifiers,
                     freq, ncycles, stim_mode, dac_code_offset=None, rec_current_externally=False,
                     debug=False, plot_ylims='default'):
    # earlier recordings of this kind are deleted, MaxWell would append _0, _1, .. instead
    tag = f"_f{freq:04d}Hz_CurrentLSB" + (f"_DACoffset{dac_code_offset}" if dac_code_offset is not None else "")
    _remove_existing_lsb_files(dirname, tag)

    array = setup_stim_unit_characterization(dirname)
    if rec_current_externally:
        pico_rec = PicoRecorder(voltage_range="5V", resolution="16BIT", sample_interval_us=50)
    else:
        pico_rec = None
    dac_settings = pd.read_csv(os.path.join(dirname, "processed", "StimUnits_characterization.csv"))

    # postprocessing thread evaluates each recording while the next one is recorded;
    # only file-saving matplotlib backend, the thread plots outside the main thread
    plt.switch_backend('Agg')
    if plot_ylims == 'default':
        plot_ylims = LSB_PLOT_YLIMS.get(stim_mode)
    eval_kwargs = dict(R=R, sine_ampl_DAC_units=sine_ampl_DAC_units, stim_mode=stim_mode,
                       debug=debug, plot_ylims=plot_ylims)
    rows, stop_event = [], threading.Event()
    pp = threading.Thread(target=_run_postprocess_lsb_live_guarded,
                          args=(dirname, tag, eval_kwargs, rows, stop_event), daemon=True)
    pp.start()
    pp_logfile = _get_pp_logger(dirname)[1]
    print(f"\nLSB postprocessing runs in a thread, its output goes to a log file. "
          f"Follow it in a second terminal with:\n    tail -n +1 -f '{pp_logfile}'\n", flush=True)
    try:
        _char_current_lsb_loop(dirname, array, dac_settings, stim_units, n_amplifiers, tag,
                               sine_ampl_DAC_units, freq, ncycles, stim_mode, dac_code_offset,
                               pico_rec, R)
    finally:
        stop_event.set()   # triggers final sweep
        pp.join()
    if rows:
        result_fullfname = _save_lsb_results(dirname, rows, stim_mode)
        _plot_lsb_summary(result_fullfname, R, stim_units, stim_mode)

def _char_current_lsb_loop(dirname, array, dac_settings, stim_units, n_amplifiers, tag,
                           sine_ampl_DAC_units, freq, ncycles, stim_mode, dac_code_offset,
                           pico_rec, R, settle_time=0.1, ampl_cache=None):
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

            fname = f"config_StimUnit{int(stim_unit):02d}_Ampl{ampl_id:04d}{tag}"
            _record_sine(dirname, array, stim_unit, ampl_id, fname, dac_code, sine_seq,
                         seq_duration, stim_mode, pico_rec=pico_rec, settle_time=settle_time)
            # written after stop_saving + pico stop -> "recording finished" signal for postproc
            pd.Series({
                "stimunit_id": stim_unit, "ampl_id": ampl_id, "dac_code": dac_code,
                "sine_amplitude_DAC_units": sine_ampl_DAC_units, "freq": freq,
                "ncycles": ncycles, "stim_mode": stim_mode, "resistance": R,
            }).to_frame().T.to_csv(os.path.join(dirname, f"{fname}.csv"), index=False)

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

def char_current_lsb_sweep(dirname, R, stim_units, n_amplifiers, stim_mode, conditions,
                           subfolder="LSB_sweep", rec_current_externally=False, debug=False,
                           plot_ylims='default'):
    """Current LSB at several recording gains and sine frequencies. Everything (recordings,
    marker csvs, processed/ with the results) goes into dirname/subfolder, the zero current
    DAC codes come from dirname/processed/StimUnits_characterization.csv (char_stim_units).
    conditions: see make_lsb_conditions. The same amplifiers are used in every condition.
    Evaluated live in a postprocess thread (log file, see _get_pp_logger)."""
    sweep_dir = os.path.join(dirname, subfolder)
    os.makedirs(os.path.join(sweep_dir, "processed"), exist_ok=True)
    for c in conditions:
        c['tag'] = f"_f{c['freq']:04d}Hz_Gain{c['gain']}_Sine{c['sine_ampl_DAC_units']}_CurrentLSB"
        _remove_existing_lsb_files(sweep_dir, c['tag'])

    dac_settings = pd.read_csv(os.path.join(dirname, "processed", "StimUnits_characterization.csv"))
    pico_rec = (PicoRecorder(voltage_range="5V", resolution="16BIT", sample_interval_us=50)
                if rec_current_externally else None)

    plt.switch_backend('Agg')
    if plot_ylims == 'default':
        plot_ylims = LSB_PLOT_YLIMS.get(stim_mode)
    # sine amplitude and gain are parsed from the file name
    eval_kwargs = dict(R=R, sine_ampl_DAC_units=None, stim_mode=stim_mode, debug=debug,
                       plot_ylims=plot_ylims)
    save_kwargs = dict(group_by=('stimunit_id', 'gain', 'freq', 'sine_amplitude_DAC_units'),
                       name="lsb_sweep")
    rows, stop_event = [], threading.Event()
    pp = threading.Thread(target=_run_postprocess_lsb_live_guarded,
                          args=(sweep_dir, "_CurrentLSB", eval_kwargs, rows, stop_event, save_kwargs),
                          daemon=True)
    pp.start()
    pp_logfile = _get_pp_logger(sweep_dir)[1]
    print(f"\nLSB postprocessing runs in a thread, its output goes to a log file. "
          f"Follow it in a second terminal with:\n    tail -n +1 -f '{pp_logfile}'\n", flush=True)

    ampl_cache, array, cur_gain = {}, None, None
    try:
        for i, c in enumerate(conditions):
            print(f"\n######## Condition {i+1}/{len(conditions)}: gain {c['gain']}, {c['freq']} Hz, "
                  f"{c['ncycles']} cycles, sine ±{c['sine_ampl_DAC_units']} DAC units")
            if c['gain'] != cur_gain:
                array = setup_stim_unit_characterization(sweep_dir, gain=c['gain'])
                cur_gain = c['gain']
            _char_current_lsb_loop(sweep_dir, array, dac_settings, stim_units, n_amplifiers, c['tag'],
                                   c['sine_ampl_DAC_units'], c['freq'], c['ncycles'], stim_mode,
                                   None, pico_rec, R, settle_time=max(0.1, 2/c['freq']),
                                   ampl_cache=ampl_cache)
    finally:
        stop_event.set()   # triggers final sweep
        pp.join()
    if rows:
        result_fullfname = _save_lsb_results(sweep_dir, rows, stim_mode, **save_kwargs)
        _plot_lsb_sweep_summary(result_fullfname, R, stim_mode)

def _plot_lsb_sweep_summary(result_fullfname, R, stim_mode):
    aggr = pd.read_csv(result_fullfname)
    gains, freqs = sorted(aggr['gain'].unique()), sorted(aggr['freq'].unique())
    colors = {g: plt.cm.tab10(i) for i, g in enumerate(gains)}
    markers = {f: m for f, m in zip(freqs, ['o', 's', '^', 'D', 'v', 'P'])}
    lsb = f"{stim_mode}_LSB_nA"

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"LSB sweep, {stim_mode} (R={R/1000:g}kΩ)")
    for g in gains:
        for f in freqs:
            d = aggr[(aggr['gain'] == g) & (aggr['freq'] == f)]
            if d.empty:
                continue
            ax[0, 0].scatter(d['stimunit_id'], d[lsb], s=25, alpha=.5, color=colors[g], marker=markers[f],
                             label=f"gain {g:g}, {f} Hz")
            ax[0, 1].scatter([f]*len(d), d[lsb], s=20, alpha=.3, color=colors[g])
            ph = np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(d['phase_shift_deg'])))))
            ax[1, 0].scatter([f], [ph], s=60, color=colors[g], marker=markers[f])
            ratio = d[f"{stim_mode}_external_LSB_nA"] / d[lsb]
            ax[1, 1].scatter([f]*len(d), ratio, s=20, alpha=.4, color=colors[g])
    for g in gains:
        med = aggr[aggr['gain'] == g].groupby('freq')[lsb].median()
        ax[0, 1].plot(med.index, med.values, color=colors[g], marker='_', ms=14, label=f"gain {g:g}")
        ratio = (aggr[aggr['gain'] == g].assign(r=lambda x: x[f"{stim_mode}_external_LSB_nA"] / x[lsb])
                 .groupby('freq')['r'].median())
        ax[1, 1].plot(ratio.index, ratio.values, color=colors[g], marker='_', ms=14, label=f"gain {g:g}")
    ax[0, 0].set(title="Amplifier LSB per stim unit", xlabel="Stim Unit ID", ylabel="LSB [nA]")
    ax[0, 0].legend(fontsize=6, ncol=2)
    ax[0, 1].set(title="Amplifier LSB vs frequency (line: median)", xlabel="Frequency [Hz]",
                 ylabel="LSB [nA]", xscale='log')
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].set(title="Phase shift vs DAC (circular mean)", xlabel="Frequency [Hz]",
                 ylabel="Phase [deg]", xscale='log')
    ax[1, 1].set(title="External (pico) LSB / amplifier LSB, 1 = amplifier gain correct",
                 xlabel="Frequency [Hz]", ylabel="ratio", xscale='log')
    ax[1, 1].axhline(1, color='gray', linestyle='--', alpha=.5)
    ax[1, 1].legend(fontsize=8)
    for a in ax.flat:
        a.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(result_fullfname.replace(".csv", ".png"), dpi=300)
    plt.close(fig)

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

# fixed y-limits of the live debug plots, per stim mode (None -> auto-scaled)
# v: amplifier voltage [uV], i: picoscope current through R [uA]; *_raw: unprocessed panel
LSB_PLOT_YLIMS = {
    'large_current': dict(v=(-200_000, 200_000), v_raw=(-300_000, 300_000),
                          i=(-0.25, 0.25), i_raw=(-0.3, 0.3)),
    'small_current': None,
}

def _eval_lsb_file(dirname, fname, R, sine_ampl_DAC_units, stim_mode, debug=False,
                   plot_ylims=None, n_zoom_cycles=5, log=print):
    """Lock-in amplitude of the amplifier voltage and (if recorded) the picoscope current
    for one sine LSB recording. Returns one result row."""
    plot_ylims = plot_ylims or {}
    if "_DACoffset" in fname:
        dac_code_offset = int(re.search(r'DACoffset(-?\d+)', fname).group(1))
    else:
        dac_code_offset = 0
    file_freq = int(re.search(r'f(\d{4})Hz', fname).group(1))
    stimunit_id = int(fname.split("_")[1].replace("StimUnit", ""))
    ampl_id = int(fname.split("_")[2].replace("Ampl", ""))
    if "_Sine" in fname:
        # sweep recordings carry their own sine amplitude (varies with gain)
        sine_ampl_DAC_units = int(re.search(r'_Sine(\d+)', fname).group(1))

    log(f"Processing {fname} ampl={ampl_id} stimunit={stimunit_id}")
    # sweep recordings carry their recording gain in the name, otherwise the constant applies
    gain = float(re.search(r'_Gain(\d+)', fname).group(1)) if "_Gain" in fname else None
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

def _save_lsb_results(dirname, rows, stim_mode, group_by=('stimunit_id',), name="lsb_characterization"):
    """Aggregate per-recording rows (median per group_by, default per stim unit) and write the LSB csv."""
    aggr = pd.DataFrame(rows)
    group_by = list(group_by)
    aggr['median_amplitude_uV'] = aggr.groupby(group_by)['mean_amplitude_uV'].transform('median')
    aggr[f'{stim_mode}_median_LSB_nA'] = aggr.groupby(group_by)[f"{stim_mode}_LSB_nA"].transform('median')
    result_fullfname = os.path.join(dirname, "processed", f"{stim_mode}_{name}.csv")
    os.makedirs(os.path.dirname(result_fullfname), exist_ok=True)
    aggr.to_csv(result_fullfname, index=False)
    return result_fullfname

def eval_current_lsb(dirname, R, sine_ampl_DAC_units, stim_units, freq, ncycles, stim_mode, debug=False,
                     plot_ylims='default'):
    """Offline evaluation of all sine LSB recordings in dirname (char_current_lsb does the
    same live while recording)."""
    if plot_ylims == 'default':
        plot_ylims = LSB_PLOT_YLIMS.get(stim_mode)
    aggr = []
    for i, fname in enumerate(sorted(os.listdir(dirname))):
        # if not fname.endswith("_CurrentLSB.raw.h5"):
        if "_CurrentLSB" not in fname or not fname.endswith(".raw.h5"):
            continue
        if not re.search(r'f(\d{4})Hz', fname):
            continue
        if int(fname.split("_")[1].replace("StimUnit", "")) not in stim_units:
            continue
        aggr.append(_eval_lsb_file(dirname, fname, R, sine_ampl_DAC_units, stim_mode,
                                   debug=debug, plot_ylims=plot_ylims))
        if debug:
            time.sleep(1)

    result_fullfname = _save_lsb_results(dirname, aggr, stim_mode)
    _plot_lsb_summary(result_fullfname, R, stim_units, stim_mode)

def _plot_lsb_summary(result_fullfname, R, stim_units, stim_mode):
    aggr = pd.read_csv(result_fullfname) # read back to ensure clean data for plotting
    print(aggr)
    
    fig = plt.figure(figsize=(14, 8))
    ax = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), 
          fig.add_subplot(2, 2, 3, projection='polar'), fig.add_subplot(2, 2, 4)]
    
    unique_med = aggr.drop_duplicates('stimunit_id')
    unique_freqs = sorted(aggr['freq'].unique())
    markers = ['o', 's', '^']
    
    # 1.) Zero current DAC code
    ax[0].set_title("1.) Zero current DAC code")
    ax[0].set_xlabel("Stim Unit ID")
    ax[0].set_ylabel("Zero current DAC code")
    ax[0].set_xticks(stim_units)
    for i, freq_val in enumerate(unique_freqs):
        freq_data = aggr[aggr['freq'] == freq_val]
        print(freq_data)
        marker = markers[i % len(markers)]
        ax[0].scatter(freq_data['stimunit_id'], freq_data[f'{stim_mode}_zero_current_DAC'], 
                      s=50, c=freq_data['dac_code_offset'], alpha=0.6, marker=marker)
    ax[0].tick_params(axis='x', which='both', rotation=90)
    ax[0].grid(True)
    
    # 2.) Mean Amplitude
    ax[1].set_title("2.) Mean Amplitude [uV]")
    ax[1].set_xlabel("Stim Unit ID")
    ax[1].set_ylabel("Mean Amplitude [uV]")
    ax[1].set_xticks(stim_units)
    ax[1].grid(True)
    # ax[1].set_ylim(0, aggr['mean_amplitude_uV'].max() * 1.2)
    
    for i, freq_val in enumerate(unique_freqs):
        freq_data = aggr[aggr['freq'] == freq_val]
        marker = markers[i % len(markers)]
        ax[1].scatter(freq_data['stimunit_id'], freq_data['mean_amplitude_uV'], alpha=0.8, 
                      c=freq_data['dac_code_offset'], edgecolor='none', s=30, 
                      marker=marker, label=f'{freq_val}Hz')
    ax[1].scatter(unique_med['stimunit_id'], unique_med['median_amplitude_uV'], color='blue', 
                  s=200, marker='_', linewidths=2, zorder=5, label='Median')
    ax[1].legend(fontsize=9)
    ax[1].tick_params(axis='x', which='both', rotation=90)
    
    # Phase Shift (Polar)
    ax[2].set_title("Phase Shift (Polar)", pad=20)
    for i, freq_val in enumerate(unique_freqs):
        freq_data = aggr[aggr['freq'] == freq_val]
        phase_radians = np.mod(np.deg2rad(freq_data['phase_shift_deg']), 2 * np.pi)
        radius = 1 + freq_data['stimunit_id']
        marker = markers[i % len(markers)]
        ax[2].scatter(phase_radians, radius, 
                     s=50, color='green', alpha=0.6, marker=marker, label=f'{freq_val}Hz')
    ax[2].set_theta_zero_location('N')
    ax[2].set_theta_direction(-1)
    ax[2].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.1, 1.0))
    
    # 3.) LSB [nA]
    ax[3].set_title(f"3.) LSB, {stim_mode} mode [nA]\n(Linear scaling by resistance R={R/1000:g}kΩ)")
    ax[3].set_xlabel("Stim Unit ID")
    ax[3].set_ylabel("LSB [nA]")
    ax[3].set_xticks(stim_units)
    for i, freq_val in enumerate(unique_freqs):
        freq_data = aggr[aggr['freq'] == freq_val]
        marker = markers[i % len(markers)]
        ax[3].scatter(freq_data['stimunit_id'], freq_data[f"{stim_mode}_LSB_nA"], alpha=0.3, 
                      color='blue', edgecolor='none', s=30, marker=marker, label=f'{freq_val}Hz')
    ax[3].scatter(unique_med['stimunit_id'], unique_med[f"{stim_mode}_median_LSB_nA"], color='blue', 
                  s=200, marker='_', linewidths=2, zorder=5, label='Median')
    ax[3].tick_params(axis='x', which='both', rotation=90)
    ax[3].grid(True)
    ax[3].legend(fontsize=9)

    plt.tight_layout()

    # # copy to devices/headstage_devices basedir 
    # aggr_essential = aggr[['stimunit_id', f"{stim_mode}_zero_current_DAC", f"{stim_mode}_median_LSB_nA"]].drop_duplicates('stimunit_id')
    # aggr_essential[f"{stim_mode}_LSB_nA"] = aggr.groupby('stimunit_id')[f"{stim_mode}_LSB_nA"].transform('mean')
    # aggr_essential.to_csv(os.path.join(dirname, "..", "..", f"{stim_mode}_stimunit_characterization.csv"), index=False)
    
    plt.savefig(result_fullfname.replace(".csv", ".png"), dpi=300)
    # if debug:
    #     plt.show()
    plt.close()
    
def plot_all_DAC_sweeps(dirname, stim_units, zero_current_DAC_code=None, debug=False, label=None):
    # 4 rows, and according number of columns to get n stim_units subplots
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(4*8, 16), sharey=True, sharex=True)
    axes = axes.flatten()
    
    for stim_unit in (stim_units):
        res_fullfnames = glob.glob(os.path.join(dirname, f"results_StimUnit{stim_unit:02d}_*.csv"))
        if len(res_fullfnames) == 0:
            print(f"No result files found for stim unit {stim_unit}, skipping plot.")
            continue
        
        stim_unit_data = []
        for fullfname in res_fullfnames:
            res_df = pd.read_csv(fullfname)
            stim_unit_data.append(res_df)
        stim_unit_data = pd.concat(stim_unit_data, axis=0, ignore_index=True)
        # dac sweeps
        axes[stim_unit].scatter(stim_unit_data.DAC_val, stim_unit_data.peak_uV, alpha=0.4, 
                        s=20, label=f"StimUnit {stim_unit:02d} samples")
        # stim settings of calibration
        if zero_current_DAC_code is not None and stim_unit in zero_current_DAC_code.index:
            dac_code = zero_current_DAC_code.loc[stim_unit]
            axes[stim_unit].axvline(dac_code, color='red', linestyle='--', label=f"OLD Zero-Current DAC: {dac_code}")
        
        axes[stim_unit].set_title(f"Stim Unit {stim_unit:02d}", y=.95)
        axes[stim_unit].set_xlabel("DAC value", fontsize=10)
        axes[stim_unit].set_ylabel("Peak uV")
        axes[stim_unit].legend(fontsize=6)
        axes[stim_unit].axhspan(-25000, 25000, color='gray', alpha=0.15, zorder=0, label="±25k uV Target Range")
        [axes[stim_unit].spines[side].set_visible(False) for side in ['top', 'right', 'left', 'bottom']]
        
    # save in processed subdir
    plot_path = os.path.join(dirname, "processed", f"all_StimUnit_DAC_sweeps.png")
    plt.savefig(plot_path, dpi=300)
    return fig, axes
            
            
def check_zero_current_DAC_drift(headstage_name, stim_mode, stim_units=list(range(32)),
                                 delta=64, set_size=16, down_step_size=8, max_iterations=2,
                                 skip_stim=False, n_amplifiers=1, debug=False):
    # read current characterization CSV
    nas_dir = device_paths()[0]
    path = os.path.join(nas_dir, f"devices/headstage_devices/{headstage_name}")
    fullfname = os.path.join(path, f"{headstage_name}_stimunit_characterization.csv")
    print(pd.read_csv(fullfname).set_index("stimunit_id", drop=True))
    stim_settings = pd.read_csv(fullfname).set_index("stimunit_id", drop=True).loc[stim_units]
    
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
    rec_dir = f"{t}_{stim_mode}_check_zero_current_DAC_drift"
    # rec_dir = "2026-05-20_19.15_large_current_check_zero_current_DAC_drift"
    # rec_dir = "2026-06-30_08.13_large_current_check_zero_current_DAC_drift"
    rec_path = os.path.join(path, "recordings", rec_dir)
    if not skip_stim:
        setup_stim_unit_characterization(os.path.join(rec_path, "processed"))
        for stim_unit in stim_units:
            prior = stim_settings.loc[stim_unit, f"{stim_mode}_zero_current_DAC"]
            char_stim_units(rec_path, stim_mode=stim_mode, n_amplifiers=n_amplifiers, 
                            stim_units=[stim_unit], debug=debug, down_step_size=down_step_size,
                            start_centered_around=prior, delta=delta, set_size=set_size,
                            max_iterations=max_iterations, fast_setup=True)
    result = pd.read_csv(os.path.join(rec_path, "processed", "StimUnits_characterization.csv"), index_col=0)
    old_result = stim_settings.loc[:, f"{stim_mode}_zero_current_DAC"]
    
    fig, ax = plot_all_DAC_sweeps(rec_path, stim_units=stim_units, zero_current_DAC_code=old_result, 
                        debug=False, label=f"{stim_mode} DAC sweep")
    plt.suptitle(f"{rec_dir}", fontsize=16, fontweight="bold")
    # draw result zero current DAC codes into the axes
    for stim_unit in stim_units:
        if stim_unit not in result.index:
            continue
        dac_code = result.loc[stim_unit, "zero_current_DAC_code"]
        ax[stim_unit].axvline(dac_code, color='green', linestyle=':', linewidth=2,
                              label=f"NEW Zero-Current DAC: {dac_code}")
        ax[stim_unit].legend(fontsize=6)
    
    # save in processed subdir
    plot_fullfname = os.path.join(rec_path, "processed", f"all_StimUnit_DAC_sweeps.png")
    plt.savefig(plot_fullfname, dpi=300)
    print(plot_fullfname)
    return rec_dir

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
    debug = True
    L = Logger()
    L.init_logger(None, None, "WARNING")
    
    STIM_MODE = 'large_current'  # 'small_current' or 'large_current'
    # STIM_MODE = 'small_current'  # 'small_current' or 'large_current'
    # headstage_name = "MEA1K27"
    
    # copy over the new values to final csv
    
    #     
    # rec_dir = check_zero_current_DAC_drift(headstage_name, 'large_current', delta=32, down_step_size=2, skip_stim=False, max_iterations=3, 
    #                              debug=debug, n_amplifiers=3,  )#stim_units=[31])
    # check_zero_current_DAC_drift(headstage_name, 'small_current', delta=64, down_step_size=4, max_iterations=2, n_amplifiers=3)
    # exit()
    
    
    
    nas_dir = device_paths()[0]
    # nas_dir = '/home/houmanjava/nas_imitation/'
    device_dir = "devices/well_devices/4988/recordings"
    # device_dir = f"devices/headstage_devices/{headstage_name}/recordings"
    R = 1000_000  # 1 MOhm
    # R = 1_024_000  # 1 MOhm
    sine_ampl_DAC_units = 3 if STIM_MODE == 'large_current' else 30
    
    sine_ampl_DAC_units = 2
    stim_units = list(range(32))
    # stim_units = [1,2,3,4]

    t = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
    rec_dir = f"{t}_{R=}_CharStim_ShowCase2_gain1_{STIM_MODE}"
    # rec_dir = "2026-05-20_19.15_large_current_check_zero_current_DAC_drift" 
    # rec_dir = "2026-05-20_20.19_small_current_check_zero_current_DAC_drift"
    # rec_dir = "2026-05-28_09.46_R=100000_CharStimUnitsLateLateMay_small_current"
    # rec_dir = "2026-06-04_15.48_R=1000000_CharStimUnitsJune_large_current"
    # rec_dir = '2026-07-12_17.39_R=1000000_CharStim_NEWFAST+LSB_large_current'
    # rec_dir = '2026-07-12_17.20_R=1000000_CharStim_NEWFAST+LSB_small_current'
    # rec_dir = "2026-07-15_11.43_large_current_check_zero_current_DAC_drift"
    
    # rec_dir = '2026-07-11_16.09_R=1000000_CharStim_NEWFAST+LSB_small_current'
    # rec_dir = '2026-08-29_09.30_large_current_check_zero_current_DAC_drift'
    # rec_dir = '2026-09-11_16.36_large_current_check_zero_current_DAC_drift'
    # rec_dir = '2026-09-11_21.11_large_current_check_zero_current_DAC_drift'
    # rec_dir = "2026-10-05_08.57_R=1000000_CharStim_ShowCase_large_current"
    
    # rec_dir = "2026-10-25_19.46_R=1024000_CharStim_ShowCase2_large_current"
    # rec_dir = "2026-10-25_22.37_R=1024000_CharStim_ShowCase2_small_current"
    
    
    # rec_dir = "2026-10-26_17.32_R=100000_CharStim_ShowCase2_large_current"

    full_path = os.path.join(nas_dir, device_dir, rec_dir)
    
    
    # transfer_stimunit_characterization(full_path, stim_mode=STIM_MODE, 
    #                                     headstage_name=headstage_name)
    # exit()


    n_amplifiers = 2
    # Sweep the DAC values and continuously write the Best DAC fit to StimUnits_characterization.csv
    char_stim_units(full_path, stim_mode=STIM_MODE, n_amplifiers=n_amplifiers, gain=1,
                    # start_centered_around=512, delta=256, set_size=16, max_iterations=4, 
                    start_centered_around=512, delta=32, set_size=16, max_iterations=4, 
                    stim_units=stim_units, debug=debug, rec_current_externally=True)
                    
    exit()
    n_amplifiers = 6
    # Calculate the Current LSB
    # char_current_lsb(full_path, R=R, sine_ampl_DAC_units=sine_ampl_DAC_units, stim_mode=STIM_MODE,
    #                  stim_units=stim_units, n_amplifiers=n_amplifiers, freq=1000,
    #                  ncycles=400, rec_current_externally=True, debug=debug)
    
    
    
    # (char_current_lsb evaluates live in a thread; eval_current_lsb below is only
    #  needed to re-evaluate an existing recording directory)
    # Bigger LSB set at several recording gains and frequencies, goes to full_path/LSB_sweep
    # (zero current DAC codes come from char_stim_units above). Sine amplitude is scaled down
    # with gain, as the recording full scale shrinks.
    conditions = make_lsb_conditions(gains=[7, 112, 512], freqs=[1000, 10],
                                     sine_ampl_DAC_units=sine_ampl_DAC_units,
                                     ncycles_by_freq={1000: 400,  10: 10,})
    char_current_lsb_sweep(full_path, R=R, stim_units=stim_units, n_amplifiers=3,
                           stim_mode=STIM_MODE, conditions=conditions,
                           rec_current_externally=True, debug=debug)



    # Evaluate the Final Result
    # eval_current_lsb(full_path, R=R, sine_ampl_DAC_units=sine_ampl_DAC_units, 
    #                  stim_mode=STIM_MODE, stim_units=stim_units, freq=freq, 
    #                  ncycles=ncycles, debug=debug,
    #                  )
    
    # # # copy over the new values to final csv
    # transfer_stimunit_characterization(full_path, stim_mode=STIM_MODE, 
    #                                    headstage_name=headstage_name)



if __name__ == "__main__":
    main()