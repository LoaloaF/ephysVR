"""Impedance of single MEA1K electrodes: sine current through one stim unit, voltage on the
electrode's amplifier. Each recording gets a sidecar csv (stim settings, gain, and all columns
of its target row, e.g. connectivity). Postprocessing runs in a 2nd thread and only needs the
recording dir, so reprocessing never has to repeat the electrode selection.
"""
import os
import re
import sys
from glob import glob
import time
import functools

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # debug plots are drawn in the postproc thread, files only
import matplotlib.pyplot as plt
from datetime import datetime

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

from mea1k_modules.mea1k_config_utils import start_saving, stop_saving
from mea1k_modules.mea1k_config_utils import attampt_connect_el2stim_unit, create_stim_sine_sequence
from mea1k_modules.mea1k_config_utils import turn_on_stimulation_units, turn_off_stimulation_units
from mea1k_modules.mea1k_config_utils import shift_DAC, reset_MEA1K
from mea1k_modules.mea1k_config_utils import get_maxlab_saving, get_maxlab_array

from mea1k_modules.mea1k_raw_preproc import read_raw_data, read_stim_DAC
from mea1k_modules.live_postproc import LivePostproc, save_live_fig
from mea1k_connectivity_scripts.signal_helpers import extract_amplitude
import mea1k_modules.mea1k_visualizations as vis

from PicoRecorder import PicoRecorder, mv_to_current_uA, DEFAULT_SENSE_GAIN

# ======== TARGETS: which electrodes to measure ========
# A targets table is a DataFrame with an `electrode` column. All other columns (connectivity,
# shank_id, ...) are copied into the sidecar csv of each recording and end up in the results.

def targets_from_connectivity_rec(full_recdir):
    """Electrodes of a 10mVFPGA connectivity recording (headstage-only tests, no implant mapping)."""
    conn = pd.read_csv(os.path.join(full_recdir, "processed", "extr_connectivity.csv"))
    conn = conn.rename(columns={"el": "electrode"}).sort_values("connectivity", ascending=False)
    return conn[["electrode", "connectivity"]].drop_duplicates("electrode")

def targets_from_implant_mapping(implant_name, shank_subset=None, every_pad=None, every_pi_el=False):
    """Electrodes of a bonded implant. every_pad: only the n best electrodes per pad (ranks),
    every_pi_el: only the best electrode per polyimide electrode."""
    nas_dir = device_paths()[0]
    mapping = pd.read_csv(os.path.join(nas_dir, "devices", "implant_devices", implant_name, "bonding",
                                       f"bonding_mapping_{implant_name}.csv"))
    mapping = mapping.dropna(subset=["mea1k_el"])
    if shank_subset is not None:
        mapping = mapping[mapping.shank_id.isin(shank_subset)]
    if every_pad is not None:
        mapping = mapping[mapping.pad_connectivity_rank.isin(every_pad)]
    elif every_pi_el:
        mapping = mapping[mapping.el_connectivity_rank == 1]
    targets = mapping.rename(columns={"mea1k_el": "electrode", "mea1k_connectivity": "connectivity"})
    targets["electrode"] = targets.electrode.astype(int)
    targets = targets.sort_values("connectivity", ascending=False)
    return targets[["electrode", "connectivity", "shank_id", "pad_id", "pad_metal"]].drop_duplicates("electrode")

def targets_from_short_islands(connectivity_graph_csv, min_island_size=3):
    """Electrodes of shorted islands with >= min_island_size electrodes, from the
    extracted_connectivity_graph.csv of a TileShorts recording."""
    graph = pd.read_csv(connectivity_graph_csv)
    graph = graph[graph.short_island_id.notna()]
    island_els = pd.concat([graph[["short_island_id", "source"]].rename(columns={"source": "electrode"}),
                            graph[["short_island_id", "target"]].rename(columns={"target": "electrode"})])
    island_els = island_els.drop_duplicates().rename(columns={"short_island_id": "island_id"})
    island_size = island_els.groupby("island_id").electrode.transform("nunique")
    targets = island_els[island_size >= min_island_size].astype({"electrode": int})
    return targets.drop_duplicates("electrode").sort_values("electrode")

def index_stimulation_configs(configs_dir):
    """Electrode -> stim unit + config file, from configs_dir/StimUnitXX/el_config_ElXXXXX_*.cfg"""
    rows = []
    for config_fullfname in glob(os.path.join(configs_dir, "StimUnit*", "*.cfg")):
        el = int(re.search(r"El(\d+)_", os.path.basename(config_fullfname)).group(1))
        stim_unit = int(re.search(r"StimUnit(\d+)", os.path.dirname(config_fullfname)).group(1))
        rows.append((el, stim_unit, config_fullfname))
    config_index = pd.DataFrame(rows, columns=["electrode", "stim_unit", "config_fullfname"])
    return config_index.sort_values("config_fullfname").drop_duplicates("electrode")

def select_targets(targets, config_index, connectivity_threshold=None, stim_units=None,
                   sample_fraction=None, negative_ctrl=False):
    """Adds stim unit + config file to each target and filters. Electrodes without a stim
    config are reported and dropped. negative_ctrl: adds the lowest-connectivity electrode
    of each stim unit (from the unfiltered targets), marked with negative_ctrl=True."""
    targets = targets.merge(config_index, on="electrode", how="left")
    no_config = targets.config_fullfname.isna()
    if no_config.any():
        print(f"{no_config.sum()} target electrode(s) without stim config, skipped: "
              f"{targets.electrode[no_config].tolist()}")
    targets = targets[~no_config].astype({"stim_unit": int})
    if stim_units is not None:
        targets = targets[targets.stim_unit.isin(stim_units)]
    candidates = targets

    if connectivity_threshold is not None:
        targets = targets[targets.connectivity > connectivity_threshold]
        print(f"{len(targets):,} electrodes with connectivity > {connectivity_threshold}")
    if sample_fraction is not None:
        targets = targets.sample(frac=sample_fraction)
        print(f"Randomly sampled {len(targets):,} electrodes ({sample_fraction:.0%})")
    targets = targets.assign(negative_ctrl=False)
    if negative_ctrl:
        neg = candidates.sort_values("connectivity").groupby("stim_unit").head(1)
        neg = neg[~neg.electrode.isin(targets.electrode)].assign(negative_ctrl=True)
        targets = pd.concat([targets, neg])
    targets = targets.sort_values(["stim_unit", "electrode"]).reset_index(drop=True)
    print(f"Measuring {len(targets):,} electrodes on {targets.stim_unit.nunique()} stim units")
    return targets

# ======== RECORDING ========

def _stimulate_electrode(target, full_recdir, s, seq, pico_rec, stim_info):
    """Record one electrode while the sine is played through its stim unit. The sidecar csv
    (stim_info + target columns) is written after stop_saving: "recording finished" signal."""
    el, stim_unit = int(target.electrode), int(target.stim_unit)
    array = get_maxlab_array()
    array.load_config(target.config_fullfname)

    fname = os.path.basename(target.config_fullfname).replace(".cfg", "")
    start_saving(s, dir_name=full_recdir, fname=fname, legacy=True)
    # everything up to stop_saving in try/finally: MaxWell keeps recording until saving is
    # stopped, an exception or Ctrl+C in between would leave it running (server crashes)
    try:
        if pico_rec:
            pico_rec.start(os.path.join(full_recdir, f"{fname}.npz"))

        time.sleep(.1)
        attampt_connect_el2stim_unit(el, array, used_up_stim_units=[],
                                     with_download=True)

        seq_duration = (stim_info["stim_sequence_ncycles"] / stim_info["stim_sequence_frequency_Hz"])
        print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ El{el} with StimUnit{stim_unit} for {seq_duration:.1f}s")
        seq.send()
        time.sleep(seq_duration)
        time.sleep(.1)  # wait a bit before disconnecting the electrode from stimulation
        array.disconnect_electrode_from_stimulation(el)
        array.download()
        time.sleep(.3)  # wait a bit before disconnecting the electrode from stimulation
    finally:
        stop_saving(s)
        if pico_rec:
            pico_rec.stop()  # make sure to stop the PicoRecorder to save the file properly
    array.close()

    sidecar = pd.concat([pd.Series(stim_info), target.drop(list(stim_info), errors="ignore")])
    sidecar.to_frame().T.to_csv(os.path.join(full_recdir, f"{fname}.csv"), index=False)
    return f"{fname}.raw.h5"

def _record_targets(full_recdir, targets, s, pico_rec, pp, stim_settings, stim_mode, gain,
                    dac_id, dac_sine_amplitude, stim_sequence_frequency_Hz,
                    stim_sequence_ncycles, dac_offset):
    for stim_unit, unit_targets in targets.groupby("stim_unit"):
        # set the DAC to zero current, create the sine sequence around it
        dac_code = int(stim_settings.loc[stim_unit, f"{stim_mode}_zero_current_DAC"])
        shift_DAC(dac_code+dac_offset)
        seq = create_stim_sine_sequence(dac_id=dac_id, amplitude=dac_sine_amplitude,
                                        f=stim_sequence_frequency_Hz,
                                        ncycles=stim_sequence_ncycles,
                                        center_around=dac_code+dac_offset)
        turn_on_stimulation_units([stim_unit], dac_id=dac_id, mode=stim_mode)
        stim_info = {
            "stim_unit": stim_unit,
            "stim_mode": stim_mode,
            "LSB_nA": stim_settings.loc[stim_unit, f"{stim_mode}_median_LSB_nA"].item(),
            "dac_id": dac_id,
            "dac_code": dac_code,
            "dac_offset": dac_offset,
            "stim_sequence_DAC_amplitude": dac_sine_amplitude,
            "stim_sequence_frequency_Hz": stim_sequence_frequency_Hz,
            "stim_sequence_ncycles": stim_sequence_ncycles,
            "gain": gain,
        }
        try:
            for i, (_, target) in enumerate(unit_targets.iterrows()):
                print(f"\nStimUnit{stim_unit}: electrode {i+1}/{len(unit_targets)} "
                      f"| [postproc] {pp.status()}", flush=True)
                fname = _stimulate_electrode(target, full_recdir, s, seq, pico_rec, stim_info)
                pp.add(fname)
        finally:
            turn_off_stimulation_units([stim_unit])
        # all_impedance.csv + live_figures/impedance_analysis_live.png, in the postproc thread
        pp.run_in_thread(lambda: save_impedance_results(full_recdir, pp.results_df()))

# ======== POSTPROCESSING ========

def _extract_sine_amplitude(dir_name, fname, min_band, max_band, gain, debug=True,):
    amplifier = int(fname.split("Ampl")[-1].split("_")[0][:-7])
    data = read_raw_data(dir_name, fname, convert2uV=True,
                            subtract_dc_offset=False, override_gain=gain)
    dac = read_stim_DAC(dir_name, fname)
    # analysis window comes from the DAC onset/offset; notch + demean on by default
    mean_ampl, phase_shift, _, _ = extract_amplitude(
        data[amplifier].astype(float),
        sampling_rate=20_000,
        highpass=100,
        notch=True,
        min_band=min_band, max_band=max_band,
        zoom_from=-150, zoom_to=150,
        dac=dac.astype(float),
        use_dac_interval=True,
        name=fname.replace('.raw.h5', ''),
        plot_fname='debug_amplifier',
        debug=debug,
    )
    return mean_ampl, phase_shift

def _extract_pico_current(dir_name, fname, min_band, max_band, name, log, debug=False):
    """Current amplitude [uA] from the picoscope recording, NaNs if there is none."""
    pico_fullfname = os.path.join(dir_name, fname.replace('.raw.h5', '.npz'))
    if not os.path.exists(pico_fullfname):
        return np.nan, np.nan, np.nan
    pico_data = np.load(pico_fullfname)
    if len(pico_data['time_s']) < 2:
        log(f"{fname}: picoscope data too short")
        return np.nan, np.nan, np.nan
    pico_current_uA = mv_to_current_uA(pico_data['mv'], DEFAULT_SENSE_GAIN)
    pico_ampl, _, baseline_dc, sine_dc = extract_amplitude(
        pico_current_uA,
        sampling_rate=20_000,
        min_band=min_band, max_band=max_band,
        estimate_amplitude_from=3600,
        estimate_amplitude_to=11000,
        lowpass=3000.0,
        name=name,
        ylim=(-0.28, 0.28),
        debug=debug,
        plot_fname='debug_picoscope_signal',
        save_to_dir=os.path.join(dir_name, "processed"),
    )
    return pico_ampl, baseline_dc, sine_dc

def process_impedance_recording(rec_dir, fname, log, fallback_gain=None, fallback_targets=None,
                                debug=False):
    """process_fn for LivePostproc: one recording -> one row of impedance results.
    Recordings from before gain/target columns were stored in the sidecar use fallback_gain
    and fallback_targets (targets table indexed by electrode) for what is missing."""
    sidecar_fullfname = os.path.join(rec_dir, fname.replace(".raw.h5", ".csv"))
    if not os.path.exists(sidecar_fullfname):
        log(f"{fname}: no sidecar csv, recording was interrupted?")
        return None
    stim_info = pd.read_csv(sidecar_fullfname).iloc[0].drop("Unnamed: 0", errors="ignore")
    if "gain" not in stim_info:
        if fallback_gain is None:
            log(f"{fname}: no gain in sidecar csv and no fallback_gain given")
            return None
        stim_info["gain"] = fallback_gain
    if fallback_targets is not None and stim_info.electrode in fallback_targets.index:
        missing = fallback_targets.loc[stim_info.electrode].drop(stim_info.index, errors="ignore")
        stim_info = pd.concat([stim_info, missing])
    if "connectivity" not in stim_info:
        stim_info["connectivity"] = np.nan

    freq = stim_info.stim_sequence_frequency_Hz
    rng = max(1, freq//100) # allow 1% frequency deviation
    ampl, phase_shift = _extract_sine_amplitude(rec_dir, fname, min_band=freq-rng,
                                                max_band=freq+rng, gain=stim_info.gain,
                                                debug=debug)
    pico_name = (f"StimUnit={stim_info.stim_unit}, DAC={stim_info.get('dac_code', '?')}"
                 f"+{stim_info.get('dac_offset', '?')}, el={int(stim_info.electrode):05d}")
    ext_current_uA, baseline_dc, sine_dc = _extract_pico_current(
        rec_dir, fname, freq-rng, freq+rng, pico_name, log, debug=debug)

    amplitude_uA = stim_info.stim_sequence_DAC_amplitude * stim_info.LSB_nA / 1000  # in uA
    stim_res = pd.Series({
        "amplitude_uV": ampl,
        "current_uA": amplitude_uA,
        "external_current_uA": ext_current_uA,
        "external_current_pre_DC_offset": baseline_dc,
        "external_current_DC_offset": sine_dc,
        # micro volt / (LSB in nA * sine amplitude in DAC units / 1000 to convert to uA)) = Ohm
        "impedance_Ohm": ampl / amplitude_uA,  # in Ohm
        "external_impedance_Ohm": ampl / ext_current_uA,
        "phase_shift_deg": phase_shift,
    })
    res = pd.concat([stim_info, stim_res])
    bar = "=" * 60
    log(f"{fname}\n{bar}\n  El{int(res.electrode):05d} StimUnit{int(res.stim_unit):02d}:  "
        f"IMPEDANCE {res.impedance_Ohm/1e3:,.1f} kOhm   (phase {phase_shift:.1f} deg)\n{bar}\n"
        f"{res.drop(LOG_HIDDEN_FIELDS, errors='ignore').map(comma_formatter).to_string()}")
    return pd.DataFrame([res.to_dict()])

# kept in the results csv, just not printed to the postproc log
LOG_HIDDEN_FIELDS = ["dac_id", "dac_code", "LSB_nA", "stim_sequence_frequency_Hz",
                     "stim_sequence_ncycles", "config_fullfname", "negative_ctrl"]

def comma_formatter(x):
    # Check if the value is a number (and not a boolean)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        # If it's an integer or a float representing a whole number (like 1000.0)
        if isinstance(x, int) or x.is_integer():
            return f"{int(x):,}"
        else:
            return f"{x:,.2f}"  # Change .2f to adjust decimal places
    return str(x)   

def summarize_impedance(results, last_result, target_Ohm=100_000, tolerance=.15):
    imp = results.impedance_Ohm.astype(float)
    n_ok = imp.between(target_Ohm*(1-tolerance), target_Ohm*(1+tolerance)).sum()
    msg = f"{n_ok}/{len(imp)} within {target_Ohm/1e3:,.0f} kOhm \u00b1{tolerance:.0%}"
    if last_result is not None:
        msg += f", last {last_result.impedance_Ohm.iloc[0]/1e3:,.0f} kOhm"
    return msg

def save_impedance_results(full_recdir, results, final=False):
    """processed/all_impedance.csv + the impedance analysis plot to live_figures/
    (final: also to processed/). Called after every stim unit and at the end."""
    if len(results) == 0:
        return
    processed_dir = os.path.join(full_recdir, "processed")
    results.to_csv(os.path.join(processed_dir, "all_impedance.csv"), index=False)
    data_plotting = results.copy()
    if "pad_metal" not in data_plotting.columns:
        data_plotting['pad_metal'] = 1
    fig = vis.plot_impedance_analysis(processed_dir, data_plotting, skip_bottom=False,
                                      color_by_metal=False, use_imp_connectivity=True)
    save_live_fig(fig, "impedance_analysis_live", dpi=100)
    if final:
        fig.savefig(os.path.join(processed_dir, "all_impedance_analysis.png"), dpi=300)
    plt.close(fig)

def measure_impedance(full_recdir, targets, stim_settings, stim_mode, gain, dac_id,
                      dac_sine_amplitude, stim_sequence_frequency_Hz, stim_sequence_ncycles,
                      dac_offset=0, rec_current_externally=False, skip_stimulation=False,
                      reprocess=False, fallback_targets=None, target_imp_Ohm=100_000,
                      debug=False):
    """Stimulate + record all targets (see select_targets), each recording is postprocessed
    in a 2nd thread. skip_stimulation: only (re)process the recordings in full_recdir."""
    s = pico_rec = None
    if not skip_stimulation:
        reset_MEA1K(gain=gain, enable_stimulation_power=True)
        s = get_maxlab_saving()
        if rec_current_externally:
            pico_rec = PicoRecorder(voltage_range="5V", resolution="16BIT", sample_interval_us=50)
    elif not os.path.exists(full_recdir):
        raise FileNotFoundError(f"Nothing to postprocess, no such recording dir: {full_recdir}")

    # recordings from before gain was stored in the sidecar csv fall back to `gain`
    process_fn = functools.partial(process_impedance_recording, fallback_gain=gain,
                                   fallback_targets=fallback_targets, debug=debug)
    summarize_fn = functools.partial(summarize_impedance, target_Ohm=target_imp_Ohm)
    pp = LivePostproc(full_recdir, process_fn, summarize_fn, name="postproc_record_imp",
                      reprocess=reprocess)
    pp.start()
    try:
        if not skip_stimulation:
            _record_targets(full_recdir, targets, s, pico_rec, pp, stim_settings, stim_mode, gain,
                            dac_id, dac_sine_amplitude, stim_sequence_frequency_Hz,
                            stim_sequence_ncycles, dac_offset)
    finally:
        results = pp.finish()
        # close the picoscope once, after all stim units are done
        if pico_rec is not None:
            pico_rec.close()

    save_impedance_results(full_recdir, results, final=True)
    if len(results):
        print(f"Saved {os.path.join(full_recdir, 'processed', 'all_impedance.csv')} + all_impedance_analysis.png")
    return results

def main():
    np.random.seed(42)
    L = Logger()
    L.init_logger(None, None, "WARNING")
    
    # ======== PARAMETERS ========
    nas_dir = device_paths()[0]
    # headstage_name = "MEA1K22"
    # implant_name = "260611_MEA1K24_S1688pad14shankB5"
    # headstage_name = "MEA1K23"
    # implant_name = "260602_MEA1K23_S844pad8shankB5"
    headstage_name = "MEA1K27"
    # implant_name = "260625_MEA1K23_S1688pad14shankB6"
    # implant_name = "260715_MEA1K23_J1688pad2shankB1"
    # implant_name = "260703_MEA1K23_S844pad8shankB1"
    
    # subdir = f"devices/implant_devices/{implant_name}/recordings"
    subdir = f"devices/headstage_devices/{headstage_name}/recordings"
    configs_basepath = f"mea1k_configs/single_el2stimunit_configs"
    stimulater_settings_path = f"devices/headstage_devices/{headstage_name}/{headstage_name}_stimunit_characterization.csv"
    stim_settings = pd.read_csv(os.path.join(nas_dir, stimulater_settings_path)).set_index("stimunit_id", drop=True)
    
    gain = 7
    target_imp_Ohm = 100_000 # live readout: n electrodes within this impedance +-15%
    dac_id = 0
    debug = False
    # ======== PARAMETERS ========
    # stim_mode = 'small_current'
    stim_mode = 'large_current'
    stim_sequence_frequency_Hz = 1000
    stim_sequence_ncycles = 400
    dac_sine_amplitude = 3 if stim_mode == 'large_current' else 30

    
    t = datetime.now().strftime("%Y-%m-%d_%H.%M")
    # rec_dir = 'test'
    # rec_dir = f"{t}_Imp_Bond_extCur10KSensePico_PT4_{stim_mode}_f{stim_sequence_frequency_Hz:04d}Hz"
    # rec_dir = f"{t}_ +0DAC_PT4_{stim_mode}_f{stim_sequence_frequency_Hz:04d}Hz"

    rec_name = f'2026-06-02_12.24_8Sh4SilverPaint_VRefFPGA_ampl15_PT1_2.6mm'
    rec_name = f'2026-06-02_12.40_8Sh4SilverPaint_VRefFPGA_ampl15_PT1_2.2mm'
    rec_name = f'2026-06-02_12.55_8Sh4SilverPaint_VRefFPGA_ampl15_PT1_1.9mm'
    rec_name = f'2026-06-02_15.00_8Sh4SilverPaint_VRefFPGA_ampl15_PT1_1.7mm'
    rec_name = f'2026-06-25_13.59_Bond5_VrefFPGAStim_ampl15_PT4_1.7mm'
    rec_name = '2026-07-13_15.50_JDesignFirstSoldered2_VRef15_PT2'
    rec_name = '2026-07-15_07.55_JDesignFirstSoldered2_VRef15_PT3'
    rec_name = '2026-07-16_16.45_JDesignFirstSoldered2_VRef15_PT4' # tightened to check the bottom half of the device, does imp improve or worsen with pressure?  
    rec_name = '2026-07-21_14.47_JDesignFirstSoldered2_VRef15_PT5' 
    # rec_name = '2026-07-15_11.38_ReEtched14ShankReGoldPlated_VRef15_PT1'
    # rec_name = '2026-07-15_12.31_ReEtched14ShankReGoldPlated_VRef15_PT2'
    # rec_name = f'2026-07-15_13.59_ReEtched14ShankReGoldPlated_VRef15_PT3'
    
    # rec_name = f'2026-06-18_09.48_Bond4_VrefFPGAStim_ampl15_14shanksSilver_PT0'
    # rec_name = f'2026-06-18_09.48_Bond4_VrefFPGAStim_ampl15_14shanksSilver_PT0'
    
    # rec_name = f'2026-06-23_09.53_Bond4_VrefFPGAStim_ampl15_14shanksSilver_PT0'
    # rec_name = f'2026-06-23_11.28_Bond4_VrefFPGAStim_ampl15_PT1_2.2mm'
    # rec_name = f'2026-06-23_13.15_Bond4_VrefFPGAStim_ampl15_PT2_2.0mm'
    # rec_name = f'2026-06-24_20.24_Bond4_VrefFPGAStim_ampl15_PT3_1.8mm'
    # rec_name = f'2026-06-24_21.41_Bond4_VrefFPGAStim_ampl15_PT4_1.6mm'
    
    # rec_name = f'2026-07-10_14.03_ACF_VRefAmpl15_Try9_PT1'
    # rec_name = f'2026-07-16_14.58_ACF_VRefAmpl15_Try9_PTT3'
    # rec_name = f'2026-07-20_15.09_ACF_PIEl_VRefAmpl15_Try10_PT4'
    # rec_name = "2026-07-20_16.17_ACF_PIEl_VRefAmpl15_Try10_PT5_2mm"
    
    rec_name = f'2026-08-28_20.05_3rdTry_ACF_J2_61umP_VRef15_PT1'
    rec_name = f'2026-08-29_22.44_7thTry_ACF_J2_36umP_VRef15_PT1'
    rec_name = f'2026-08-30_20.52_noACF_J2_61umP_VRef15_PT1'
    rec_name = f'2026-08-30_21.15_noACF_J2_61umP_VRef15_PT1'

    rec_name = f'2026-09-01_10.12_noACFTry2_J2_36umP_VRef15_PT1_1.9mm'
    rec_name = f'2026-09-01_11.18_noACFTry2_J2_36umP_VRef15_PT3_1.8mmcorner'
    
    # rec_name = f'2026-09-02_15.28_noACFNoEcoFlexTry3_J2_61umP_VRef15_PT2'
    # rec_name = "2026-09-02_16.33_noACFNoEcoFlexTry3_J2_61umP_VRef15_PT3"
    
    # rec_name = "2026-09-02_18.21_noACFTry4_J2_61umP_VRef15_PT2"
    
    rec_name = f'2026-09-04_09.55_J2_2x8x56umP_PT1_10mVFPGA'
    rec_name = f'2026-09-08_11.59_2x8x56umP_PT3_10mVFPGA'
    
    rec_name = f'2026-09-08_12.09_2x8x56umP_PT4_10mVFPGA' # tiny tightenting of bottom right screw
    rec_name = f'2026-09-08_12.30_2x8x56umP_PT4_10mVFPGA' # added more ringer

    rec_name = f'2026-09-08_14.13_test_100k_35umP_PT2_10mVFPGA'
    
    rec_name = f'2026-09-11_12.59_test2_100k_35umP_PT2_2.8mm_10mVFPGA'
    
    rec_name = f'2026-09-11_22.25_test4_100k_61umP_PT4_2.6mm_10mVFPGA'

    rec_name = f'2026-09-12_20.40_test8_100k_36umP_PT3_10mVFPGA'
    
    # rec_name = f"2026-09-12_00.17_test6_100k_61umP_PT2_10mVFPGA" 
    
    rec_name = f'2026-09-14_11.57_2x8x56umP_16`of16_100K_PT4_10mVFPGA'
    
    rec_name = f'2026-10-15_14.21_test10_100k_36umP_PT4_reSoldered_20mVFPGAExt'
    
    rec_name = f'2026-11-01_20.17_test12_100k_36umP_56umPE_PT2_10mVFPGA' # bottom device only 4-5 electrodes? 
    
    
    # rec_name = f'2026-10-21_14.45_test11_100k_36umP_PT1_10mVFPGA' 
    # rec_name = f'2026-10-22_13.58_test11_100k_36umP_PT3_10mVFPGA' 
    # rec_name = f'2026-10-22_15.21_test11_100k_36umP_PT4_10mVFPGA' 
    
    # rec_name = f'2026-10-25_17.03_test11_100k_36umP_PT7_10mVFPGA' # 
    
    
    
    
    
    # ======== TARGETS ========
    # headstage-only test, no implant mapping: electrodes of a 10mVFPGA connectivity recording
    targets = targets_from_connectivity_rec(os.path.join(nas_dir, subdir, rec_name))
    # bonded implant:
    # targets = targets_from_implant_mapping(implant_name, shank_subset=None, every_pad=None, every_pi_el=False)
    # electrodes of shorted islands (TileShorts recording):
    # targets = targets_from_short_islands(os.path.join(nas_dir, subdir, "<TileShorts rec>",
    #                                      "processed", "extracted_connectivity_graph.csv"), min_island_size=3)
    # explicit list:
    # targets = pd.DataFrame({"electrode": [16752, 7555]})
    # region subset, e.g. bottom rows only:
    # targets = targets[targets.electrode // 220 > 60]

    rec_current_externally = False
    skip_stimulation = True  # if True, only (re)process the recordings in postproc_rec_dir
    reprocess = True         # if True, also redo recordings that already have a processed/ output
    postproc_rec_dir = "2026-11-02_12.33_test12_100k_36umP_56umPE_PT2_ImpNew"





    # ======== PARAMETERS ========
    
    # ---- consistent naming theme: {t}_{name}_{tightening_str}_<recording type> ----
    # update name/tightening_str each session; recording type stays "Imp" for this script
    
    # name = "2x8x56umP"
    # name = "test_100k_35umP_post64C_pressTest"
    # tightening_str = "PT5" # tiny bit on left side again
    
    # name = "test6_100k_61umP"
    # tightening_str = "PT2"
    # name = "test4_100k_61umP"
    # tightening_str = "PT5_2.6mm"
    
    # name = "test8_100k_36umP"
    # tightening_str = "PT3"
    
    name = "2x8x56umP_1of16_100k"
    tightening_str = "PT4"
    
    name = "test10_100k_36umP"
    tightening_str = "PT4_resoldered"
    
    name = "test11_100k_36umP"
    tightening_str = "PT7"
    
    name = "test12_100k_36umP_56umPE"
    tightening_str = "PT2"

    rec_dir = f'{t}_{name}_{tightening_str}_ImpNew'
    if skip_stimulation:
        rec_dir = postproc_rec_dir
    else:
        config_index = index_stimulation_configs(os.path.join(nas_dir, configs_basepath))
        targets = select_targets(targets, config_index,
                                 connectivity_threshold=.4,
                                 # stim_units=list(range(32)),
                                 stim_units=list(range(16,32)),
                                 sample_fraction=1,
                                 negative_ctrl=False)

    measure_impedance(os.path.join(nas_dir, subdir, rec_dir), targets,
                      stim_settings, stim_mode, gain=gain, dac_id=dac_id,
                      dac_sine_amplitude=dac_sine_amplitude,
                      stim_sequence_frequency_Hz=stim_sequence_frequency_Hz,
                      stim_sequence_ncycles=stim_sequence_ncycles,
                      dac_offset=0,
                      rec_current_externally=rec_current_externally,
                      skip_stimulation=skip_stimulation,
                      reprocess=reprocess,
                      target_imp_Ohm=target_imp_Ohm,
                      # old recordings without target columns in their sidecar csv
                    #   fallback_targets=targets.set_index("electrode"),
                      debug=debug)
    # cp_rec_dir_to_implant_dir(full_recdir, implant_name)
        
if __name__ == "__main__":
    main()
