"""Minimal impedance test for electrodes in sufficiently large shorted islands.

The electrodes are loaded from a TileShorts connectivity graph. Only electrodes
belonging to islands containing at least three electrodes are tested.
"""
import os
import re
import sys
import time
from glob import glob
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

from mea1k_modules.mea1k_config_utils import (
    start_saving, stop_saving, attampt_connect_el2stim_unit,
    create_stim_sine_sequence, turn_on_stimulation_units,
    turn_off_stimulation_units, shift_DAC, reset_MEA1K,
    get_maxlab_saving, get_maxlab_array, get_headstage_stim_settings,
)
from mea1k_modules.mea1k_raw_preproc import read_raw_data, read_stim_DAC
from mea1k_connectivity_scripts.signal_helpers import extract_amplitude
import mea1k_modules.mea1k_visualizations as vis


def get_short_island_electrodes(connectivity_csv, min_island_size=3):
    """Return electrodes in graph islands with at least ``min_island_size`` nodes."""
    graph = pd.read_csv(connectivity_csv)
    required_columns = {"source", "target", "short_island_id"}
    missing_columns = required_columns.difference(graph.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns {sorted(missing_columns)} in {connectivity_csv}; "
            f"found {list(graph.columns)}"
        )

    reciprocal_graph = graph[graph["short_island_id"].notna()].copy()
    island_electrodes = pd.concat([
        reciprocal_graph[["short_island_id", "source"]].rename(
            columns={"source": "electrode"}
        ),
        reciprocal_graph[["short_island_id", "target"]].rename(
            columns={"target": "electrode"}
        ),
    ]).drop_duplicates()

    valid_islands = (
        island_electrodes.groupby("short_island_id")["electrode"]
        .nunique()
        .loc[lambda sizes: sizes >= min_island_size]
        .index
    )
    return (
        island_electrodes[
            island_electrodes["short_island_id"].isin(valid_islands)
        ]["electrode"]
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


def index_stimulation_configs(nas_dir, configs_basepath):
    """Index config files by electrode using exact StimUnit directories."""
    config_by_electrode = {}
    for stim_unit in range(32):
        config_pattern = os.path.join(
            nas_dir, configs_basepath, f"StimUnit{stim_unit:02d}", "*.cfg"
        )
        for config_fullfname in glob(config_pattern):
            match = re.search(r"El(\d+)_", os.path.basename(config_fullfname))
            if match:
                config_by_electrode.setdefault(int(match.group(1)), config_fullfname)
    return config_by_electrode

def stimulate_electrode(config_fullfname, full_recdir, s, dac_id, seq, stim_mode,
                        stim_settings, dac_sine_amplitude, stim_sequence_frequency_Hz,
                        stim_sequence_ncycles):
    config_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    array = get_maxlab_array()
    array.load_config(config_fullfname)

    fname = os.path.basename(config_fullfname).replace(".cfg", "")
    start_saving(s, dir_name=full_recdir, fname=fname, legacy=True)

    el = int(config_map.electrode.item())
    stim_unit = int(config_map.stim_unit.item())
    time.sleep(.1)
    attampt_connect_el2stim_unit(el, array, used_up_stim_units=[], with_download=True)

    seq_duration = stim_sequence_ncycles / stim_sequence_frequency_Hz
    print(f"\nStimulating El{el} with StimUnit{stim_unit} for {seq_duration:.1f}s")
    seq.send()
    time.sleep(seq_duration)
    time.sleep(.1)
    array.disconnect_electrode_from_stimulation(el)
    array.download()
    time.sleep(.3)

    stop_saving(s)
    array.close()

    pd.Series({
        "stim_unit": stim_unit,
        "stim_mode": stim_mode,
        "LSB_nA": stim_settings.loc[stim_unit, f"{stim_mode}_median_LSB_nA"].item(),
        "electrode": el,
        "dac_id": dac_id,
        "stim_sequence_DAC_amplitude": dac_sine_amplitude,
        "stim_sequence_frequency_Hz": stim_sequence_frequency_Hz,
        "stim_sequence_ncycles": stim_sequence_ncycles,
    }).to_frame().T.to_csv(os.path.join(full_recdir, f"{fname}.csv"), index=True)
    return fname

def compute_impedance(full_recdir, fname, stim_sequence_frequency_Hz, debug=False):
    h5_fname = fname + ".raw.h5"
    amplifier = int(h5_fname.split("Ampl")[-1].split("_")[0][:-7])
    rng = max(1, stim_sequence_frequency_Hz // 100)

    data = read_raw_data(full_recdir, h5_fname, convert2uV=True, subtract_dc_offset=False)
    dac = read_stim_DAC(full_recdir, h5_fname)
    ampl, phase_shift, _, _ = extract_amplitude(
        data[amplifier].astype(float), sampling_rate=20_000, highpass=100, notch=True,
        min_band=stim_sequence_frequency_Hz - rng, max_band=stim_sequence_frequency_Hz + rng,
        zoom_from=-150, zoom_to=150, dac=dac.astype(float), use_dac_interval=True,
        name=fname, plot_fname='debug_amplifier', debug=debug,
    )

    stim_info = pd.read_csv(os.path.join(full_recdir, f"{fname}.csv")).iloc[0]
    amplitude_uA = stim_info.loc["stim_sequence_DAC_amplitude"].item() * stim_info.loc["LSB_nA"].item() / 1000
    return pd.concat([stim_info, pd.Series({
        "amplitude_uV": ampl,
        "current_uA": amplitude_uA,
        "impedance_Ohm": ampl / amplitude_uA,
        "phase_shift_deg": phase_shift,
    })])

def main():
    L = Logger()
    L.init_logger(None, None, "WARNING")

    # ======== PARAMETERS ========
    nas_dir = device_paths()[0]
    headstage_name = "MEA1K24"
    configs_basepath = "mea1k_configs/single_el2stimunit_configs"
    stim_settings = get_headstage_stim_settings(headstage_name)

    connectivity_csv = (
        "/mnt/SpatialSequenceLearning/devices/headstage_devices/MEA1K24/"
        "recordings/2026-09-04_13.50.35_PDMS_J2_2x61umP,35umP_PT2_TileShorts/"
        "processed/extracted_connectivity_graph.csv"
    )

    gain = 7
    dac_id = 0
    debug = False
    stim_mode = 'large_current'
    stim_sequence_frequency_Hz = 1000
    stim_sequence_ncycles = 400
    dac_sine_amplitude = 3 if stim_mode == 'large_current' else 30
    # ======== PARAMETERS ========

    target_electrodes = get_short_island_electrodes(connectivity_csv)
    if not target_electrodes:
        raise RuntimeError(
            f"No islands with at least 3 electrodes found in {connectivity_csv}"
        )
    print(
        f"Found {len(target_electrodes)} electrodes in islands with at least "
        f"3 electrodes: {target_electrodes}"
    )

    t = datetime.now().strftime("%Y-%m-%d_%H.%M")
    rec_dir = f"{t}_PDMS_J2_2x61umP,35umP_PT2_Imp"
    full_recdir = os.path.join(nas_dir, f"devices/headstage_devices/{headstage_name}/recordings", rec_dir)
    os.makedirs(full_recdir, exist_ok=True)

    reset_MEA1K(gain=gain, enable_stimulation_power=True)
    s = get_maxlab_saving()

    results = []
    skipped_electrodes = []
    electrodes_by_stim_unit = {}

    # Scan each exact StimUnit directory once, then resolve locally and make
    # StimUnit the top-level iteration, as in record_imp.py.
    config_by_electrode = index_stimulation_configs(nas_dir, configs_basepath)
    for electrode in target_electrodes:
        config_fullfname = config_by_electrode.get(electrode)
        if config_fullfname is None:
            print(f"Skipping electrode {electrode}: No stim config found")
            skipped_electrodes.append(electrode)
            continue
        stim_unit = int(os.path.basename(os.path.dirname(config_fullfname)).replace("StimUnit", ""))
        electrodes_by_stim_unit.setdefault(stim_unit, []).append(
            (electrode, config_fullfname)
        )

    for stim_unit, electrode_configs in sorted(electrodes_by_stim_unit.items()):
        dac_code = int(stim_settings.loc[stim_unit, f"{stim_mode}_zero_current_DAC"])

        shift_DAC(dac_code)
        seq = create_stim_sine_sequence(dac_id=dac_id, amplitude=dac_sine_amplitude,
                                        f=stim_sequence_frequency_Hz,
                                        ncycles=stim_sequence_ncycles,
                                        center_around=dac_code)
        turn_on_stimulation_units([stim_unit], dac_id=dac_id, mode=stim_mode)

        try:
            for electrode, config_fullfname in electrode_configs:
                print(f"StimUnit{stim_unit}: stimulating electrode {electrode}")
                fname = stimulate_electrode(
                    config_fullfname, full_recdir, s, dac_id, seq, stim_mode,
                    stim_settings, dac_sine_amplitude,
                    stim_sequence_frequency_Hz, stim_sequence_ncycles,
                )

                res = compute_impedance(
                    full_recdir, fname, stim_sequence_frequency_Hz, debug=debug
                )
                print(res)
                results.append(res)
        finally:
            turn_off_stimulation_units([stim_unit])

    if skipped_electrodes:
        print(f"\nSkipped {len(skipped_electrodes)} electrode(s) with no stim config: "
              f"{skipped_electrodes}")

    if not results:
        print("No electrodes were successfully measured, skipping post-processing.")
        return

    aggr_df = pd.DataFrame(results)
    processed_dir = os.path.join(full_recdir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    aggr_df.to_csv(os.path.join(processed_dir, "all_impedance.csv"), index=False)
    print(f"\nSaved results to {os.path.join(processed_dir, 'all_impedance.csv')}")

    # no connectivity/pad_metal data available here (no implant mapping used) - dummy values
    # so plot_impedance_analysis can still render
    data_plotting = aggr_df.copy()
    data_plotting['connectivity'] = np.nan
    data_plotting['pad_metal'] = 1
    fig = vis.plot_impedance_analysis(processed_dir, data_plotting, skip_bottom=False,
                                      color_by_metal=False, use_imp_connectivity=True)
    fig.savefig(os.path.join(processed_dir, "all_impedance_analysis.png"), dpi=300)
    print(f"Saved plot to {os.path.join(processed_dir, 'all_impedance_analysis.png')}")

if __name__ == "__main__":
    main()


