import os
import sys
from glob import glob
import time
import datetime
import threading
import functools

import pandas as pd
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

# import parent dir with general modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mea1k_modules.mea1k_config_utils import try_routing
from mea1k_modules.mea1k_config_utils import get_zeroCurrent_centered_sine_sequences, start_saving, stop_saving, get_zeroCurrentDAC_value, shift_DAC
from mea1k_modules.mea1k_config_utils import attampt_connect_el2stim_unit, create_stim_sine_sequence
from mea1k_modules.mea1k_config_utils import reset_MEA1K, turn_on_stimulation_units, array_config2df, turn_off_stimulation_units
from mea1k_modules.mea1k_config_utils import get_maxlab_saving, get_maxlab_array
# from mea1k_modules.mea1k_config_utils import get_zeroCurrentDAC_value
from mea1k_utils import cp_rec_dir_to_implant_dir
from mea1k_connectivity_scripts.analyze_shortcut_stim import get_aggr_results

import ephys_constants as EC
from mea1k_modules.mea1k_raw_preproc import read_raw_data
from signal_helpers import estimate_frequency_power


CHIP_W, CHIP_H, PITCH = 220, 120, 17.5
FIG_DIR = os.path.join(os.path.dirname(__file__), "live_figures")

def get_hdf5_fnames_from_dir(subdir):
    fnames, ids = [], []
    for fname in sorted(os.listdir(subdir)):
        if fname.endswith('raw.h5'):
            fnames.append(fname)
            ids.append(fname.replace('.raw.h5', '').split('_')[-1])
    return fnames, ids

def extract_sine_voltages(subdir, fname, debug):
    """Shared primitive: one raw file -> df with sine_voltage_uV per electrode."""
    stimulated = pd.read_csv(os.path.join(subdir, fname.replace(".raw.h5", ".csv")))
    data = read_raw_data(subdir, fname, convert2uV=True, subtract_dc_offset=True,
                        #  col_slice=slice(14000, 22500, None))
    )
    stim_amplifier = stimulated[stimulated.stim].index[0]
    mean_ampl = np.array([estimate_frequency_power(row.astype(float)[14500:22000],
                          sampling_rate=EC.SAMPLING_RATE, min_band=960, 
                          max_band=1040, debug=True if debug and irow==stim_amplifier else False)[0]
                          for irow, row in enumerate(data)])
    if "Unnamed: 0" in stimulated.columns:
        stimulated.drop("Unnamed: 0", axis=1, inplace=True)
    stimulated['sine_voltage_uV'] = mean_ampl
    return stimulated

def _config_csv_is_ready(path, min_age_s=5):
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) >= min_age_s

def add_connectivity_single(stimulated):           # was analyze_single_stim
    stim_row = stimulated[stimulated.stim]
    if len(stim_row) == 0:
        return None
    stimulated['tile_connectivity'] = stimulated.sine_voltage_uV / stim_row.sine_voltage_uV.values[0]
    stimulated['tile'] = 0
    return stimulated

def add_connectivity_tiles(stimulated):            # was analyze_shorts
    stimulated = stimulated.sort_values(by=['tile', 'stim']).reset_index(drop=True)
    ratios = stimulated.groupby("tile").apply(
        lambda x: x.sine_voltage_uV / x[x.stim].sine_voltage_uV.item())
    if len(ratios.shape) == 2:
        ratios = ratios.iloc[0]
    stimulated['tile_connectivity'] = ratios.values
    return stimulated

def _el_xy(el):
    el = np.asarray(el)
    return (el % CHIP_W) * PITCH + PITCH/4, (el // CHIP_W) * PITCH + PITCH/4

def plot_config_debug(stimulated, fname, out_dir=FIG_DIR):
    """Live chip view: overwrites a single PNG so a viewer can auto-refresh."""
    fig = Figure(figsize=(24, 13))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # full chip canvas, faint
    bx, by = _el_xy(np.arange(CHIP_W * CHIP_H))
    ax.scatter(bx, by, s=10, marker='s', c='0.93', edgecolors='none')

    stim = stimulated[stimulated.stim]
    stim_ampl = stim.sine_voltage_uV.values[0] if len(stim) else None

    # config electrodes: fill = sine amplitude, vmax pinned to stim amplitude
    x, y = _el_xy(stimulated.electrode.values)
    sc = ax.scatter(x, y, c=stimulated.sine_voltage_uV, s=22, marker='s',
                    cmap='viridis', vmin=0, vmax=stim_ampl, edgecolors='none')

    # stimulated electrode(s): thin red circle around it
    if stim_ampl is not None:
        sx, sy = _el_xy(stim.electrode.values)
        ax.scatter(sx, sy, s=260, marker='o', facecolors='none',
                   edgecolors='yellow', linewidths=2)
        title = f"{fname}\nstim sine = {stim_ampl:.1f} uV"

        # flag electrodes brighter than the stim electrode
        over = stimulated[~stimulated.stim & (stimulated.sine_voltage_uV > stim_ampl)]
        if len(over):
            max_excess = over.sine_voltage_uV.max() - stim_ampl
            ax.text(0.5, 0.97,
                    f"⚠ {len(over)} electrode(s) exceed stim amplitude "
                    f"(max +{max_excess:.1f} uV, {over.sine_voltage_uV.max()/stim_ampl:.1f}x)",
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=20, color='red', weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))
    else:
        title = f"{fname}\nno stim electrode"

    fig.colorbar(sc, ax=ax, label='sine_voltage_uV', pad=0.02)
    ax.set_xlim(0, CHIP_W * PITCH)
    ax.set_ylim(CHIP_H * PITCH, 0)
    ax.set_aspect('equal', adjustable='box')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title(title, fontsize=20)

    # write to temp then replace, so a refreshing viewer never reads a half-written file
    out = "./live_figures/shortcut_stim_config_res.png"
    fig.savefig(out, dpi=130, bbox_inches='tight')
    os.replace(out, out)

def postprocess_file(subdir, fname, connectivity_fn, debug=False):
    in_csv  = os.path.join(subdir, fname.replace(".raw.h5", ".csv"))
    out_csv = os.path.join(subdir, "processed", fname.replace(".raw.h5", ".csv"))
    if os.path.exists(out_csv) or not _config_csv_is_ready(in_csv):
        return  # already done, OR recording not finished yet (no config csv)
    stimulated = connectivity_fn(extract_sine_voltages(subdir, fname, debug=debug))
    if stimulated is None:
        return
    if debug:
        plot_config_debug(stimulated, fname.replace(".raw.h5", ""))
    stimulated.index = pd.MultiIndex.from_product(
        [[fname.replace(".raw.h5", "")], stimulated.index], names=['config', 'el'])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    stimulated.to_csv(out_csv)
    Logger().logger.info(f"postproc -> {os.path.basename(out_csv)}")

def postprocess_live(subdir, connectivity_fn, stop_event, debug=False, poll_interval=2):
    """Run in a thread. Polls for finished recordings; one final sweep after stop."""
    while True:
        stopped = stop_event.is_set()
        if os.path.exists(subdir):
            for fname, _ in zip(*get_hdf5_fnames_from_dir(subdir)):
                postprocess_file(subdir, fname, connectivity_fn, debug=debug)
                print(f"Finished: {fname}", end="\r", flush=True)
        if stopped:
            break
        time.sleep(poll_interval)

def _fatal_thread_exit(message, exc=None):
    logger = Logger().logger
    if exc is None:
        logger.critical(message)
    else:
        logger.critical(message, exc_info=exc)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)

def _run_postprocess_live_guarded(subdir, connectivity_fn, stop_event, debug=False, poll_interval=2):
    """Run postprocess_live, but hard-exit the process if it fails unexpectedly."""
    try:
        postprocess_live(subdir, connectivity_fn, stop_event, debug=debug, poll_interval=poll_interval)
    except BaseException as exc:
        _fatal_thread_exit("postprocess_live thread crashed", exc)
    finally:
        if not stop_event.is_set():
            _fatal_thread_exit("postprocess_live thread exited unexpectedly")

def _stimulate_and_record(array, config_map, stim_units, path, fname, rec_time,
                          post_download_wait_time, s, stim_seq, mode):
    """Shared primitive: given a fully routed+downloaded array with stim_units already
    connected, record while stimulating, then tear down and write the config csv."""
    start_saving(s, dir_name=path, fname=fname, legacy=False)

    time.sleep((post_download_wait_time/3) *1)
    if len(stim_units) == 2:
        turn_on_stimulation_units(stim_units[0:1], dac_id=0, mode=mode)
        turn_on_stimulation_units(stim_units[1:2], dac_id=1, mode='large_current')
        shift_DAC(1, 502)
        # 6	502	51.72211128	452	6.427219226
    else:
        turn_on_stimulation_units(stim_units, mode=mode)

    time.sleep((post_download_wait_time/3) *1)

    # stimulation
    print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ on {stim_units} ")
    if len(stim_units) == 1 and mode != "voltage":
        stim_seq[stim_units[0]].send() # single unit, current mode, centered around zeroCurrentDAC_value
    elif len(stim_units) == 2 and mode != "voltage":
        stim_seq[stim_units[0]].send() # current mode, centered around zeroCurrentDAC_value
    else:
        stim_seq.send() # voltage mode, on many units ceneterd arround 512
    time.sleep(rec_time)

    # turn off
    turn_off_stimulation_units(stim_units)
    time.sleep((post_download_wait_time/6) *2)
    # disconnect electrode
    for stim_el in config_map.electrode[config_map.stim].tolist():
        array.disconnect_electrode_from_stimulation(stim_el)
    array.download()
    time.sleep((post_download_wait_time/6) *2)
    array.close()
    stop_saving(s)
    # write the config map AFTER stop_saving -> this is the "recording finished" signal
    config_map.to_csv(os.path.join(path, f"{fname}.csv"))

def run_stim_config(config_fullfname, path, rec_time, post_download_wait_time, s,
                    stim_seq, mode):
    array = get_maxlab_array()
    array.load_config(config_fullfname)

    config_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    config_map['stim_unit'] = None
    stim_units = []
    for stim_el in config_map[config_map.stim].electrode.tolist():
        print(stim_el)
        success, stim_units = attampt_connect_el2stim_unit(stim_el, array,
                                                          used_up_stim_units=stim_units,
                                                          with_download=False)
        config_map.loc[config_map.electrode == stim_el, 'stim_unit'] = stim_units[-1]

    print(f"Used {stim_units=}") # turn on
    array.download() # needed before saving otherwise order is off
    fname = os.path.basename(config_fullfname).replace(".cfg", "")
    _stimulate_and_record(array, config_map, stim_units, path, fname, rec_time,
                          post_download_wait_time, s, stim_seq, mode)

def run_stim_onfly(stim_el, path, rec_time, post_download_wait_time, s, stim_seq,
                   mode, base_els):
    """Like run_stim_config, but routes+stimulates a single electrode from an in-memory
    candidate list instead of loading a pre-saved .cfg/.csv pair from disk."""
    succ_routed, failed_routing, array = try_routing(base_els, stim_electrodes=[stim_el],
                                                      return_array=True)
    if len(failed_routing) > 0:
        print(f"Failed to route stim for el {stim_el}, skipping.")
        array.close()
        return

    config_map = array_config2df(array)
    config_map['stim'] = config_map.electrode == stim_el
    stim_unit = int(array.query_stimulation_at_electrode(stim_el))
    config_map['stim_unit'] = None
    config_map.loc[config_map.stim, 'stim_unit'] = stim_unit
    print(f"Used stim_units=[{stim_unit}]")

    fname = f"el_config_El{stim_el:05d}"
    _stimulate_and_record(array, config_map, [stim_unit], path, fname, rec_time,
                          post_download_wait_time, s, stim_seq, mode)

def _run_recording_loop(items, run_one_fn, full_recdir, connectivity_fn, s, stim_seq,
                        rec_time, post_download_wait_time, mode, debug=False,
                        skip_stimulation=False):
    """Start the postproc thread, run all items via run_one_fn, ensure clean shutdown."""
    processed_dir = os.path.join(full_recdir, "processed")
    stop_event = threading.Event()
    pp = threading.Thread(target=_run_postprocess_live_guarded,
                          args=(full_recdir, connectivity_fn, stop_event),
                          kwargs=dict(debug=debug),
                          daemon=True)
    pp.start()
    if skip_stimulation:
        print("Skipping stimulation, only postprocessing existing recordings.")
        # wait for postproc to finish existing recordings, then exit
        while True:
            if not pp.is_alive():
                break
            time.sleep(1)
        return
    items = sorted(items)
    try:
        for i, item in enumerate(items):
            print(f"\nConfig {i+1}/{len(items)}: {item}", flush=True)
            run_one_fn(item, full_recdir, rec_time, post_download_wait_time, s,
                      stim_seq=stim_seq, mode=mode)
    finally:
        stop_event.set()   # triggers final sweep
        pp.join()
        get_aggr_results(processed_dir)

def run_tile_square_stim(subdir, which_configs, rec_dir, skip_stimulation=False, 
                         debug=False, post_download_wait_time=1, rec_time=1.2, gain=7,
                         amplitude=10, mode="voltage"):
    nas_dir = device_paths()[0]
    configs_basepath = os.path.join(nas_dir, "mea1k_configs")

    full_recdir = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(full_recdir)} - ", full_recdir)

    stim_seq = create_stim_sine_sequence(dac_id=0, amplitude=amplitude, f=1000, ncycles=400,
                                        #  nreps=1, voltage_conversion=mode=='voltage')
                                         nreps=1, voltage_conversion=True)

    s = get_maxlab_saving()
    reset_MEA1K(gain=gain, enable_stimulation_power=True)

    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    print(f"Found {len(fnames)} configs in {configs_basepath}/{which_configs}")
    if len(fnames) == 0:
        raise ValueError(f"No config files found in {configs_basepath}/{which_configs}")
    _run_recording_loop(fnames, run_stim_config, full_recdir, add_connectivity_tiles, s, stim_seq,
                        rec_time, post_download_wait_time, mode,
                        skip_stimulation=skip_stimulation, debug=debug)

def run_single_stim(subdir, implant_name, config_dirname, rec_dir, skip_stimulation=False, 
                    debug=False, post_download_wait_time=1, rec_time=1.2, gain=7,
                    amplitude=10, mode="voltage", ):
    nas_dir = device_paths()[0]
    configs_basepath = os.path.join(nas_dir, "devices", "implant_devices",
                                    implant_name, 'bonding', config_dirname)

    full_recdir = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(full_recdir)} - ", full_recdir)

    # create stim_sequence bank (centered around zeroCurrentDAC_value) to avoid DC offset issues
    if mode != 'voltage':
        headstage_name = implant_name[implant_name.find("MEA1K"):implant_name.find("MEA1K")+7]
        stim_seq = get_zeroCurrent_centered_sine_sequences(headstage_name, mode, # stim_units=[5,6,7],
                                                                dac_id=0, amplitude=amplitude, 
                                                                f=1000, ncycles=400)
    else:
        stim_seq = create_stim_sine_sequence(dac_id=0, amplitude=amplitude, f=1000, ncycles=400,
                                            nreps=1, voltage_conversion=mode=='voltage')
    
    s = get_maxlab_saving()
    reset_MEA1K(gain=gain, enable_stimulation_power=True)

    fnames = glob(os.path.join(configs_basepath, "*.cfg"))
    if len(fnames) == 0:
        raise ValueError(f"No config files found in {configs_basepath}")
    print(f"Found {len(fnames)} configs in {configs_basepath}")
    _run_recording_loop(fnames, run_stim_config, full_recdir, add_connectivity_single, s, stim_seq,
                        rec_time, post_download_wait_time, mode,
                        skip_stimulation=skip_stimulation, debug=debug)

def main():
    # ======== WHICH RUN ========
    # which = "single"          # "single", "tile"
    which = "tile"          # "single", "tile"
    
    # ======== COMMON PARAMETERS ========
    nas_dir = device_paths()[0]
    debug = True              # live chip plot -> ./live_figures/shortcut_stim_config_res.png
    post_download_wait_time = 1
    rec_time = 1.2
    gain = 7
    amplitude = 10
    # mode = "large_current"          # or "small_current"
    mode = "voltage"          # or "small_current"
    skip_stimulation = False  # if True, only postprocess existing recordings (no stimulation)
    common = dict(debug=debug, post_download_wait_time=post_download_wait_time,
                  rec_time=rec_time, gain=gain, amplitude=amplitude, mode=mode)

    t = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    # implant_name = "260713_MEA1K23_J1688pad2shankB1"
    # implant_name = "260715_MEA1K24_S1688pad14shankB7"
    subdir = "devices/headstage_devices/MEA1K22/recordings"
    subdir = "devices/well_devices/4988/recordings"
    # subdir = f"devices/implant_devices/{implant_name}/recordings"
    

    # ======== SINGLE-STIM CHIP / IMPLANT / DIRS ========
    # single_implant_name = "260611_MEA1K24_S1688pad14shankB5"
    # single_config_dir   = "260625_MEA1K23_S1688pad14shankB6_26.Jun_658El_SingleStimConfigs/"
    # single_config_dir   = "260703_MEA1K23_S844pad8shankB1_03.Jul_822El_SingleStimConfigs/"
    # single_config_dir   = "OnTheFly_StimConfigs" # for 2 stim units 2 pad tests
    # single_config_dir   = "shorted_islands_configs/ShortedIslands_07.Jul_325El_SingleStimConfigs/"
    # single_config_dir   = "260713_MEA1K23_J1688pad2shankB1_13.Jul_132El_SingleStimConfigs"
    # single_config_dir   = "260713_MEA1K23_J1688pad2shankB1_13.Jul_257El_SingleStimConfigs"
    # single_config_dir   = "260715_MEA1K24_S1688pad14shankB7_15.Jul_579El_SingleStimConfigs"
    # single_config_dir   = "260715_MEA1K24_S1688pad14shankB7_15.Jul_618El_SingleStimConfigs"
    # single_rec_dir      = f"{t}_ACF_Try4_shorted_islands_config_{single_implant_name}"
    # single_rec_dir      = f"{t}_SingleShorts_JFirstSoldered_PT4"
    # single_rec_dir      = f"{t}_ReEtched14ShankReGoldPlated_SingleShorts_PT3"
    # single_rec_dir      = "2026-07-09_09.48.23_SC_JFirstTrySoldered_PT4"
    # single_rec_dir      = f"{t}_SC_postCurCheckNoExtCurStim_{single_implant_name}_negCtrl"
    # override
    # alt implants:
    # "260413_MEA1K22_S1688pad14shankB5" / "...21.Apr_783El_SingleStimConfigs"
    # "260501_MEA1K24_S1688pad14shankB5" / "...12.May_531El_SingleStimConfigs"

    # ======== TILE-STIM CHIP / DIRS ========
    tile_which_configs = "3x3_stim_seed42"
    # tile_which_configs = "3x3_islandsubset_stim_seed42"
    # tile_which_configs = "3x3_islandsubset2_stim_seed42"
    tile_rec_dir       = f"{t}_ReEtched14ShankReGoldPlated_TileShortsFull_PT1"
    # tile_rec_dir       = f"2026-07-15_07.58.07_JamesSoldered2_TileShortsFull_PT3"
    # tile_rec_dir       = f"{t}_SC_JFirstTrySoldered_PT4"
    # tile_rec_dir       = f"{t}_ACF_Try6_PT2"
    tile_rec_dir       = f"{t}_SC_Pt_MEA_els_PT0"
    # tile_rec_dir      = f"{t}_Unkown_SC_TileShorts"
    
    
    # tile_rec_dir       = f"{t}_FullShortTiles_JFirstTry_PT3"
    # tile_rec_dir       = "2026-07-06_20.43.57_ShinTsuFlim_Try4_PT1_EcoFLexNewFilmSpringInteronnect"
    # ===================================================

    if which == "single":
        run_single_stim(subdir, implant_name, single_config_dir,
                        single_rec_dir, skip_stimulation, **common)
        # cp_rec_dir_to_implant_dir(os.path.join(nas_dir, single_subdir, single_rec_dir), implant_name)
    elif which == "tile":
        run_tile_square_stim(subdir, tile_which_configs, tile_rec_dir,
                             skip_stimulation, **common)
        # cp_rec_dir_to_implant_dir(os.path.join(nas_dir, tile_subdir, tile_rec_dir), implant_name)

if __name__ == "__main__":
    main()