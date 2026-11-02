import os
import sys
from glob import glob
import time
import datetime
import threading
import functools

import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d

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
from mea1k_modules.mea1k_config_utils import attampt_connect_el2stim_unit, create_stim_sine_sequence, set_DAC
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

def find_sequential_windows(data, stimulated, f=1000, trim=0.2):
    """Sequential tile stim: locate the n_steps sine bursts in one recording.
    Burst timing comes from the data itself (software sleeps are not sample-exact):
    lock-in envelope at `f` on the stim electrodes' rows, summed, thresholded. The k-th
    burst belongs to the stim electrode with stim_order == k. Returns {tile: slice}
    (central 1-2*trim of each burst, away from onset/offset), or None if the number of
    bursts found != number of stim electrodes."""
    fs = EC.SAMPLING_RATE
    stim_rows = np.flatnonzero(stimulated.stim.values)
    order = stimulated.stim_order.values[stim_rows].astype(int)
    stim_rows, order = stim_rows[np.argsort(order)], np.sort(order)   # row of step k at [k]

    x = data[stim_rows].astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    t = np.arange(x.shape[1]) / fs
    size = 5 * int(round(fs / f))
    I = uniform_filter1d(x * np.cos(2*np.pi*f*t), size, axis=1)
    Q = uniform_filter1d(x * np.sin(2*np.pi*f*t), size, axis=1)
    env_rows = np.hypot(I, Q)
    env = env_rows.sum(axis=0)

    # plateau level of a burst; the transients (spikes, 10-50x the plateau) are too few
    # samples to move the median. Threshold at 40%: the baseline after a transient rings
    # down at ~2x noise, far below that, so it doesn't glue onto the next burst.
    plateau = np.median(env[env > 0.2 * np.percentile(env, 99)])
    mask = env > 0.4 * plateau
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    # keep flat runs only: a burst is a plateau (peak ~1.1x median), the startup / unit
    # power-up transients are spikes (peak >> median) whose length varies per file.
    # Judged on the central part only (the part later measured): a burst may carry a small
    # switching transient at its edge when its unit is parked again.
    lens = ends - starts
    centers = [env[a + int(trim * (b - a)): b - int(trim * (b - a))] for a, b in zip(starts, ends)]
    flat = np.array([len(c) > 0 and c.max() <= 2 * np.median(c) for c in centers])
    flat &= lens > 0.05 * fs
    if not flat.any():
        Logger().logger.error("No stim bursts found")
        return None
    # The gap between bursts is software-timed and varies (down to none), so bursts can
    # run into each other. Duration of one burst is fixed -> split each run into
    # round(len/T) equal bursts.
    T = np.percentile(lens[flat], 25)
    bursts = []
    for a, b in zip(starts[flat], ends[flat]):
        m = int(round((b - a) / T))
        if m < 1 or abs((b - a) - m * T) > 0.3 * T:
            continue
        cuts = np.linspace(a, b, m + 1).astype(int)
        bursts += list(zip(cuts[:-1], cuts[1:]))
    starts, ends = (np.array(v) for v in zip(*bursts)) if bursts else (np.array([]),) * 2
    if len(starts) != len(stim_rows):
        Logger().logger.error(f"Found {len(starts)} stim bursts, expected {len(stim_rows)}")
        return None

    # sanity: each stim electrode should respond most strongly in its own step
    own_mean = np.array([env_rows[:, a:b].mean(axis=1) for a, b in zip(starts, ends)]).T
    n_off = (own_mean.argmax(axis=1) != np.arange(len(stim_rows))).sum()
    if n_off:
        Logger().logger.warning(f"{n_off}/{len(stim_rows)} stim electrodes respond most "
                                f"strongly outside their own step (shorted, or order off?)")

    windows = {}
    for k, (a, b) in enumerate(zip(starts, ends)):
        pad = int(trim * (b - a))
        windows[stimulated.tile.values[stim_rows[k]]] = slice(a + pad, b - pad)
    return windows

def extract_sine_voltages(subdir, fname, debug):
    """Shared primitive: one raw file -> df with sine_voltage_uV per electrode.
    If the config csv has a stim_order column (sequential tile stim), every electrode is
    evaluated in the burst of its own tile; otherwise in one fixed window (all stim at once)."""
    stimulated = pd.read_csv(os.path.join(subdir, fname.replace(".raw.h5", ".csv")))
    data = read_raw_data(subdir, fname, convert2uV=True, subtract_dc_offset=True,
                        #  col_slice=slice(14000, 22500, None))
    )
    stim_amplifier = stimulated[stimulated.stim].index[0]
    if 'stim_order' in stimulated.columns and stimulated.stim_order.notna().any():
        windows = find_sequential_windows(data, stimulated)
        if windows is None:
            return None
        tiles = stimulated.tile.values
        mean_ampl = np.array([estimate_frequency_power(row.astype(float)[windows[tiles[irow]]],
                              sampling_rate=EC.SAMPLING_RATE, min_band=960,
                              max_band=1040, debug=True if debug and irow==stim_amplifier else False)[0]
                              for irow, row in enumerate(data)])
    else:
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

_postproc_failed = set()   # files whose processing returned None, don't re-read every poll

def postprocess_file(subdir, fname, connectivity_fn, debug=False):
    in_csv  = os.path.join(subdir, fname.replace(".raw.h5", ".csv"))
    out_csv = os.path.join(subdir, "processed", fname.replace(".raw.h5", ".csv"))
    if (os.path.exists(out_csv) or not _config_csv_is_ready(in_csv)
            or (subdir, fname) in _postproc_failed):
        return False  # already done, OR recording not finished yet (no config csv)
    stimulated = extract_sine_voltages(subdir, fname, debug=debug)
    stimulated = connectivity_fn(stimulated) if stimulated is not None else None
    if stimulated is None:
        _postproc_failed.add((subdir, fname))
        Logger().logger.error(f"postproc failed, skipping {fname}")
        return False
    if debug:
        plot_config_debug(stimulated, fname.replace(".raw.h5", ""))
    stimulated.index = pd.MultiIndex.from_product(
        [[fname.replace(".raw.h5", "")], stimulated.index], names=['config', 'el'])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    stimulated.to_csv(out_csv)
    Logger().logger.info(f"postproc -> {os.path.basename(out_csv)}")
    return True

def postprocess_live(subdir, connectivity_fn, stop_event, debug=False, poll_interval=2):
    """Run in a thread. Polls for finished recordings; one final sweep after stop."""
    n_done = 0
    while True:
        stopped = stop_event.is_set()
        if os.path.exists(subdir):
            fnames = get_hdf5_fnames_from_dir(subdir)[0]
            for fname in fnames:
                if postprocess_file(subdir, fname, connectivity_fn, debug=debug):
                    n_done += 1
                    print(f"[postproc] {n_done} processed this run "
                          f"({len(_postproc_failed)} failed), last: {fname}", end="\r", flush=True)
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

def _stimulate_sequential(stim_units, stim_seq, step_s, gap_s):
    """One burst per stim unit, in ascending unit order. All units are already powered up
    and parked on DAC 1 (idle at 512); a unit follows DAC 0 (the sine) only during its own
    step, so no wake-up time is needed between steps."""
    for k, unit in enumerate(sorted(stim_units)):
        print(f"\nStep {k+1}/{len(stim_units)}: stimulating stim unit {unit}", end="", flush=True)
        turn_on_stimulation_units([unit], dac_id=0, mode='voltage')
        stim_seq.send()
        time.sleep(step_s + gap_s)
        turn_on_stimulation_units([unit], dac_id=1, mode='voltage')
    print()

def _stimulate_and_record(array, config_map, stim_units, path, fname, rec_time,
                          post_download_wait_time, s, stim_seq, mode, seq_step=None):
    """Shared primitive: given a fully routed+downloaded array with stim_units already
    connected, record while stimulating, then tear down and write the config csv.
    seq_step=(step_s, gap_s) stimulates the units one after another (voltage mode only)
    instead of all together, and records the order in config_map.stim_order."""
    if seq_step is not None:
        if mode != 'voltage':
            raise ValueError("sequential stim is voltage mode only")
        config_map['stim_order'] = config_map.stim_unit.map(
            {u: i for i, u in enumerate(sorted(stim_units))})
    start_saving(s, dir_name=path, fname=fname, legacy=False)

    time.sleep((post_download_wait_time/3) *1)
    if seq_step is not None:
        turn_on_stimulation_units(stim_units, dac_id=1, mode=mode)
        set_DAC(1, 512)
    elif len(stim_units) == 2:
        turn_on_stimulation_units(stim_units[0:1], dac_id=0, mode=mode)
        turn_on_stimulation_units(stim_units[1:2], dac_id=1, mode='large_current')
        shift_DAC(1, 502)
        # 6	502	51.72211128	452	6.427219226
    else:
        turn_on_stimulation_units(stim_units, mode=mode)

    time.sleep((post_download_wait_time/3) *1)

    # stimulation
    print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ on {stim_units} ")
    if seq_step is not None:
        _stimulate_sequential(stim_units, stim_seq, *seq_step)
        time.sleep(post_download_wait_time/3)  # tail after the last burst
    elif len(stim_units) == 1 and mode != "voltage":
        stim_seq[stim_units[0]].send() # single unit, current mode, centered around zeroCurrentDAC_value
    elif len(stim_units) == 2 and mode != "voltage":
        stim_seq[stim_units[0]].send() # current mode, centered around zeroCurrentDAC_value
    else:
        stim_seq.send() # voltage mode, on many units ceneterd arround 512
    if seq_step is None:
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
                    stim_seq, mode, seq_step=None):
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
                          post_download_wait_time, s, stim_seq, mode, seq_step=seq_step)

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
        n_raw = len(get_hdf5_fnames_from_dir(full_recdir)[0])
        n_proc = len(glob(os.path.join(processed_dir, "el_config_*.csv")))
        print(f"Skipping stimulation, only postprocessing existing recordings: "
              f"{n_raw} raw, {n_proc} already processed.")
        stop_event.set()   # the thread's final sweep covers all existing recordings, then ends
        pp.join()
        print(f"\nPostprocessing done: {len(_postproc_failed)} failed, aggregating...")
        get_aggr_results(processed_dir)
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
                         amplitude=10, mode="voltage", sequential=False,
                         step_cycles=200, step_gap_s=0.1):
    """sequential=True: stim units fire one after another (ascending unit number), each a
    `step_cycles`-cycle 1 kHz burst + `step_gap_s` gap, all within ONE recording per config."""
    nas_dir = device_paths()[0]
    configs_basepath = os.path.join(nas_dir, "mea1k_configs")

    full_recdir = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(full_recdir)} - ", full_recdir)

    seq_step = (step_cycles/1000, step_gap_s) if sequential else None
    stim_seq = s = None
    if skip_stimulation:   # postproc only: no hardware access at all
        if not os.path.exists(full_recdir):
            raise FileNotFoundError(f"Nothing to postprocess, no such recording dir: {full_recdir}")
    else:
        stim_seq = create_stim_sine_sequence(dac_id=0, amplitude=amplitude, f=1000,
                                             ncycles=step_cycles if sequential else 400,
                                            #  nreps=1, voltage_conversion=mode=='voltage')
                                             nreps=1, voltage_conversion=True)
        s = get_maxlab_saving()
        reset_MEA1K(gain=gain, enable_stimulation_power=True)

    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    print(f"Found {len(fnames)} configs in {configs_basepath}/{which_configs}")
    if len(fnames) == 0:
        raise ValueError(f"No config files found in {configs_basepath}/{which_configs}")
    run_one = functools.partial(run_stim_config, seq_step=seq_step)
    _run_recording_loop(fnames, run_one, full_recdir, add_connectivity_tiles, s, stim_seq,
                        rec_time, post_download_wait_time, mode,
                        skip_stimulation=skip_stimulation, debug=debug)

def run_single_stim(subdir, configs_basepath, rec_dir, headstage_name=None,
                    skip_stimulation=False, 
                    debug=False, post_download_wait_time=1, rec_time=1.2, gain=7,
                    amplitude=10, mode="voltage", ):
    """`configs_basepath` is the full path to a dir of single-stim .cfg/.csv pairs
    (implant bonding dir, or an island config dir from make_island_single_stim_configs).
    `headstage_name` is only needed for the current stim modes."""
    nas_dir = device_paths()[0]

    full_recdir = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(full_recdir)} - ", full_recdir)

    stim_seq = s = None
    if skip_stimulation:   # postproc only: no hardware access at all
        if not os.path.exists(full_recdir):
            raise FileNotFoundError(f"Nothing to postprocess, no such recording dir: {full_recdir}")
    else:
        # create stim_sequence bank (centered around zeroCurrentDAC_value) to avoid DC offset issues
        if mode != 'voltage':
            if headstage_name is None:
                raise ValueError(f"headstage_name is required for {mode=}")
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
    debug = False              # live chip plot -> ./live_figures/shortcut_stim_config_res.png
    post_download_wait_time = 1
    rec_time = 1.2
    gain = 7
    amplitude = 10
    # mode = "large_current"          # or "small_current"
    mode = "voltage"          # or "small_current"
    skip_stimulation = False  # if True, only postprocess existing recordings (no stimulation, no MEA1K access)
    common = dict(debug=debug, post_download_wait_time=post_download_wait_time,
                  rec_time=rec_time, gain=gain, amplitude=amplitude, mode=mode)

    t = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    # implant_name = "260713_MEA1K23_J1688pad2shankB1"
    # implant_name = "260715_MEA1K24_S1688pad14shankB7"
    subdir = "devices/headstage_devices/MEA1K27/recordings"
    # subdir = "devices/well_devices/4988/recordings"
    # subdir = f"devices/implant_devices/{implant_name}/recordings"

    name = "5thTry_ACF_J2_36umP"
    name = "PDMS_J2_2x61umP,35umP"
    name = "2x8x56umP_16of16"
    name = "2x8x56umP"
    name = "test12_100k_36umP_56umPE"

    tightening_str = "PT2"







    
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
    single_rec_dir      = f"{t}_{name}_{tightening_str}_SingleShorts"
    # island configs (one el per shorted island, from a TileShorts graph) live in
    # mea1k_configs/shorted_islands_configs instead of an implant bonding dir
    single_configs_basepath = os.path.join(
        nas_dir, "mea1k_configs", "shorted_islands_configs",
        # "2026-09-08_02.16.27_2x8x56umP_PT1_TileShorts_08.Sep_1020Islands_SingleStimConfigs")
        "2026-09-30_16.22.55_2x8x56umP_PT4_TileShorts_30.Sep_542Islands_SingleStimConfigs")
    single_headstage_name = "MEA1K24"
    # implant bonding dir instead:
    # single_configs_basepath = os.path.join(nas_dir, "devices", "implant_devices",
    #                                        single_implant_name, 'bonding', single_config_dir)
    # override
    # alt implants:
    # "260413_MEA1K22_S1688pad14shankB5" / "...21.Apr_783El_SingleStimConfigs"
    # "260501_MEA1K24_S1688pad14shankB5" / "...12.May_531El_SingleStimConfigs"







    # ======== TILE-STIM CHIP / DIRS ========
    tile_which_configs = "3x3_stim_seed42"
    tile_sequential = True   # True: one stim unit at a time (by unit number), single file per config
    tile_step_cycles = 200    # sequential only: 1 kHz cycles per burst
    tile_step_gap_s = 0.1     # sequential only: pause between bursts (can jitter to none, postproc splits those)
    # tile_which_configs = "3x3_islandsubset_stim_seed42"
    # tile_which_configs = "3x3_islandsubset2_stim_seed42"
    # tile_rec_dir       = f"{t}_ReEtched14ShankReGoldPlated_TileShortsFull_PT1"
    # tile_rec_dir       = f"2026-07-15_07.58.07_JamesSoldered2_TileShortsFull_PT3"
    # tile_rec_dir       = f"{t}_SC_JFirstTrySoldered_PT4"
    # tile_rec_dir       = f"{t}_ACF_Try6_PT2"
    # tile_rec_dir       = f"{t}_SC_Pt_MEA_els_PT0"
    # tile_rec_dir      = f"{t}_SC_TileShorts_J2"
    # tile_rec_dir      = f"{t}_5thTry_ACF_J2_36Pads_TileShorts_PT4_2mm"
    tile_rec_dir      = f"{t}_{name}_{tightening_str}_TileShorts"

    
    # tile_rec_dir       = f"{t}_FullShortTiles_JFirstTry_PT3"
    # tile_rec_dir       = "2026-07-06_20.43.57_ShinTsuFlim_Try4_PT1_EcoFLexNewFilmSpringInteronnect"
    # ===================================================

    if which == "single":
        run_single_stim(subdir, single_configs_basepath, single_rec_dir,
                        headstage_name=single_headstage_name,
                        skip_stimulation=skip_stimulation, **common)
        # cp_rec_dir_to_implant_dir(os.path.join(nas_dir, single_subdir, single_rec_dir), implant_name)
    elif which == "tile":
        run_tile_square_stim(subdir, tile_which_configs, tile_rec_dir,
                             skip_stimulation, sequential=tile_sequential,
                             step_cycles=tile_step_cycles, step_gap_s=tile_step_gap_s,
                             **common)
        # cp_rec_dir_to_implant_dir(os.path.join(nas_dir, tile_subdir, tile_rec_dir), implant_name)

if __name__ == "__main__":
    main()