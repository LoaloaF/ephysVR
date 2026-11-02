import os
import sys
from glob import glob
import time
import functools
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # debug plots are drawn in the postproc thread, files only

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

from mea1k_modules.mea1k_config_utils import create_stim_sine_sequence, get_zeroCurrentDAC_value, try_routing, turn_on_stimulation_units
from mea1k_modules.mea1k_config_utils import reset_MEA1K
from mea1k_modules.mea1k_config_utils import get_maxlab_saving
from mea1k_modules.mea1k_config_utils import get_maxlab_array
from mea1k_modules.mea1k_config_utils import init_fpga_sine_stim
from mea1k_modules.mea1k_config_utils import begin_fpga_sine_stim
from mea1k_modules.mea1k_config_utils import end_fpga_sine_stim
from mea1k_modules.mea1k_config_utils import start_saving
from mea1k_modules.mea1k_config_utils import stop_saving
from mea1k_modules.mea1k_config_utils import shift_DAC   

import maxlab
from mea1k_modules.mea1k_config_utils import (
    reset_MEA1K, try_routing, attampt_connect_el2stim_unit,
    turn_on_stimulation_units, turn_off_stimulation_units,
    start_saving, stop_saving, get_maxlab_saving, array_config2df,
)
from mea1k_modules.live_postproc import LivePostproc
from mea1k_connectivity_scripts.analyze_extSine_connectivity import (
    process_connectivity_recording, _save_output,
)

def rec_config_set():
    # ======== PARAMETERS ========
    nas_dir = device_paths()[0]
    headstage_name = "MEA1K27"
    subdir = f"devices/headstage_devices/{headstage_name}/recordings"
    # subdir = f"devices/implant_devices/260713_MEA1K23_J1688pad2shankB1/recordings"
    t = datetime.now().strftime("%Y-%m-%d_%H.%M")
    # subdir = "devices/well_devices/4983/recordings"
    # rec_dir = "4thBond4Shank_rec2_VrefFPGAStim_ampl15"
    # rec_dir = f"{t}_Bond3_PT4_B5_14Shank_VrefFPGAStim_ampl15"
    # rec_dir = f"{t}_8Sh4SilverPaint_Ext0.2uA_Singles_PT1_1.7mm_+MANYd_S1-8no7,8"
    # rec_dir = f"{t}_Bond3_16Shank_VrefFPGAStim_ampl15_14shanks_PT1_2.2mm"
    # rec_dir = f"{t}_Bond3_VrefFPGAStim_ampl15_14shanksSilver_PT1_2.2mm"
    # rec_dir = f"{t}_JDesignFirstSoldered2_VRef15_PT5"
    # rec_dir = f"{t}_ACF_PIEl_VRefAmpl15_Try10_PT5_2mm"
    # rec_dir = f"{t}_2ndTry_ACF_J2_VRef15_PT3"
    # rec_dir = f"{t}_noACFTry2_J2_36Pads_VRef15_PT1_1.9mm"
    # rec_dir = f"{t}_noACFTry4_J2_61Pads_VRef15_PT2"

    # ---- consistent naming theme: {t}_{name}_{tightening_str}_<recording type> ----
    # update name/tightening_str each session; recording type stays "10mVFPGA" for this script
    # name = "PDMS_J2_2x61umP,35umP"
    # name = "2x8x56umP_1of16100K"
    # name = "PDMS_J2_2x61umP,35umP"
    name = "test12_100k_36umP_56umPE"
    # name = "2x8x56umP_16`of16_100K"
    
    # tightening_str = "PT2_2.8mm"
    # tightening_str = "PT4_2.6mm"
    # tightening_str = "PT4_2.8mm"
    tightening_str = "PT2"
    
    rec_dir = f"{t}_{name}_{tightening_str}_10mVFPGA"
    
    post_download_wait_time = .6
    rec_time = 1.5
    gain = 7 # set on the chip, stored in each sidecar csv, used by postproc (never read from the file)
    with_external_sine = True
    external_sine_freq = 1000
    external_sine_amp_in_bits = 15 # 10 mV amplitude?
    input_ampl_mV = 10 # sine amplitude at the electrodes, connectivity = measured / input
    configs_basepath = os.path.join(nas_dir, "mea1k_configs", '')
    # configs_basepath = os.path.join(nas_dir, "devices", 'implant_devices', '260413_MEA1K22_S1688pad14shankB5')
    which_configs = "all_parallel"
    # which_configs = "bonding"
    # which_configs = "4x4_tile_meshgrid_seed42"

    # postproc, runs in a 2nd thread while recording
    postproc_n_samples = 8000
    connectivity_threshold = .7
    debug = False # postproc debug plots -> live_figures/debug_amplifier.png (first 10 els per config)
    reprocess = False # True: redo recordings that already have a processed/ output
    # True: no hardware access, only (re)process the recordings in postproc_rec_dir
    skip_stimulation = True
    postproc_rec_dir = "2026-11-01_20.17_test12_100k_36umP_56umPE_PT2_10mVFPGA"
    # ======== PARAMETERS ========
    
    if skip_stimulation:
        rec_dir = postproc_rec_dir
    path = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(path)} - ", path)

    # recordings from before sidecar csvs existed fall back to input_ampl_mV and gain
    process_fn = functools.partial(process_connectivity_recording,
                                   n_samples=postproc_n_samples,
                                   fallback_input_ampl_mV=input_ampl_mV, fallback_gain=gain,
                                   debug=debug)
    def summarize_fn(results, last_result):
        n_above = results.el[results.connectivity > connectivity_threshold].nunique()
        return f"{n_above} els > {connectivity_threshold}"
    
    if skip_stimulation:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Nothing to postprocess, no such recording dir: {path}")
        pp = LivePostproc(path, process_fn, summarize_fn, name="postproc_record_config_set",
                          reprocess=reprocess)
        pp.start()
        results = pp.finish()
    else:
        results = _record_configs(path, process_fn, summarize_fn, reprocess, configs_basepath,
                                  which_configs, gain, with_external_sine, external_sine_freq,
                                  external_sine_amp_in_bits, input_ampl_mV,
                                  post_download_wait_time, rec_time)
    if len(results):
        _save_output(path, results.set_index(['config', 'el']), "extr_connectivity.csv")

def _record_configs(path, process_fn, summarize_fn, reprocess, configs_basepath,
                    which_configs, gain, with_external_sine, external_sine_freq,
                    external_sine_amp_in_bits, input_ampl_mV, post_download_wait_time,
                    rec_time):
    reset_MEA1K(gain=gain, enable_stimulation_power=False)
    s = get_maxlab_saving()
    
    if with_external_sine:
        init_fpga_sine_stim(1/external_sine_freq, external_sine_amp_in_bits)

    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    print(f"Found {len(fnames)} configs in {os.path.join(configs_basepath, which_configs)}")
    os.makedirs(path, exist_ok=True)
    pp = LivePostproc(path, process_fn, summarize_fn, name="postproc_record_config_set",
                      reprocess=reprocess)
    pp.start()
    try:
        for i, config_fullfname in enumerate(sorted(fnames)):
            print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}")
            print(f"[postproc] {pp.status()}")
            
            array = get_maxlab_array()
            array.load_config(config_fullfname)
            print("Downloading presaved config...")
            array.download()
            if with_external_sine:
                begin_fpga_sine_stim()
            time.sleep(post_download_wait_time)
            # time.sleep(500)
            
            fname = os.path.basename(config_fullfname).replace(".cfg", "")
            start_saving(s, dir_name=path, fname=fname)
            time.sleep(rec_time)
            
            if with_external_sine:
                end_fpga_sine_stim()
            array.close()
            stop_saving(s)

            # sidecar csv, written after stop_saving: what postproc needs to know about
            # this recording. Then hand the recording to the postproc thread.
            pd.Series({
                "config_fullfname": config_fullfname,
                "gain": gain,
                "with_external_sine": with_external_sine,
                "external_sine_freq": external_sine_freq,
                "external_sine_amp_in_bits": external_sine_amp_in_bits,
                "input_ampl_mV": input_ampl_mV,
                "rec_time": rec_time,
            }).to_frame().T.to_csv(os.path.join(path, f"{fname}.csv"), index=False)
            pp.add(f"{fname}.raw.h5")
    finally:
        results = pp.finish()
    return results

def main():
    rec_config_set()
    

if __name__ == "__main__":
    main()