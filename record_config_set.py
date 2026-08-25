import os
import sys
from glob import glob
import time
import pandas as pd
from datetime import datetime

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

def rec_config_set():
    # ======== PARAMETERS ========
    nas_dir = device_paths()[0]
    headstage_name = "MEA1K23"
    subdir = f"devices/headstage_devices/{headstage_name}/recordings"
    # subdir = f"devices/implant_devices/260713_MEA1K23_J1688pad2shankB1/recordings"
    t = datetime.now().strftime("%Y-%m-%d_%H.%M")
    # subdir = "devices/well_devices/4983/recordings"
    # rec_dir = "4thBond4Shank_rec2_VrefFPGAStim_ampl15"
    # rec_dir = f"{t}_Bond3_PT4_B5_14Shank_VrefFPGAStim_ampl15"
    # rec_dir = f"{t}_8Sh4SilverPaint_Ext0.2uA_Singles_PT1_1.7mm_+MANYd_S1-8no7,8"
    # rec_dir = f"{t}_Bond3_16Shank_VrefFPGAStim_ampl15_14shanks_PT1_2.2mm"
    # rec_dir = f"{t}_Bond3_VrefFPGAStim_ampl15_14shanksSilver_PT1_2.2mm"
    rec_dir = f"{t}_JDesignFirstSoldered2_VRef15_PT5"
    # rec_dir = f"{t}_ACF_PIEl_VRefAmpl15_Try10_PT5_2mm"
    # rec_dir = f"{t}_ReEtched14ShankReGoldPlated_VRef15_PT3"
    
    post_download_wait_time = .6
    rec_time = 1.5
    gain = 7
    with_external_sine = True
    external_sine_freq = 1000
    external_sine_amp_in_bits = 15 # 10 mV amplitude?
    configs_basepath = os.path.join(nas_dir, "mea1k_configs", '')
    # configs_basepath = os.path.join(nas_dir, "devices", 'implant_devices', '260413_MEA1K22_S1688pad14shankB5')
    which_configs = "all_parallel"
    # which_configs = "bonding"
    # which_configs = "4x4_tile_meshgrid_seed42"
    # ======== PARAMETERS ========
    
    path = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(path)} - ", path)
    reset_MEA1K(gain=gain, enable_stimulation_power=False)
    s = get_maxlab_saving()
    
    if with_external_sine:
        init_fpga_sine_stim(1/external_sine_freq, external_sine_amp_in_bits)

    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    print(f"Found {len(fnames)} configs in {os.path.join(configs_basepath, which_configs)}")
    for i, config_fullfname in enumerate(sorted(fnames)):
        print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}")
        
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

def main():
    rec_config_set()
    

if __name__ == "__main__":
    main()