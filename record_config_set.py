import os
import sys
from glob import glob
import time

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

from mea1k_modules.mea1k_config_utils import reset_MEA1K
from mea1k_modules.mea1k_config_utils import get_maxlab_saving
from mea1k_modules.mea1k_config_utils import get_maxlab_array
from mea1k_modules.mea1k_config_utils import init_fpga_sine_stim
from mea1k_modules.mea1k_config_utils import begin_fpga_sine_stim
from mea1k_modules.mea1k_config_utils import end_fpga_sine_stim
from mea1k_modules.mea1k_config_utils import start_saving
from mea1k_modules.mea1k_config_utils import stop_saving


def main():
    # ======== PARAMETERS ========
    nas_dir = device_paths()[0]
    subdir = "devices/headstage_devices/MEA1K12/recordings"
    # rec_dir = "4thBond4Shank_rec2_VrefFPGAStim_ampl15"
    rec_dir = "5thBond1Shank_rec3_VrefFPGAStim_ampl15-"
    post_download_wait_time = .6
    rec_time = .5
    gain = 7
    with_external_sine = True
    external_sine_freq = 1000
    external_sine_amp_in_bits = 15 # 10 mV amplitude?
    configs_basepath = os.path.join(nas_dir, "mea1k_configs", '')
    which_configs = "all_parallel"
    # which_configs = "4x4_tile_meshgrid_seed42"
    # ======== PARAMETERS ========
    
    path = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(path)} - ", path)
    reset_MEA1K(gain=gain)    
    s = get_maxlab_saving()
    
    if with_external_sine:
        init_fpga_sine_stim(1/external_sine_freq, external_sine_amp_in_bits)
    exit()
    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    for i, config_fullfname in enumerate(sorted(fnames)):
        print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}")
        
        array = get_maxlab_array()
        array.load_config(config_fullfname)
        print("Downloading presaved config...")
        array.download()
        if with_external_sine:
            begin_fpga_sine_stim()
        time.sleep(post_download_wait_time)        
        
        fname = os.path.basename(config_fullfname).replace(".cfg", "")
        start_saving(s, dir_name=path, fname=fname)
        time.sleep(rec_time)
        
        if with_external_sine:
            end_fpga_sine_stim()
        array.close()
        stop_saving(s)

if __name__ == "__main__":
    main()