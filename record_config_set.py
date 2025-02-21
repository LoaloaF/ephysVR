import os
from glob import glob
import time

from ephys_constants import device_paths
import mea1k_modules.mea1k_config_utils as mea1k

def main():
    # ======== PARAMETERS ========
    nas_dir = device_paths()[0]
    # subdir = "mea1k_well_devices//4983/"
    subdir = "headstage_devices/MEA1K06/recordings"
    rec_dir = "bonding5_4shank_B6_241211_ext5mV1Khz_silk_rec3"
    post_download_wait_time = 1.6
    rec_time = 2
    gain = 7
    configs_basepath = os.path.join(nas_dir, "mea1k_configs", '')
    which_configs = "all_parallel"
    # which_configs = "4x4_tile_meshgrid_seed42"
    # ======== PARAMETERS ========
    
    path = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(path)} - ", path)
    mea1k.reset_MEA1K(gain=gain)    
    s = mea1k.get_maxlab_saving()
    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    for i, config_fullfname in enumerate(sorted(fnames)):
        print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}")
        
        array = mea1k.get_maxlab_array()
        array.load_config(config_fullfname)
        print("Downloading presaved config...")
        array.download()
        time.sleep(post_download_wait_time)        
        
        fname = os.path.basename(config_fullfname).replace(".cfg", "")
        mea1k.start_saving(s, dir_name=path, fname=fname)
        time.sleep(rec_time)
        
        array.close()
        mea1k.stop_saving(s)

if __name__ == "__main__":
    main()