import os
from glob import glob
import time

import maxlab

import ephys_constants as C
from mea1k_utils import start_saving, stop_saving, reset_MEA1K
def main():

    # ======== PARAMETERS ========
    # subdir = "implant_devices/241101_headstage09_50pad1shank/"
    # subdir = "mea1k_well_devices//4983/"
    subdir = "implant_devices/241016_headstage03_46pad4shank/recordings"
    rec_dir = "complete_padlayout_rec1"
    # rec_dir = "meshconfigs_rec1/"
    post_download_wait_time = .6
    rec_time = 2
    # which_configs = "all_parallel"
    # configs_path = os.path.join(C.NAS_DIR, "mea1k_configs", '')
    configs_basepath = os.path.join(C.NAS_DIR, "implant_devices", '241016_headstage03_46pad4shank')
    which_configs = "full_padlayout_configs_real"
    # which_configs = "9x3x16_meshgrid"
    # ======== PARAMETERS ========
    
    path = os.path.join(C.NAS_DIR, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(path)} - ", path)
    
    s = maxlab.Saving()

    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    for i, config_fullfname in enumerate(fnames):
        print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}")
        
        array = maxlab.chip.Array()
        array.load_config(config_fullfname)
        print("Downloading presaved config...")
        array.download()
        time.sleep(post_download_wait_time)        
        
        fname = os.path.basename(config_fullfname).replace(".cfg", "")
        start_saving(s, dir_name=path, fname=fname)
        time.sleep(rec_time)
        
        array.close()
        stop_saving(s)

if __name__ == "__main__":
    main()