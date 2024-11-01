import os
from glob import glob
import time

import maxlab

import ephys_constants as C
from mea1k_utils import start_saving, stop_saving
def main():

    # ======== PARAMETERS ========
    subdir = "implant_devices/241101_headstage09_50pad1shank/"
    rec_dir = "ext25mVSine_1KHz_1024_rec1/"
    rec_dir = "meshconfigs_rec1/"
    post_download_wait_time = .6
    rec_time = 2
    # which_configs = "all_parallel"
    which_configs = "9x3x16_meshgrid"
    # ======== PARAMETERS ========
    
    path = C.NAS_DIR + subdir + rec_dir
    print(f"Recording path exists: {os.path.exists(path)} - ", path)
    configs_path = C.CODE_DIR + f"/assets/mea1k_configs/{which_configs}"
    
    s = maxlab.Saving()

    for i, config_fullfname in enumerate(glob(configs_path + "/*.cfg")):
        print(f"\nConfig {i+1}/{len(glob(configs_path + '/*.cfg'))}")
        
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