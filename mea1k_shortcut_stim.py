import os
from glob import glob
import time
import datetime
import maxlab

import pandas as pd

import ephys_constants as C
from mea1k_utils import start_saving, stop_saving

#tmp
from make_mea1k_config import get_all_9x3x16_meshgrid_electrodes
tile_el_mapping = get_all_9x3x16_meshgrid_electrodes()

def process_config(config_fullfname, path, rec_time, post_download_wait_time, s):
    # array = maxlab.chip.Array()
    # array.load_config(config_fullfname)
    # array.download()
    # time.sleep(post_download_wait_time)    
    
    config_details = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    print(config_details)
    tile_el_mapping
    exit()
    fname = os.path.basename(config_fullfname).replace(".cfg", "")
    
    # start_saving(s, dir_name=path, fname=fname)
    time.sleep(rec_time)
    array.close()
    # stop_saving(s)


def main():
    # ======== PARAMETERS ========
    subdir = "headstage_devices/MEA1K05/recordings"
    date = datetime.datetime.now().strftime("%Y%m%d")
    rec_dir = f"{date}_noGP_shortcut_stim_rec1"
    post_download_wait_time = .6
    rec_time = 1
    configs_basepath = os.path.join(C.NAS_DIR, "mea1k_configs")
    which_configs = "4x4_tile_meshgrid_seed42"
    which_configs = "9x3x16_meshgrid_seed42"
    # ======== PARAMETERS ========
    
    path = os.path.join(C.NAS_DIR, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(path)} - ", path)
    
    s = maxlab.Saving()

    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    for i, config_fullfname in enumerate(fnames):
        print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}")
        
        process_config(config_fullfname, path, rec_time, post_download_wait_time, s)


if __name__ == "__main__":
    main()