import os
from glob import glob
import time

import maxlab

import ephys_constants as C
from mea1k_ephys import get_implant_mapping
from mea1k_utils import start_saving, stop_saving

from mea1k_utils import setup_array

def main():
    implant_mapping = get_implant_mapping(C.NAS_DIR, C.DEVICE_NAME)
    els = implant_mapping[implant_mapping.routed].mea1k_el
    post_download_wait_time = .6
    print("Making config for", len(els), "electrodes...")
    
    config_fullfname = os.path.join(C.NAS_DIR, C.DEVICE_NAME, f"bonded_{C.DEVICE_NAME}_201124_{len(els)}chnls.cfg")
    
    array = setup_array(els)
    array.download()
    time.sleep(post_download_wait_time)        
    array.save_config(config_fullfname)
    array.close()

if __name__ == "__main__":
    main()