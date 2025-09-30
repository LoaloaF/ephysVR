import os
import sys
from glob import glob
import time
import datetime

import pandas as pd
import numpy as np

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

# import parent dir with general modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mea1k_modules.mea1k_config_utils import start_saving, stop_saving, try_routing
from mea1k_modules.mea1k_config_utils import attampt_connect_el2stim_unit, create_stim_sine_sequence
from mea1k_modules.mea1k_config_utils import reset_MEA1K, turn_on_stimulation_units, array_config2df, turn_off_stimulation_units
from mea1k_modules.mea1k_config_utils import get_maxlab_saving, get_maxlab_array

def process_config(config_fullfname, path, rec_time, post_download_wait_time, s, 
                   stim_seq, mode):
    array = get_maxlab_array()
    array.load_config(config_fullfname)
    
    config_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    stim_units = []
    for stim_el in config_map.electrode[config_map.stim].tolist():
        success, stim_units = attampt_connect_el2stim_unit(stim_el, array, 
                                                          used_up_stim_units=stim_units, 
                                                          with_download=False)
    print(f"Used {stim_units=}") # turn on
        
    array.download()
    fname = os.path.basename(config_fullfname).replace(".cfg", "")
    start_saving(s, dir_name=path, fname=fname, legacy=False)
    
    turn_on_stimulation_units(stim_units, mode=mode)
    time.sleep((post_download_wait_time/3) *2)
    
    # stimulation
    print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ on {stim_units} ")
    stim_seq.send()
    time.sleep(rec_time)
    
    # turn off
    turn_off_stimulation_units(stim_units)
    time.sleep(post_download_wait_time/3)
    array.close()
    stop_saving(s)
    # copy the config map to the recording dir
    config_map.to_csv(os.path.join(path, os.path.basename(config_fullfname).replace(".cfg", ".csv")))
    

def main():
    # ======== PARAMETERS ========
    # implant_name = "250205_MEA1K03_H1278pad4shankB5" # animal 10
    # implant_name = "241211_MEA1K06_H1278pad4shankB5" # animal 11
    subdir = f"devices/headstage_devices/MEA1K12/recordings"
    # nas_dir = "/home/houmanjava/nas_imitation"
    nas_dir = device_paths()[0]
    
    post_download_wait_time = .4
    rec_time = .4
    gain = 112
    amplitude = 10
    configs_basepath = os.path.join(nas_dir, "mea1k_configs")
    which_configs = "3x3_stim_seed42"
    # mode = "voltage"
    mode = "small_current"
    # ======== PARAMETERS ========
    
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    rec_dir = f"{t}_2ndBondBatch5SingelShankTight_{mode=}_{amplitude=}"
    full_recdir = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(full_recdir)} - ", full_recdir)
    
    stim_seq = create_stim_sine_sequence(dac_id=0, amplitude=amplitude, f=1000, ncycles=400, 
                                         nreps=1, voltage_conversion=mode=='voltage')

    s = get_maxlab_saving()
    reset_MEA1K(gain=gain, enable_stimulation_power=True)
    
    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    print(f"Found {len(fnames)} configs in {configs_basepath}/{which_configs}")
    for i, config_fullfname in enumerate(sorted(fnames)):
        print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}", flush=True)
        process_config(config_fullfname, full_recdir, rec_time, post_download_wait_time, 
                       s, stim_seq=stim_seq, mode=mode, )
        # if i>3:
        #     break
        
if __name__ == "__main__":
    main()