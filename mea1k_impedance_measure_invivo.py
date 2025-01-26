import os
import sys
from glob import glob
import time
import datetime
import maxlab

import pandas as pd
import numpy as np

import ephys_constants as C
from mea1k_utils import start_saving, stop_saving, try_routing
from mea1k_utils import attampt_connect_el2stim_unit, create_stim_sine_sequence
from mea1k_utils import reset_MEA1K, turn_on_stimulation_units, array_config2df

def stimulate(base_config_name, path, rec_time, post_download_wait_time, s, stim_seq):
    array = maxlab.chip.Array()
    config_fullfname = os.path.join(path, "..", "..", "bonding", base_config_name)
    print(f"Loading config: {config_fullfname}")
    array.load_config(config_fullfname)
    start_saving(s, dir_name=path, fname="recording")

    config_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    for pad in config_map.pads.unique():
        stim_el = config_map.electrode[config_map.pads == pad].iloc[0]
        print(f"Stimulating pad {pad} with electrode {stim_el}")
        attampt_connect_el2stim_unit(stim_el, array, with_download=True)
        array.download()
        time.sleep(post_download_wait_time)
    
        print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ ")
        stim_seq.send()
        time.sleep(rec_time)
        array.disconnect_electrode_from_stimulation(stim_el)
        
    # stimulation
    array.close()
    stop_saving(s)

def main():
    # ======== PARAMETERS ========
    implant_name = C.DEVICE_NAME_RAT006
    base_config_name = 'Stim_R06_12.Nov_829ElConfig_S46_241016_MEA1K03_H1278pad4shankB5.cfg'
    amplitude = 50
    stim_mode = 'small_current'
    rec_dir = datetime.datetime.now().strftime("%H:%M") + f"_current_imped_{amplitude}"
    post_download_wait_time = .4
    log2file = False
    rec_time = 3
    gain = 512
    # ======== PARAMETERS ========
    
    fullpath = os.path.join(C.NAS_DIR, "implant_devices", implant_name, 'recordings', rec_dir)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)
    print(fullpath)
    if log2file:
        log_fname = os.path.join(fullpath, "log.txt")
        logfile = open(log_fname, "w")
        sys.stdout = logfile
    print(f"Recording path exists: {os.path.exists(fullpath)} - ", fullpath)
    
    s = maxlab.Saving()
    stim_seq = create_stim_sine_sequence(dac_id=0, amplitude=amplitude, f=1000, 
                                         ncycles=1000, nreps=1)
    reset_MEA1K(gain=gain, enable_stimulation_power=True)
    turn_on_stimulation_units(list(range(32)), mode=stim_mode)
    
    stimulate(base_config_name, fullpath, rec_time, 
              post_download_wait_time, s, stim_seq)    
        
    if log2file:
        logfile.close()
        
if __name__ == "__main__":
    main()