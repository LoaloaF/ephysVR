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

def process_config(config_fullfname, path, rec_time, post_download_wait_time, s, stim_seq):
    array = maxlab.chip.Array()
    array.load_config(config_fullfname)
    
    config_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    for stim_el in config_map.electrode[config_map.stim].tolist():
        attampt_connect_el2stim_unit(stim_el, array, with_download=False)
    array.download()
    time.sleep(post_download_wait_time)
    
    fname = os.path.basename(config_fullfname).replace(".cfg", "")
    print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ "
          f"{config_map.stim.sum()} electrodes ")
    start_saving(s, dir_name=path, fname=fname)
    stim_seq.send()
    time.sleep(rec_time)
    # stimulation
    array.close()
    stop_saving(s)

def main():
    # ======== PARAMETERS ========
    subdir = "headstage_devices/MEA1K06/recordings"
    rec_dir = "shortcut_stim_bonding3_singleshank_B6_241207"
    rec_dir = "bonding4_2+2shank_B6_241209_ripped"
    post_download_wait_time = .4
    log2file = False
    rec_time = .6
    gain = 112
    configs_basepath = os.path.join(C.NAS_DIR, "mea1k_configs", '')
    which_configs = "3x3_stim_seed42"
    # ======== PARAMETERS ========
    
    if log2file:
        log_fname = os.path.join(C.NAS_DIR, subdir, rec_dir, "log.txt")
        logfile = open(log_fname, "w")
        sys.stdout = logfile
    
    path = os.path.join(C.NAS_DIR, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(path)} - ", path)
    
    s = maxlab.Saving()
    stim_seq = create_stim_sine_sequence(dac_id=0, amplitude=5, f=1000, ncycles=1000, nreps=1)
    reset_MEA1K(gain=gain, enable_stimulation_power=True)
    turn_on_stimulation_units(list(range(32)), mode='voltage')
    
    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    for i, config_fullfname in enumerate(sorted(fnames)):
        print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}", flush=True)
        process_config(config_fullfname, path, rec_time, post_download_wait_time, s, stim_seq)
        
    if log2file:
        logfile.close()
        
if __name__ == "__main__":
    main()