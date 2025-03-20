import os
import sys
from glob import glob
import time
import datetime
import maxlab

import pandas as pd
import numpy as np

import ephys_constants as C
from mea1k_modules.mea1k_config_utils import start_saving, stop_saving, try_routing
from mea1k_modules.mea1k_config_utils import attampt_connect_el2stim_unit, create_stim_sine_sequence
from mea1k_modules.mea1k_config_utils import reset_MEA1K, turn_on_stimulation_units, array_config2df, turn_off_stimulation_units

from mea1k_modules.mea1k_config_utils import create_stim_pulse_sequence
from mea1k_modules.mea1k_config_utils import create_stim_onoff_sequence

def process_config(config_fullfname, path, rec_time, post_download_wait_time, s, 
                   stim_seq, mode):
    array = maxlab.chip.Array()
    array.load_config(config_fullfname)
    
    config_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    # copy to output dir
    for stim_el in config_map.electrode[config_map.stim].tolist():
        success, stim_units = attampt_connect_el2stim_unit(stim_el, array, with_download=False)
        print(success) # turn on ?
        
    array.download()
    fname = os.path.basename(config_fullfname).replace(".cfg", "")
    start_saving(s, dir_name=path, fname=fname)
    time.sleep(post_download_wait_time/2)
    
    turn_on_stimulation_units(stim_units, mode=mode)
    time.sleep(post_download_wait_time/2)
    
    print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ "
          f"on {stim_units} ")
    stim_seq.send()
    time.sleep(rec_time)
    # stimulation
    # turn off
    turn_off_stimulation_units(stim_units)
    array.download()
    time.sleep(.1)
    array.close()
    stop_saving(s)
    config_map.to_csv(os.path.join(path, os.path.basename(config_fullfname).replace(".cfg", ".csv")))
    

def main():
    # ======== PARAMETERS ========
    # subdir = f"well_devices/4983/recordings"
    subdir = f"devices/implant_devices/250308_MEA1K07_H1628pad1shankB6/recordings"
    nas_dir = C.device_paths()[0]
    
    amplitude = 10
    # mode = "voltage"
    mode = "small_current"
    stimpulse = 'sine'
    # stimpulse = 'onoff'
    # stimpulse = 'bursting'
    # rec_time = .1
    # stimpulse = 'sine'
    
    t = datetime.datetime.now().strftime("%H.%M.%S")
    rec_dir = f"{t}_noSilk_tapwater_GND_REF_cable_imp8_localstim_{mode=}_{stimpulse=}2_{amplitude=}"
    
    post_download_wait_time = 1
    log2file = False
    rec_time = 1
    gain = 7
    configs_basepath = os.path.join(nas_dir, "devices", "implant_devices", "250308_MEA1K07_H1628pad1shankB6", 
                                    'bonding', )
    which_configs = "imp_rec_configs2"
    
    # stim
    if stimpulse == 'sine':
        stim_seq = create_stim_sine_sequence(dac_id=0, amplitude=amplitude, f=1000, ncycles=400, 
                                             nreps=1, voltage_conversion=mode=='voltage')
    elif stimpulse == 'bursting':
        stim_seq = create_stim_pulse_sequence(dac_id=0, amplitude=amplitude, 
                                              pulse_duration=167e-6, 
                                              inter_phase_interval=67e-6, 
                                              frequency=50, 
                                              burst_duration=400e-3, nreps=1,
                                              voltage_conversion=mode=='voltage')
    
    elif stimpulse == 'onoff':
        stim_seq = create_stim_onoff_sequence(dac_id=0, amplitude=amplitude,
                                               pulse_duration=2_000_000, 
                                               voltage_conversion=mode=='voltage')
    # ======== PARAMETERS ========
    
    if log2file:
        log_fname = os.path.join(nas_dir, subdir, rec_dir, "log.txt")
        logfile = open(log_fname, "w")
        sys.stdout = logfile
    
    path = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(path)} - ", path)
    
    s = maxlab.Saving()
    reset_MEA1K(gain=gain, enable_stimulation_power=True)
    # turn_off_stimulation_units(list(range(32)))
    
    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    print(f"Found {len(fnames)} configs in {configs_basepath}/{which_configs}")
    for i, config_fullfname in enumerate(sorted(fnames)):
        print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}", flush=True)
        process_config(config_fullfname, path, rec_time, post_download_wait_time, 
                       s, stim_seq, mode=mode)
        # if i>3:
        #     break
    if log2file:
        logfile.close()
        
if __name__ == "__main__":
    main()