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
    print("new config")
    array = maxlab.chip.Array()
    array.load_config(config_fullfname)
    print("new config loaded")
    
    config_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    # copy to output dir
    stim_units = []
    for stim_el in config_map.electrode[config_map.stim].tolist():
        success, stim_units = attampt_connect_el2stim_unit(stim_el, array, 
                                                          used_up_stim_units=stim_units, 
                                                          with_download=False)
    print(f"Used {stim_units=}") # turn on
        
        # adjust DAC sine wave with offset 
        # stim_seq = create_stim_sine_sequence(dac_id=0, amplitude=10, f=1000, ncycles=400, 
        #                                      nreps=1, adjust_offset_for_stimunit=stim_units[0])
        # time.sleep(.2)
        
    array.download()
    fname = os.path.basename(config_fullfname).replace(".cfg", "")
    start_saving(s, dir_name=path, fname=fname, legacy_mode=False)
    
    turn_on_stimulation_units(stim_units, mode=mode)
    time.sleep((post_download_wait_time/3) *2)
    
    print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ "
          f"on {stim_units} ")
    stim_seq.send()
    time.sleep(rec_time)
    # stimulation
    # turn off
    turn_off_stimulation_units(stim_units)
    # array.download()
    time.sleep(post_download_wait_time/3)
    # time.sleep(.5)
    array.close()
    stop_saving(s)

    # offset_map = {0: 0, 1: 0, 2: 427, 3: 0, 4: 492, 5: 0, 6: 347, 7: 703, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 895, 14: 0, 15: 0, 16: 959, 17: 634, 18: 0, 19: 0, 20: 325, 21: 1023, 22: 1023, 23: 255, 24: 0, 25: 1023, 26: 1023, 27: 630, 28: 1023, 29: 245, 30: 0, 31: 895}
    # config_map['stim_unit_offset'] = offset_map[stim_units[0]]
    print(config_map)
    config_map.to_csv(os.path.join(path, os.path.basename(config_fullfname).replace(".cfg", ".csv")))
    

def main():
    # ======== PARAMETERS ========
    # implant_name = "250205_MEA1K03_H1278pad4shankB5" # animal 10
    # implant_name = "241211_MEA1K06_H1278pad4shankB5" # animal 11
    # subdir = f"devices/well_devices/4983/recordings"
    subdir = f"devices/headstage_devices/MEA1K12/recordings"
    nas_dir = C.device_paths()[0]
    local_basedir = '/home/houmanjava/local_data'
    
    amplitude = 10
    mode = "voltage"
    # mode = "small_current"
    stimpulse = 'sine'
    # stimpulse = 'onoff'
    # stimpulse = 'bursting'
    # rec_time = .1
    # stimpulse = 'sine'
    
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    rec_dir = f"{t}_2ndBondBatch5SingelShankTight_{mode=}_{stimpulse=}_{amplitude=}"
    
    post_download_wait_time = .4
    log2file = False
    rec_time = .4
    gain = 112
    # configs_basepath = os.path.join(nas_dir, "devices", "implant_devices", implant_name, 'bonding', )
    configs_basepath = os.path.join(nas_dir, "mea1k_configs")
    which_configs = "3x3_stim_seed42"
    
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
    
    full_recdir = os.path.join(nas_dir, subdir, rec_dir)
    # full_recdir = os.path.join(local_basedir, subdir, rec_dir)
    if log2file:
        log_fname = os.path.join(full_recdir, "log.txt")
        logfile = open(log_fname, "w")
        sys.stdout = logfile
    
    print(f"Recording path exists: {os.path.exists(full_recdir)} - ", full_recdir)
    
    s = maxlab.Saving()
    reset_MEA1K(gain=gain, enable_stimulation_power=True)
    # turn_off_stimulation_units(list(range(32)))
    
    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    print(f"Found {len(fnames)} configs in {configs_basepath}/{which_configs}")
    for i, config_fullfname in enumerate(sorted(fnames)):
        print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}", flush=True)
        process_config(config_fullfname, full_recdir, rec_time, post_download_wait_time, 
                       s, stim_seq=stim_seq, mode=mode, )
        # if i>3:
        #     break
    if log2file:
        logfile.close()
        
if __name__ == "__main__":
    main()