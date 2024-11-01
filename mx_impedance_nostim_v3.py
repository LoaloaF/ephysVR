import sys
import os
# sys.path.insert(0, '/home/Alexei/nt_simon_nas/MaxLab/toolboxes/python/modules/')

import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util

import random
import time
import numpy as np
import pandas as pd

# from maxlab.saving import Saving

def reset_MEA1K():
    print("Resetting MEA1K...", end='', flush=True)
    maxlab.util.initialize()
    maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
    maxlab.send(maxlab.chip.Amplifier().set_gain(1))
    print("Done.")

def setup_array(electrodes, stim_electrodes=None, config_name="default_name"):
    print(f"Setting up array with {len(electrodes)} electrodes (reset,route&download)...", end='', flush=True)
    array = maxlab.chip.Array("offline")
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(electrodes[-1:], weight=1000)
    array.select_electrodes(electrodes[:-1])
    # array.connect_all_floating_amplifiers()
    # array.connect_amplifier_to_ringnode(0)

    if stim_electrodes is not None:
        array.select_stimulation_electrodes(stim_electrodes)
    array.route()
    array.download()
    print("Done.")
    return array

def turn_on_stimulation_units(stim_units):
    print(f"Setting up stim units {len(stim_units)}...", end="", flush=True)
    for stim_unit in stim_units:
        stim = maxlab.chip.StimulationUnit(str(stim_unit))
        stim.power_up(True)
        stim.connect(True)
        stim.set_current_mode()
        stim.set_small_current_range()
        # stim.set_large_current_range()
        stim.dac_source(0)
        maxlab.send(stim)
        time.sleep(.1)
    print("Done.")

def create_stim_sequence(dac=0, amplitude=25, npulses=10, nreps=3, inter_pulse_interval=100, rep_delay_s=.1):
    def append_stimulation_pulse(seq, amplitude):
        seq.append(maxlab.chip.DAC(0, 512-amplitude))
        seq.append(maxlab.system.DelaySamples(4))
        seq.append(maxlab.chip.DAC(0, 512+amplitude))
        seq.append(maxlab.system.DelaySamples(4))
        seq.append(maxlab.chip.DAC(0, 512))
        return seq

    seq = maxlab.Sequence()
    for i in range(nreps):
        for j in range(npulses):
            append_stimulation_pulse(seq, amplitude) # 25 *2.83mV - current mode?
            seq.append(maxlab.system.DelaySamples(inter_pulse_interval)) #5ms
        time.sleep(rep_delay_s)
    return seq

def connect_el2stim_units(array, stim_electrodes):
    # stim_els collects electrodes that are sucessfully connected    
    stim_els, stim_units = [], []
    # failed_stim_els collects electrodes where no stimulation units could be connected to
    failed_stim_els = []
    for el in stim_electrodes:
        array.connect_electrode_to_stimulation(el)
        stim_unit = array.query_stimulation_at_electrode(el)
        
        # unknown error case, could not find routing?
        if not stim_unit:
            print(f"Warning - Could not connect El{el} to a stim unit.")
            failed_stim_els.append(el)
        
        # stim unit not used yet, 
        elif int(stim_unit) not in stim_units:
            stim_units.append(int(stim_unit))
            stim_els.append(el)
            
            if len(stim_units) == 32:
                print("Used up all 32 stim units.")
                break
        
        # stim unit already assigned case        
        else:
            array.disconnect_electrode_from_stimulation(el)
    return stim_els, stim_units, failed_stim_els

def start_saving(s, dir_name, fname):
    s.set_legacy_format(True)
    s.open_directory(dir_name)
    s.start_file(fname)
    s.group_delete_all()
    s.group_define(0, "all_channels", list(range(1024)))
    print(f"Successfully opened file and defined group. Starting recording {dir_name}/{fname}")
    s.start_recording([0])

def stop_saving(s):
    print("Stopping recording...")
    s.stop_recording()
    s.stop_file()
    s.group_delete_all()

def main():
    random.seed(2)
    np.random.seed(2)


    PATH = '/mnt/SpatialSequenceLearning/Simon/impedance/device_headmount_new3EpoxyWalls/connectivity_bonded4_D8_brain_1024_rec2/'
    log2file = False
    
    if log2file:
        logfile = open(f"{PATH}/mxstimpy.log", "w")
        sys.stdout = logfile

    s = maxlab.Saving()

    config_fnames = os.listdir(f"./assets/mea1k_configs")
    # print(config_fnames)
    
    # exit()
        
    for i, config_fname in enumerate(config_fnames):
        # reset_MEA1K()
        
        mea1k_config_name = config_fname.replace(".csv", ".cfg")
        
        config_set_name = f"./assets/mea1k_configs/el_config_{i:03}.csv"
        # electrodes = pd.read_csv(config_set_name).values.flatten().astype(int)
        # print(config_set_name, len(electrodes))
        # array = setup_array(electrodes, stim_electrodes=None)
        array = maxlab.chip.Array("offline")
        array.load_config(config_set_name.replace(".csv", ".cfg"))
        # array.reset()
        # array.clear_selected_electrodes()
        # array.route()
        array.download()
        
        # config_set_name = f"./assets/mea1k_configs/el_config_{i:03}.csv"
        # electrodes = pd.read_csv(config_set_name).values.flatten().astype(int)
        # print(config_set_name, len(electrodes))
        # array = setup_array(electrodes, stim_electrodes=None)
        # # array.load_config(config_set_name)

        start_saving(s, PATH, mea1k_config_name.replace(".cfg", ""))
        array.save_config(f"./assets/mea1k_configs/{mea1k_config_name}")
        time.sleep(2)
        array.close()
        stop_saving(s)
        

    if log2file:
        logfile.close()

if __name__ == "__main__":
    main()