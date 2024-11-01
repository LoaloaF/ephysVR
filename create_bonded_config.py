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

def main():
    random.seed(2)
    np.random.seed(2)

    PATH = '/mnt/SpatialSequenceLearning/Simon/impedance/device_headmount_new3EpoxyWalls/impedance_bonded4_D0_1KHz_1024_rec1/'
    log2file = False
    
    if log2file:
        logfile = open(f"{PATH}/mxstimpy.log", "w")
        sys.stdout = logfile

    pads = pd.read_pickle(f"{PATH}/final_pads.pkl")
    electrodes = pads.index.values
    
    reset_MEA1K()
    array = setup_array(electrodes, stim_electrodes=None)
    array.save_config(f"{PATH}/bonded.cfg")
    print(f"Saved config to {PATH}/bonded_new.cfg")
    
    if log2file:
        logfile.close()

if __name__ == "__main__":
    main()