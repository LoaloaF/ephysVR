import os
import sys
from glob import glob
import time
import datetime
import maxlab

import pandas as pd
import numpy as np

import ephys_constants as C
from mea1k_utils import start_saving, stop_saving 
from mea1k_utils import attampt_connect_el2stim_unit, create_stim_sine_sequence
from mea1k_utils import reset_MEA1K, turn_on_stimulation_units

def select_stim_els(config_el_tile_map, stim_set_i, array):
    config_uniq_tiles = config_el_tile_map.unique()
    
    stim_set_electrodes = []
    used_up_stim_units = []
    stim_set_identifer = []
    for tile_i in config_uniq_tiles:
        print(f"{len(config_el_tile_map)}", end='')
        
        # condition never met? Why?
        tile_el_tile_map = config_el_tile_map[config_el_tile_map==tile_i]
        if tile_el_tile_map.size == 0:
            print(f"TILE {tile_i} no more els, ", end='\n\n')
            continue
        
        # try differnt els in a tile until one can be connected to 1 of 32 DACs
        tried_els = []
        tile_els = tile_el_tile_map.index.tolist()
        while True:
            tile_els = [el for el in tile_els if el not in tried_els]
            if len(tile_els) == 0:
                print("skip tile!!!", end='')
                break
            # sample a random electrode from the tile
            stim_el = tile_els[np.random.randint(0, len(tile_els))]
            
            success, used_up_stim_units = attampt_connect_el2stim_unit(stim_el, 
                                                                    array, 
                                                                    used_up_stim_units)
            if success:
                print("S...", end='')
                break
            print("F", end='')
            tried_els.append(stim_el)
        
        # drop the electrode from the mapping
        config_el_tile_map = config_el_tile_map.drop(stim_el)
                    
        stim_set_electrodes.append(stim_el)
        stim_set_identifer.append((stim_set_i, tile_i, stim_el))
    
    stim_set_identifer = pd.DataFrame(stim_set_identifer, 
                                      columns=['stim_set', 'tile', 'el'])
    print("\n", stim_set_identifer)
    print(f"{stim_set_identifer.shape[0]}/{len(config_uniq_tiles)}", flush=True)
    return stim_set_electrodes, stim_set_identifer, config_el_tile_map
    
def stimulate_tiles(array, path, fname, rec_time, post_download_wait_time, 
                    s, stim_seq):
    array.download()
    time.sleep(post_download_wait_time)
        
    start_saving(s, dir_name=path, fname=fname)
    stim_seq.send()
    time.sleep(rec_time)
    # stimulation
    array.close()
    stop_saving(s)
            
def process_config(config_fullfname, path, rec_time, post_download_wait_time, s, stim_seq):
    array = maxlab.chip.Array()
    array.load_config(config_fullfname)
    array.download()
    
    config_el_tile_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    config_el_tile_map = config_el_tile_map.set_index('electrode')["tile"]

    stim_set_i = 0
    start_t = time.time()
    while True:
        _ = select_stim_els(config_el_tile_map, stim_set_i, array)
        stim_set_electrodes, stim_set_identifer, config_el_tile_map = _
        
        if len(stim_set_electrodes) == 0:
            print(f"\nNo electrode could be connected for {stim_set_i:02d}!")
            break
        
        fname = os.path.basename(config_fullfname).replace(".cfg", "") + f"_stimset{stim_set_i:03d}"
        print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ set {stim_set_i:02d}, "
            f"{len(stim_set_electrodes)} electrodes ")
        
        # save csv
        stim_set_identifer.to_csv(os.path.join(path, f"{fname}.csv"), index=False)
    
        stimulate_tiles(array, path, fname, rec_time, post_download_wait_time, 
                        s, stim_seq)
        
        print(f"Disconnecting {len(stim_set_electrodes)} stimulation electrodes...", end='')
        for stim_el in stim_set_electrodes:
            array.disconnect_electrode_from_stimulation(stim_el)
        print("Done.\n\n")
        
        stim_set_i += 1
        if len(config_el_tile_map) == 0:
            
            print(f"All els in config stimulated. Config done. Took {time.time()-start_t:.2f}s")
            break
    print("==="*20)
    print("==="*20)
        



def main():
    # ======== PARAMETERS ========
    subdir = "headstage_devices/MEA1K05/recordings"
    date = datetime.datetime.now().strftime("%Y%m%d")
    rec_dir = f"{date}_noGP_shortcut_stim_rec2"
    post_download_wait_time = .6
    log2file = True
    rec_time = 0.15
    gain = 112
    seed = 42
    configs_basepath = os.path.join(C.NAS_DIR, "mea1k_configs")
    # which_configs = "4x4_tile_meshgrid_seed42"
    which_configs = "9x3x16_meshgrid_seed42"
    # ======== PARAMETERS ========
    
    
    
    
    if log2file:
        log_fname = os.path.join(C.NAS_DIR, subdir, rec_dir, "log.txt")
        logfile = open(log_fname, "w")
        sys.stdout = logfile
    
    np.random.seed(seed)
    path = os.path.join(C.NAS_DIR, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(path)} - ", path)
    
    s = maxlab.Saving()
    stim_seq = create_stim_sine_sequence()
    reset_MEA1K(gain=gain, enable_stimulation_power=True)
    turn_on_stimulation_units(list(range(32)))
    
    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    for i, config_fullfname in enumerate(fnames):
        print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}")
        
        process_config(config_fullfname, path, rec_time, post_download_wait_time, s, stim_seq)
    
    if log2file:
        logfile.close()
        

if __name__ == "__main__":
    main()