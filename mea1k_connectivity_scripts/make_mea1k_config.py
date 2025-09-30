import os
import sys

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

# import parent dir with general modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mea1k_modules.mea1k_config_utils import array_config2df
from mea1k_modules.mea1k_config_utils import try_routing
from mea1k_modules.mea1k_post_processing import animal_name2implant_device
from mea1k_modules.mea1k_post_processing import get_raw_implant_mapping

def make_bonding_config(animal_name=None, implant_name=None):
    nas_dir = device_paths()[0]
    
    if animal_name is not None:
        # get the bonding mapping for the animal
        implant_name = animal_name2implant_device(animal_name)
    
    implant_mapping = get_raw_implant_mapping(animal_name=animal_name,
                                              implant_name=implant_name)
    implant_mapping = implant_mapping[implant_mapping.shank_id.notna()]
    print(implant_mapping)

    # first try to route the best connected electrodes under a pad, then try the next rank
    sel_which_rank = 1
    els = implant_mapping[(implant_mapping.connectivity_order==sel_which_rank) & 
                          (implant_mapping.mea1k_connectivity>.8)].mea1k_el.values
    while True:
        succ_routed, failed_routing, array = try_routing(els, randomize_routing=True,
                                                         return_array=True)

        if len(failed_routing) == 0:
            print("Done.")
            break
        sel_which_rank += 1
        print(f"Trying alternative electrodes with connectivity rank {sel_which_rank}")
        
        # check which pad the failed electrodes are associated with
        missing_pads = implant_mapping[implant_mapping.mea1k_el.isin(failed_routing)].pad_id
        missing_pads = implant_mapping[implant_mapping.pad_id.isin(missing_pads)]
        
        # get the alternative electrodes with good enough connectivity
        alt_els = missing_pads[missing_pads.connectivity_order==sel_which_rank].mea1k_el
        rank_mask = missing_pads.connectivity_order == sel_which_rank
        good_enough_connec_mask = missing_pads[rank_mask].mea1k_connectivity > .8
        print(f"{sum(good_enough_connec_mask)} / {len(good_enough_connec_mask)} "
              "alternative electrodes have good enough connectivity")
        alt_els = alt_els[good_enough_connec_mask].values
        els = succ_routed + alt_els.tolist()

    day = datetime.datetime.now().strftime("%d.%b")
    config_fullfname = os.path.join(nas_dir, "devices", "implant_devices", implant_name, 'bonding', 
                                    f"{animal_name}_{day}_{len(els)}ElConfig.cfg")
    # csv of config
    print(config_fullfname)
    config_mapping = array_config2df(array)
    config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
    
    # save config in mea1k specific format
    array.save_config(config_fullfname)
    array.close()
    
def els_around_3x3_tile(center_el):
    mea1k = np.arange(26400).reshape(120, 220)
    
    center_y, center_x = np.where(mea1k==center_el)
    center_y, center_x = center_y[0], center_x[0]
    tile_indices = np.meshgrid(range(center_y-1, center_y+2), 
                               range(center_x-1, center_x+2)) 
    # limit to valid indices
    tile_indices = np.array(tile_indices).reshape(2, 9)
    tile_indices = tile_indices[:, (tile_indices[0] >= 0) & (tile_indices[0] < 120) & 
                                  (tile_indices[1] >= 0) & (tile_indices[1] < 220)]
    tile_els = mea1k[tile_indices[0], tile_indices[1]].T.flatten()
    return tile_els

def make_tile_shortcut_stim_config(config_dirname):
    fulldirname = os.path.join(nas_dir, "mea1k_configs", config_dirname)
    canvas = np.zeros((120, 220))
    mea1k_stim_els_left = np.arange(26400)
    
    config_routed_els, config_stim_els, config_el2tile_map = [], [], {}
    config_i, tile_i, fail_counter = 0, 0, 0
    while True:
        # attampt to route another 3x3 tile, with stim electrode in the center
        stim_el = np.random.choice(np.setdiff1d(mea1k_stim_els_left, config_routed_els))
        # print(len(mea1k_stim_els_left), end=' ')
        tile_els = els_around_3x3_tile(stim_el)
        if np.isin(config_routed_els, tile_els).any():
            print("\nTile el overlap with other tiles. Skipping.")
            continue
        
        _, failed_routing, array = try_routing([*config_routed_els, *tile_els], 
                                               stim_electrodes=[*config_stim_els, stim_el],
                                               return_array=True)
        # failed_routing = [stim_el] if np.random.rand() > 0.99 else []
        
        # could the new tile be routed without chaning the previous config?
        if len(failed_routing) != 0:
            if fail_counter < 10:
                fail_counter += 1
                print(f"\tFail {fail_counter}/10", end=',')
                continue
            
            # finalize the current config, save it and start a new one
            else:
                array = try_routing(config_routed_els, stim_electrodes=config_stim_els,
                                    return_array=True)[2]
                print("Failed to route 10 times in a row. Stopping.")
                # new config, save current config
                fname = f"el_config_{config_i:03}_{tile_i:03}tiles.cfg"
                config_fullfname = os.path.join(fulldirname, fname)
                print(f"Saving config number {config_i:03} with {tile_i:03} "
                      f"tiles as {config_fullfname}. {len(mea1k_stim_els_left)} left\n")
                
                # csv of config
                config_mapping = array_config2df(array)
                config_mapping["tile"] = [config_el2tile_map[el] for el in config_mapping.electrode]
                config_mapping['stim'] = config_mapping.electrode.isin(config_stim_els)
                config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
                # save config in mea1k specific format
                array.save_config(config_fullfname)
                array.close()
                
                # update for next config
                config_routed_els, config_stim_els, config_el2tile_map = [], [], {}
                tile_i, fail_counter = 0, 0
                config_i += 1

        else:
            canvas[stim_el//220, stim_el%220] += 5
            canvas[tile_els//220, tile_els%220] += 1
            
            config_routed_els.extend(tile_els)
            config_el2tile_map.update({el:tile_i for el in tile_els})
            config_stim_els.append(stim_el)
            
            tile_i += 1
            fail_counter = 0
            # drop sampled electrode from the pool
            mea1k_stim_els_left = np.setdiff1d(mea1k_stim_els_left, [stim_el])
            if len(mea1k_stim_els_left) == 0:
                print("Done.")
                break
        
    plt.imshow(canvas)
    plt.show()

def make_single_el2stimunit_configs(output_dirname, animal_name=None, implant_name=None):
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)
        
    for i, el in enumerate(list(range(26400))): # try all electrodes
        succ, _, arr = try_routing([el], stim_electrodes=[el], return_array=True)
        stim_unit = arr.query_stimulation_at_electrode(el)
        amplifier = arr.query_amplifier_at_electrode(el)
        if stim_unit == '':
            print("Couldn't route this electrode. Skipping.")
            continue
        
        fname = f"el_config_El{el:05d}_StimUnit{int(stim_unit):02d}_Ampl{int(amplifier):04d}.cfg"
        dirname = os.path.join(output_dirname, f"StimUnit{int(stim_unit):02d}")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        config_fullfname = os.path.join(dirname, fname)

        # csv of config
        config_mapping = array_config2df(arr)
        
        config_mapping['stim_unit'] = [int(stim_unit)]
        config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
        # save config in mea1k specific format
        arr.save_config(config_fullfname)
        
    
    
def main():
    L = Logger()
    L.init_logger(None, None, "DEBUG")
    nas_dir = device_paths()[0]
    
    seed = 42
    np.random.seed(seed)
    
    implant_name = "250917_MEA1K12_H1628pad1shankB5"
    animal_name = None
    # make_bonding_config(animal_name=animal_name, implant_name=implant_name)

    output_dirname = os.path.join(nas_dir, "mea1k_configs", "single_el2stimunit_configs2",)
    make_single_el2stimunit_configs(output_dirname=output_dirname,
                                   animal_name=animal_name,
                                   implant_name=implant_name)

    # seed = 42
    # np.random.seed(seed)
    # make_tile_shortcut_stim_config(config_dirname=f"3x3_stim_seed{seed}")
    

if __name__ == "__main__":
    main()