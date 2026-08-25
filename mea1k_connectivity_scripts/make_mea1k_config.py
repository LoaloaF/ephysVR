import os
import sys
import glob

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

from mea1k_modules.mea1k_visualizations import draw_mea1k

def make_bonding_config(animal_name=None, implant_name=None, shank_subset=None,
                        connectivity_threshold=0.6, route_every_pad=False):
    nas_dir = device_paths()[0]
    
    if animal_name is not None:
        # get the bonding mapping for the animal
        implant_name = animal_name2implant_device(animal_name)
    
    implant_mapping = get_raw_implant_mapping(animal_name=animal_name,
                                              implant_name=implant_name)
    if shank_subset is not None:
        implant_mapping = implant_mapping[implant_mapping.shank_id.isin(shank_subset)]
    
    sel_which_rank = 1
    key = 'pad_connectivity_rank' if route_every_pad else 'el_connectivity_rank'
    # first try to route the best connected electrodes under a pad, then try the next rank
    els = implant_mapping[(implant_mapping[key] == sel_which_rank) & 
                          (implant_mapping.mea1k_connectivity>connectivity_threshold)].mea1k_el.values.astype(int).tolist()
    print(f"Routing MEA1K electrodes with connectivity > {connectivity_threshold}"
          f" with {route_every_pad=}: n={len(els)}")

    while True:
        succ_routed, failed_routing, array = try_routing(els, randomize_routing=True,
                                                         return_array=True)
        print(f"Successfully routed {len(succ_routed)} electrodes. Failed to "
              f"route {len(failed_routing)} electrodes.")
        if len(failed_routing) == 0:
            print("Done.")
            break
        # if len(succ_routed) != len(els):
        #     print("Some electrodes failed to route. Retrying with randomization.")
        #     continue
        sel_which_rank += 1
        print(f"Trying alternative electrodes with connectivity rank {sel_which_rank}")
        
        # check which pad the failed electrodes are associated with
        missing_pads = implant_mapping[implant_mapping.mea1k_el.isin(failed_routing)].pad_id
        missing_pads = implant_mapping[implant_mapping.pad_id.isin(missing_pads)]
        
        # get the alternative electrodes with good enough connectivity
        alt_els = missing_pads[missing_pads[key]==sel_which_rank].mea1k_el
        rank_mask = missing_pads[key] == sel_which_rank
        good_enough_connec_mask = missing_pads[rank_mask].mea1k_connectivity > connectivity_threshold
        print(f"{sum(good_enough_connec_mask)} / {len(good_enough_connec_mask)} "
              f"alternative electrodes have good enough connectivity")
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

def make_tile_shortcut_stim_config(config_dirname, stim_electrodes=np.arange(26400)):
    nas_dir = device_paths()[0]
    fulldirname = os.path.join(nas_dir, "mea1k_configs", config_dirname)
    if not os.path.exists(fulldirname):
        os.mkdir(fulldirname)
    canvas = np.zeros((120, 220))
    mea1k_stim_els_left = stim_electrodes.copy()
    
    # --- RESUME LOGIC ---
    config_i = 0
    existing_csvs = glob.glob(os.path.join(fulldirname, "el_config_*.csv"))
    if existing_csvs:
        done_stims = []
        for f in existing_csvs:
            df = pd.read_csv(f)
            done_stims.extend(df[df['stim'] == True]['electrode'].tolist())
            config_i = max(config_i, int(os.path.basename(f).split('_')[2]) + 1)
        mea1k_stim_els_left = np.setdiff1d(mea1k_stim_els_left, done_stims)
        print(f"Resuming: {len(mea1k_stim_els_left)} stim electrodes left to process.")

    if len(mea1k_stim_els_left) == 0:
        print("All configurations are already completed!")
        return
    # --------------------
    config_routed_els, config_stim_els, config_el2tile_map = [], [], {}
    tile_i, fail_counter = 0, 0
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

def make_tile_shortcut_stim_config_from_mask(config_dirname, mask_path):
    # read png mask
    mask = plt.imread(mask_path) < 0.5
    stim_electrodes = np.arange(26400)[mask.flatten()]
    print(f"Found {len(stim_electrodes)} (/26400) stim electrodes in the mask.")
    make_tile_shortcut_stim_config(config_dirname=config_dirname, stim_electrodes=stim_electrodes)

def make_single_el2stimunit_configs(output_dirname):
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
    
    # read back all csv files and combine them into one
    all_csvs = glob.glob(os.path.join(output_dirname, "**", "*.csv"), recursive=True)
    print(f"Found {len(all_csvs)} csv files in {output_dirname}. Combining them into one.")
    data = []
    for i,f in enumerate(sorted(all_csvs)):
        print(f"Read {i+1}/{len(all_csvs)}", end='\r')
        df = pd.read_csv(f)
        df['config_fname'] = os.path.basename(f)
        data.append(df)
    all_configs = pd.concat(data, ignore_index=True).sort_values('electrode').reset_index(drop=True)
    
    fullfname = os.path.join(output_dirname, '..', "el2stimunit.csv")
    all_configs.to_csv(fullfname, index=False)
    print(f"Saved all single_el2stimunit configs to {fullfname},\n{all_configs}")  

def make_single_el_stim_configs(animal_name=None, implant_name=None, shank_subset=None, 
                                connectivity_threshold=0.8, route_every_pad=False):
    nas_dir = device_paths()[0]
    
    if animal_name is not None:
        # get the bonding mapping for the animal
        implant_name = animal_name2implant_device(animal_name)
    
    implant_mapping = get_raw_implant_mapping(animal_name=animal_name,
                                              implant_name=implant_name)
    implant_mapping = implant_mapping[implant_mapping.shank_id.notna()]
    if shank_subset is not None:
        implant_mapping = implant_mapping[implant_mapping.shank_id.isin(shank_subset)]
    
    # first try to route the best connected electrodes under a pad, then try the next rank
    sel_which_rank = 1
    key = 'pad_connectivity_rank' if route_every_pad else 'el_connectivity_rank'
    # key = 'pad_impedance_rank' if route_every_pad else 'el_impedance_rank'
    els = implant_mapping[(implant_mapping[key] == sel_which_rank) & 
                          (implant_mapping.mea1k_connectivity > connectivity_threshold)].mea1k_el.values.astype(int)
    while True:
        succ_routed, failed_routing, array = try_routing(els, randomize_routing=True,
                                                         return_array=True)

        if len(failed_routing) == 0:
            print("Done base routing.")
            break
        sel_which_rank += 1
        print(f"Trying alternative electrodes with connectivity rank {sel_which_rank}")
        
        # check which pad the failed electrodes are associated with
        missing_pads = implant_mapping[implant_mapping.mea1k_el.isin(failed_routing)].pad_id
        missing_pads = implant_mapping[implant_mapping.pad_id.isin(missing_pads)]
        
        # get the alternative electrodes with good enough connectivity
        alt_els = missing_pads[missing_pads[key] == sel_which_rank].mea1k_el
        rank_mask = missing_pads[key] == sel_which_rank
        good_enough_connec_mask = missing_pads[rank_mask].mea1k_connectivity > connectivity_threshold
        print(f"{sum(good_enough_connec_mask)} / {len(good_enough_connec_mask)} "
              "alternative electrodes have good enough connectivity")
        alt_els = alt_els[good_enough_connec_mask].values.astype(int)
        els = succ_routed + alt_els.tolist()
    base_els = els
    
    day = datetime.datetime.now().strftime("%d.%b")
    bond_dir = os.path.join(nas_dir, "devices", "implant_devices", implant_name, 'bonding')
    
    config_dirname = f"{animal_name if animal_name else implant_name}_{day}_{len(base_els)}El_SingleStimConfigs"
    fulldirname = os.path.join(bond_dir, config_dirname)
    if not os.path.exists(fulldirname):
        os.makedirs(fulldirname)

    print(f"Generating single stim configs in {fulldirname}")
    for i, stim_el in enumerate(base_els):
        
        succ_routed, failed_routing, array = try_routing(base_els, stim_electrodes=[stim_el], return_array=True)
        if len(failed_routing) > 0:
             print(f"Failed to route stim for el {stim_el}")
             continue
        print(f"Generating config for stim el {stim_el} ({i+1}/{len(base_els)})")
             
        fname = f"el_config_El{stim_el:05d}.cfg"
        config_fullfname = os.path.join(fulldirname, fname)
        
        config_mapping = array_config2df(array)
        config_mapping['stim'] = config_mapping.electrode == stim_el
        config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
        
        array.save_config(config_fullfname)
        array.close()

def main():
    L = Logger()
    L.init_logger(None, None, "DEBUG")
    seed = 42
    np.random.seed(seed)
    
    # implant_name = "260413_MEA1K22_S1688pad14shankB5"
    # implant_name = "260501_MEA1K24_S1688pad14shankB5"
    # implant_name = "260602_MEA1K23_S844pad8shankB5"
    # implant_name = "260611_MEA1K24_S1688pad14shankB5"
    # implant_name = "260625_MEA1K23_S1688pad14shankB6"
    implant_name = "260715_MEA1K24_S1688pad14shankB7"
    animal_name = None
    
    # make_bonding_config(animal_name=animal_name, implant_name=implant_name,
    #                     connectivity_threshold=0.15, route_every_pad=True,
    #                     shank_subset=None, )

    # needs to be only run once, is done for each 26400 electrodes
    # output_dirname = os.path.join(nas_dir, "mea1k_configs", "single_el2stimunit_configs",)
    # make_single_el2stimunit_configs(output_dirname=output_dirname)
    
    # make single el stim configs for all electrodes with good enough connectivity, trying to keep the same stim electrode as much as possible across configs
    make_single_el_stim_configs(animal_name=animal_name, implant_name=implant_name, 
                                shank_subset=[1,2,3,4,5,5,6,7,8], 
                                connectivity_threshold=.4, route_every_pad=True) 
    make_single_el_stim_configs(animal_name=animal_name, implant_name=implant_name, 
                                shank_subset=[9,10,11,12,13,14], 
                                # shank_subset=[1,2,3,4,5,5,6,7,8], 
                                connectivity_threshold=.4, route_every_pad=True)
    
    # subdir = (f'{nas_dir}/devices/headstage_devices/MEA1K22/recordings/'
    #           '2026-07-06_20.43.57_ShinTsuFlim_Try4_PT1_EcoFLexNewFilmSpringInteronnect/')
    # make_single_el_stim_configs_from_shorts(subdir, output_dir=f'{nas_dir}/mea1k_configs/shorted_islands_configs',
    #                                         rect_xy=(47*17.5, 45*17.5),
    #                                         rect_wh=(25*17.5, 70*17.5))

    # vis_mea1k_config(dirname=os.path.join(nas_dir, "devices", "implant_devices", implant_name, 'bonding'),
                    #   config_fname="None_21.Apr_1337ElConfig.csv",
    # vis_mea1k_config(dirname=os.path.join(nas_dir, "devices", "implant_devices", implant_name, 'bonding', '260413_MEA1K22_S1688pad14shankB5_21.Apr_554El_SingleStimConfigs'),
    # vis_mea1k_config(dirname=os.path.join(nas_dir, "devices", "implant_devices", implant_name, 'bonding', '260413_MEA1K22_S1688pad14shankB5_21.Apr_783El_SingleStimConfigs'),
    #                   config_fname="*.csv",
    #                   implant_mapping=get_raw_implant_mapping(animal_name=animal_name, implant_name=implant_name))


    # make_external_current_configs(animal_name=animal_name, implant_name=implant_name,
    #                               fulldirname=os.path.join(nas_dir, "devices", "implant_devices", implant_name, 'bonding', 'external_current_configs'),
    #                               min_els_per_config=0,
    #                               max_el_per_pad=4, connectivity_threshold=0.2)
    

    # make_single_el_stim_configs(animal_name=animal_name, implant_name=implant_name, 
    #                             # shank_subset=[1,2,3,4,5,5,6,7,8], 
    #                             connectivity_threshold=.5, route_every_pad=True)

    # seed = 42
    # np.random.seed(seed)
    # make_tile_shortcut_stim_config(config_dirname=f"3x3_stim_seed{seed}")
    
    # seed = 42
    # np.random.seed(seed)
    # make_tile_shortcut_stim_config_from_mask(config_dirname=f"3x3_islandsubset2_stim_seed{seed}",
    #                                         mask_path="/home/houmanjava//Pictures/islands_mask.png")
    

if __name__ == "__main__":
    main()