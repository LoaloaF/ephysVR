import os
import sys

from time import sleep
# to import from parent dir
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from CustomLogger import CustomLogger as Logger

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ephys_constants as C
# from mea1k_ephys import get_implant_mapping
from mea1k_modules.mea1k_config_utils import setup_array, array_config2df, try_routing
from mea1k_modules.mea1k_raw_preproc import animal_name2implant_device, get_raw_implant_mapping

def make_bonding_config(animal_name=None, implant_name=None):
    nas_dir = C.device_paths()[0]
    
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
    
def make_whole_bonded_pad_stim_config(implant_name, config_dirname):
    def find_pad_stim_routing(implant_mapping, pad_id, ):
        stim_el_i = 0
        n_pads_around = 2
        pad_subset = implant_mapping[implant_mapping.pad_id==pad_id]
        L.logger.debug(f"pad_id {pad_id}")
        while True:
            if pad_subset.empty:
                L.logger.debug(f"Pad {pad_id} not sucesfully bonded. Skipping.")
                return None, None, None
            
            stim_el = pad_subset.mea1k_el.values[stim_el_i]
            L.logger.debug(stim_el)

            pad_order = implant_mapping.pad_id.unique()
            around_pad_subset_idx = np.where(pad_order==pad_id)[0][0]
            # according to the order in implant_mapping get the pads around the current pad
            around_pad_subset_idx_range = np.clip(np.arange(around_pad_subset_idx-n_pads_around,
                                                            around_pad_subset_idx+n_pads_around+1), 
                                                  0, len(pad_order))
            # get the electrodes in the vicinity of the current pad
            in_vicinity = implant_mapping[np.isin(implant_mapping.pad_id, 
                                                  pad_order[around_pad_subset_idx_range])]
            els_to_route = in_vicinity.mea1k_el.tolist()

            # TODO incl controls, somehting that isn't below a pad
            # if pad_id in [1, 0, 267]:
            #     if pad_id == 1:
            #         overr_el = 13091
            #     elif pad_id == 0:
            #         overr_el = 4811
            #     elif pad_id == 267:
            #         overr_el = 220
            #     L.logger.debug(f"overriding with el {overr_el}", )
            #     stim_el = overr_el
            #     els_to_route = els_to_route + [overr_el]

            _, failed_routing, array = try_routing(els_to_route, 
                                                   stim_electrodes=[stim_el],
                                                   return_array=True)
            if len(failed_routing) != 0:
                L.logger.debug(f"Failed routing {len(failed_routing)}: {failed_routing}")
                if pad_subset.mea1k_el.values.shape[0] > stim_el_i+1:
                    L.logger.info(f"Trying next stimulation electrode for pad {pad_id}")
                    stim_el_i += 1
                    continue
                elif n_pads_around > 0:
                    L.logger.info(f"Trying to route pad {pad_id} with {n_pads_around-1} pads around.")
                    n_pads_around -= 1
                    stim_el_i = 0
                    continue
                else:
                    L.logger.warning(f"Failed to route pad {pad_id}. Skipping.")
                    return None, None, None
            else:
                break
        L.logger.info(f"Successfully routed pad {pad_id}\n\n")
        stim_unit = array.query_stimulation_at_electrode(stim_el)
        return array, stim_el, stim_unit 
    
    L = Logger()
    
    nas_dir = C.device_paths()[0]
    fulldirname = os.path.join(nas_dir, "devices", "implant_devices", implant_name, 
                               "bonding", config_dirname)
    L.logger.debug(fulldirname)
    if not os.path.exists(fulldirname):
        os.makedirs(fulldirname)
    
    implant_mapping = get_raw_implant_mapping(nas_dir, implant_name)

    implant_mapping = implant_mapping[(implant_mapping.mea1k_connectivity>.8)]
    implant_mapping = implant_mapping.sort_values(['shank_id', 'depth', 'pad_id', 'connectivity_order']).reset_index(drop=True)
    L.logger.debug(implant_mapping)
    L.logger.info(f"Finding stimulation config for {len(implant_mapping.pad_id.unique())} pads...")
    L.spacer("debug")
    
    n_fails = 0
    for pad_id in implant_mapping.pad_id.unique():
        
        array, stim_el, stim_unit = find_pad_stim_routing(implant_mapping, pad_id)
        if array is None:
            n_fails += 1
            continue
        
        fname = f"el_config_pad{int(pad_id):04}.cfg"
        config_fullfname = os.path.join(fulldirname, fname)
        array.disconnect_electrode_from_stimulation(stim_el)
        
        # csv of config
        config_mapping = array_config2df(array)
        config_mapping['stim'] = config_mapping.electrode == stim_el
        config_mapping['depth'] = implant_mapping.set_index('mea1k_el').loc[config_mapping.electrode].depth.values
        config_mapping['shank'] = implant_mapping.set_index('mea1k_el').loc[config_mapping.electrode].shank_id.values
        config_mapping['stim_unit'] = [np.nan if not is_stim_el else stim_unit
                                       for is_stim_el in config_mapping.stim]
        config_mapping['pad_id'] = implant_mapping.set_index('mea1k_el').reindex(config_mapping.electrode).pad_id.values
        L.logger.debug(config_mapping)
        config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
        # save config in mea1k specific format
        array.save_config(config_fullfname)
        array.close()
        # if pad_id == 267:
        #     break
        L.spacer("debug")
    
    L.logger.info(f"Failed to route {n_fails} pads of {len(implant_mapping.pad_id.unique())} pads.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# #TODO CHECK what is relevant below here (3x3 stim ?)

# def extend_config_to_double_pad_stim(device_name, config_name):
#     nas_dir = C.device_paths()[0]
#     implant_mapping = get_raw_implant_mapping(nas_dir, device_name)
#     # slice to electrodes under pads that are routed to a shank PI electrode
#     implant_mapping = implant_mapping[implant_mapping.shank_id.isin((1, 2))]
#     print("Filtered implant_mapping by shank_id:\n", implant_mapping)
    
#     config_fullfname = os.path.join(nas_dir, "implant_devices", device_name, 'bonding', 
#                                     f"{config_name}_{device_name}.csv")
#     config = pd.read_csv(config_fullfname, index_col=None)
#     print("Config data:\n", config)
    
#     config = config[config.electrode.isin(implant_mapping.mea1k_el)]
#     prim_els = config.electrode.values
#     print("Filtered config by electrode:\n", config)
    
#     pads = implant_mapping[implant_mapping.mea1k_el.isin(config.electrode)].pad_id.unique()
#     print("Pads:\n", pads)
#     print("Number of pads:", len(pads))
    
#     potential_second_els = implant_mapping[
#         (implant_mapping.pad_id.isin(pads)) &
#         (implant_mapping.mea1k_connectivity > 20) &
#         (~implant_mapping.mea1k_el.isin(config.electrode))
#     ]
#     print("Potential second electrodes:\n", potential_second_els)


#     # Group by pad_id and get the index of the row with the lowest connectivity_order
#     idx = potential_second_els.groupby('pad_id')['connectivity_order'].idxmin()
#     # Use these indices to filter the original DataFrame
#     secondary_els = potential_second_els.loc[idx].mea1k_el.values
#     print("Selected second electrodes:\n", secondary_els)
    
#     merged_els = np.concatenate([prim_els, secondary_els])
#     succ_routed, failed_routing, array = try_routing(merged_els, randomize_routing=True,
#                                                      return_array=True)
#     print(f"Failed routing {len(failed_routing)}: {failed_routing}")
#     config_fullfname = os.path.join(nas_dir, "implant_devices", device_name, 'bonding', 
#                                     f"Stim_{config_name}_{device_name}.cfg")
#     # csv of config
#     print(config_fullfname)
#     config_mapping = array_config2df(array)
#     config_mapping['pads'] = implant_mapping.set_index('mea1k_el').loc[config_mapping.electrode].pad_id.values
#     config_mapping['shank'] = implant_mapping.set_index('mea1k_el').loc[config_mapping.electrode].shank_id.values
#     config_mapping['depth'] = implant_mapping.set_index('mea1k_el').loc[config_mapping.electrode].depth.values
#     config_mapping.sort_values(['shank', 'depth', 'pads'], inplace=True)    
#     print(config_mapping)
    
#     config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
    
#     # save config in mea1k specific format
#     array.save_config(config_fullfname)
#     array.close()
    
    
# # def try_routing(els):
# #     array = setup_array(els, randomize_routing=False)
# #     array.download()
# #     succ_routed = [m.electrode for m in array.get_config().mappings]
# #     failed_routing = [el for el in els if el not in succ_routed]
# #     print(f"Failed routing {len(failed_routing)}: {failed_routing}")
# #     array.close()
# #     return succ_routed, failed_routing


# def make_full_padlayout_configs(device_name, ):
#     implant_mapping = get_raw_implant_mapping(nas_dir, device_name)
#     # slice to electrodes under pads that are routed to a shank PI electrode
#     implant_mapping = implant_mapping[implant_mapping.shank_id.notna()]

#     configured_pads = []
#     all_pads = implant_mapping.pad_id.unique()
#     el_counter = 0 # for tracking progress only
#     config_i =  0
#     while True:
#         config_i_els = []
#         np.random.shuffle(all_pads)
#         pad_counter = 0
#         for pad_i in all_pads:
#             if pad_i in configured_pads:
#                 continue
#             pad_els = implant_mapping[implant_mapping.pad_id==pad_i].mea1k_el.values
            
#             _, failed_routing, array = try_routing([*config_i_els, *pad_els], return_array=True)
#             if len(failed_routing) != 0:
#                 # array = setup_array(config_i_els, randomize_routing=False)
#                 # array.download()
                
#                 config_fullfname = os.path.join(nas_dir, "implant_devices", 
#                                                 C.DEVICE_NAME, "full_padlayout_configs", 
#                                                 f"el_config_{config_i:03}_{pad_counter:03}pads.cfg")
#                 print(f"Saving config number {config_i:03} with {pad_counter:03} "
#                       f"pads as {config_fullfname}")
                
#                 # csv of config
#                 config_mapping = array_config2df(array)
#                 config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
#                 # save config in mea1k specific format
#                 array.save_config(config_fullfname)
#                 array.close()
#                 config_i += 1
#                 el_counter += len(config_i_els)
#                 print(f"{el_counter}/26400 electrodes\n")
#                 break
#             # continue with increased set of els
#             config_i_els.extend(pad_els)
#             configured_pads.append(pad_i)
#             pad_counter += 1            
        
#         if len(configured_pads) == len(all_pads):
#             print(f"Done. Routing all pads in {config_i} configs.")
#             break      


# def sample_3x3_tiles(center_el):
#     mea1k = np.arange(26400).reshape(120, 220)
#     # excl_mask = np.isin(mea1k, exclude_els)
#     # tile_size = 3
    
#     center_y, center_x = np.where(mea1k==center_el)
#     center_y, center_x = center_y[0], center_x[0]
#     tile_indices = np.meshgrid(range(center_y-1, center_y+2), 
#                                range(center_x-1, center_x+2)) 
#     # limit to valid indices
#     tile_indices = np.array(tile_indices).reshape(2, 9)
#     tile_indices = tile_indices[:, (tile_indices[0] >= 0) & (tile_indices[0] < 120) & 
#                                   (tile_indices[1] >= 0) & (tile_indices[1] < 220)]
#     tile_els = mea1k[tile_indices[0], tile_indices[1]].T.flatten()
#     return tile_els


# def get_all_9x3x16_meshgrid_electrodes():
#     def new_tile_config(n_xtiles, n_ytiles, tile_size, overlap, xstep, ystep, 
#                         debug_canvas=None):
#         # Calculate the distance between tiles in the x and y directions
#         xtile_dist = 220 // n_xtiles
#         ytile_dist = 120 // n_ytiles
        
#         # Generate x indices for the tiles
#         x_indices = []
#         for i in range(0, 220, xtile_dist):
#             indices = np.array(range(i, i + tile_size)) + xstep * (tile_size - overlap)
#             if any(indices >= 220):
#                 continue
#             x_indices.append(indices)
#             if xstep > 11:  # 11 for n_tiles = 9
#                 break

#         # Generate y indices for the tiles
#         y_indices = []
#         for i in range(0, 120, ytile_dist):
#             indices = np.array(range(i, i + tile_size)) + ystep * (tile_size - overlap)
#             if any(indices >= 120):
#                 continue
#             if ystep > 19:  # 18 for n_tiles = 3
#                 break
#             y_indices.append(indices)
            
#         # Initialize the configuration matrix and electrode mapping
#         mea1k = np.arange(26400).reshape(120, 220)
#         electrodes = {}
#         tile_i = 0
#         for y_idx in (y_indices):
#             for x_idx in (x_indices):
#                 square_mesh_idx = np.meshgrid(y_idx, x_idx)
#                 electrodes[tile_i] = mea1k[square_mesh_idx[0], square_mesh_idx[1]].flatten()
#                 tile_i += 1
#                 if debug_canvas is not None:
#                     debug_canvas[square_mesh_idx[0], square_mesh_idx[1]] += 1
#         return electrodes

#     # Square size: 4, ntiles: 9x3, 27, Overlap: 2, Steps: 11.0x15.0
#     tile_size = 4
#     overlap = 2
#     n_xtiles = 9
#     n_ytiles = 3
    
#     debug_canvas = np.zeros((120, 220))
#     all_tile_sets_els = []
#     tile_set_i = 0
#     while True:
#         # Calculate the current step in the x and y directions
#         xstep = tile_set_i % 12
#         ystep = tile_set_i // 12
        
#         # Break the loop if the ystep exceeds the maximum allowed value
#         if ystep > 19:  # specific to the current tile configuration, last row
#             # plt.imshow(debug_canvas) # debugging
#             # plt.show()
#             break
        
#         # Generate the new tile configuration
#         el_tile_config = new_tile_config(n_xtiles = n_xtiles, 
#                                          n_ytiles = n_ytiles, 
#                                          tile_size = tile_size, 
#                                          overlap = overlap, 
#                                          xstep = xstep, 
#                                          ystep = ystep,
#                                          debug_canvas = debug_canvas)
        
#         # aggregate the electrode indices into a dataframe
#         tile_set_els = []
#         for tile_i, els in el_tile_config.items():
#             tile_set_els.append(pd.Series(els, name=(tile_set_i,tile_i)))
#         tile_set_els = pd.concat(tile_set_els, axis=1).T
#         all_tile_sets_els.append(tile_set_els)

#         tile_set_i += 1
#     all_tile_sets_els = pd.concat(all_tile_sets_els)
#     all_tile_sets_els.index.set_names(('set', 'tile'), inplace=True)
#     return all_tile_sets_els
    
# def make_9x3x16_meshgrid_config(config_dirname):
#     fulldirname = os.path.join(nas_dir, "mea1k_configs", config_dirname)
#     if not os.path.exists(fulldirname):
#         os.makedirs(fulldirname)
        
#     all_tile_sets_els = get_all_9x3x16_meshgrid_electrodes()
#     print(all_tile_sets_els)
    
#     configured_tile_indices = []
#     all_tile_indices = np.arange(all_tile_sets_els.shape[0])
#     config_i =  0
#     el_counter = 0
#     while True:
#         tile_set_i_els = []
#         matching_tile_indices = []
#         np.random.shuffle(all_tile_indices)
#         tile_counter = 0
#         for tile_i in all_tile_indices:
#             if tile_i in configured_tile_indices:
#                 continue
            
#             # index the electrodes in the current tile
#             tile_els = all_tile_sets_els.iloc[tile_i].values
            
#             # enusre that the current set of electrodes do not overlap with 
#             # the previous set, some tiles share electrodes (stride=2)
#             if np.isin(tile_els, tile_set_i_els).any():
#                 continue
            
#             # try routing the current set of electrodes
#             _, failed_routing, array = try_routing([*tile_set_i_els, *tile_els], 
#                                                    return_array=True)
            
#             # if any electrodes failed routing, stop appending to the current set
#             if len(failed_routing) != 0:
#                 # reroute to not have partial tile in config
#                 array = setup_array(tile_set_i_els, randomize_routing=False)
                
#                 fname = f"el_config_{config_i:03}_{tile_counter:03}tiles.cfg"
#                 config_fullfname = os.path.join(fulldirname, fname)
#                 print(f"Saving config number {config_i:03} with {tile_counter:03} "
#                       f"tiles as {config_fullfname}")
                
#                 # csv of config
#                 config_mapping = array_config2df(array)
#                 config_mapping["tile"] = matching_tile_indices
#                 config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
#                 # save config in mea1k specific format
#                 array.save_config(config_fullfname)
#                 array.close()
#                 config_i += 1
#                 el_counter += len(tile_set_i_els)
#                 print(f"{el_counter}/26400 electrodes routed\n")
#                 break
#             # continue with increased set of els
#             tile_set_i_els.extend(tile_els)
#             matching_tile_indices.extend([tile_i]*len(tile_els))
#             configured_tile_indices.append(tile_i)
#             tile_counter += 1
    
#         if len(configured_tile_indices) == len(all_tile_indices):
#             print(f"Done. Routing all pads in {config_i} configs.")
#             break 

# def make_invivo_stim_config(implant_name, config_dirname):
#     fulldirname = os.path.join(nas_dir, "implant_devices", implant_name, 
#                                "bonding", config_dirname)
#     print(fulldirname)
#     if not os.path.exists(fulldirname):
#         os.makedirs(fulldirname)
    
#     implant_mapping = get_raw_implant_mapping(nas_dir, implant_name)
#     implant_mapping = implant_mapping[(implant_mapping.shank_id<3) &
#                                       (implant_mapping.mea1k_connectivity>20) &
#                                       (implant_mapping.connectivity_order<4)]
#     implant_mapping = implant_mapping.sort_values(['shank_id', 'depth', 'connectivity_order']).reset_index(drop=True)
#     print(implant_mapping)
    
#     for pad_id in implant_mapping.pad_id.unique():
#         pad_subset = implant_mapping[implant_mapping.pad_id==pad_id]
#         stim_el = pad_subset.mea1k_el.values[0]
#         pad_depth = pad_subset.depth.values[0]
#         pad_shank = pad_subset.shank_id.values[0]
#         print(pad_depth)
#         in_vicinity = implant_mapping[(implant_mapping.shank_id==pad_shank) &
#                                       ((implant_mapping.depth > pad_depth-400) &
#                                       (implant_mapping.depth < pad_depth+400))]        
#         print(in_vicinity)
#         _, failed_routing, array = try_routing(in_vicinity.mea1k_el.values, 
#                                                stim_electrodes=[stim_el],
#                                                return_array=True)
#         if len(failed_routing) != 0:
#             print(f"Failed routing {len(failed_routing)}: {failed_routing}")
#             continue
#         fname = f"el_config_S{int(pad_shank)}D{int(pad_depth):04}.cfg"
#         config_fullfname = os.path.join(fulldirname, fname)
        
#         # csv of config
#         config_mapping = array_config2df(array)
#         config_mapping['stim'] = config_mapping.electrode == stim_el
#         config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
#         # save config in mea1k specific format
#         array.save_config(config_fullfname)
#         array.close()
    
# def make_3x3_stim_config(config_dirname):
#     fulldirname = os.path.join(nas_dir, "mea1k_configs", config_dirname)
#     canvas = np.zeros((120, 220))
#     mea1k_stim_els_left = np.arange(26400)
    
#     config_routed_els, config_stim_els, config_el2tile_map = [], [], {}
#     config_i, tile_i, fail_counter = 0, 0, 0
#     while True:
#         # attampt to route another 3x3 tile, with stim electrode in the center
#         stim_el = np.random.choice(np.setdiff1d(mea1k_stim_els_left, config_routed_els))
#         # print(len(mea1k_stim_els_left), end=' ')
#         tile_els = sample_3x3_tiles(stim_el)
#         if np.isin(config_routed_els, tile_els).any():
#             print("\nTile el overlap with other tiles. Skipping.")
#             continue
        
#         _, failed_routing, array = try_routing([*config_routed_els, *tile_els], 
#                                                stim_electrodes=[*config_stim_els, stim_el],
#                                                return_array=True)
#         # failed_routing = [stim_el] if np.random.rand() > 0.99 else []
        
#         # could the new tile be routed without chaning the previous config?
#         if len(failed_routing) != 0:
#             if fail_counter < 10:
#                 fail_counter += 1
#                 print(f"\tFail {fail_counter}/10", end=',')
#                 continue
            
#             # finalize the current config, save it and start a new one
#             else:
#                 array = try_routing(config_routed_els, stim_electrodes=config_stim_els,
#                                     return_array=True)[2]
#                 print("Failed to route 10 times in a row. Stopping.")
#                 # new config, save current config
#                 fname = f"el_config_{config_i:03}_{tile_i:03}tiles.cfg"
#                 config_fullfname = os.path.join(fulldirname, fname)
#                 print(f"Saving config number {config_i:03} with {tile_i:03} "
#                       f"tiles as {config_fullfname}. {len(mea1k_stim_els_left)} left\n")
                
#                 # csv of config
#                 config_mapping = array_config2df(array)
#                 config_mapping["tile"] = [config_el2tile_map[el] for el in config_mapping.electrode]
#                 config_mapping['stim'] = config_mapping.electrode.isin(config_stim_els)
#                 config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
#                 # save config in mea1k specific format
#                 array.save_config(config_fullfname)
#                 array.close()
                
#                 # update for next config
#                 config_routed_els, config_stim_els, config_el2tile_map = [], [], {}
#                 tile_i, fail_counter = 0, 0
#                 config_i += 1

#         else:
#             canvas[stim_el//220, stim_el%220] += 5
#             canvas[tile_els//220, tile_els%220] += 1
            
#             config_routed_els.extend(tile_els)
#             config_el2tile_map.update({el:tile_i for el in tile_els})
#             config_stim_els.append(stim_el)
            
#             tile_i += 1
#             fail_counter = 0
#             # drop sampled electrode from the pool
#             mea1k_stim_els_left = np.setdiff1d(mea1k_stim_els_left, [stim_el])
#             if len(mea1k_stim_els_left) == 0:
#                 print("Done.")
#                 break
        
#     plt.imshow(canvas)
#     plt.show()

# def determine_el2stimuni_map(output_dirname)
def make_single_el2stimunit_configs(output_dirname, animal_name=None, implant_name=None):
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)
        
    # if animal_name is not None:
    #     # get the bonding mapping for the animal
    #     implant_name = animal_name2implant_device(animal_name)
    
    # implant_mapping = get_raw_implant_mapping(animal_name=animal_name,
    #                                           implant_name=implant_name)
    # print(implant_mapping)
    # mea1k_els = implant_mapping.mea1k_el[implant_mapping.mea1k_connectivity>.8].values
    # mea1k_els_neg_controls = np.random.choice(np.setdiff1d(np.arange(26400), mea1k_els), 
    #                                           size=100, replace=False)
        
    # mea1k_stim_els_left = np.concatenate([mea1k_els, mea1k_els_neg_controls])
    
    # for i, el in enumerate(mea1k_stim_els_left):
    # for i, el in enumerate(list(range(26400))): # try all electrodes
    for i, el in enumerate(list(range(20795,26400))): # try all electrodes
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
        
def make_complete_impedance_config(output_dirname, animal_name=None, implant_name=None):
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    if animal_name is not None:
        # get the bonding mapping for the animal
        implant_name = animal_name2implant_device(animal_name)
    
    implant_mapping = get_raw_implant_mapping(animal_name=animal_name,
                                              implant_name=implant_name)
    print(implant_mapping)
    mea1k_els = implant_mapping.mea1k_el[implant_mapping.mea1k_connectivity>.8].values
    mea1k_els_neg_controls = np.random.choice(np.setdiff1d(np.arange(26400), mea1k_els), 
                                              size=100, replace=False)
        
    mea1k_stim_els_left = np.concatenate([mea1k_els, mea1k_els_neg_controls])
            
    config_routed_els, config_el2tile_map = [], {}
    config_i, fail_counter = 0, 0
    while True:
        # attampt to route another electrode
        new_config_el = np.random.choice(np.setdiff1d(mea1k_stim_els_left, config_routed_els))
        
        _, failed_routing, array = try_routing([*config_routed_els, new_config_el], 
                                               stim_electrodes=[*config_routed_els, new_config_el],
                                               return_array=True)
        # failed_routing = [stim_el] if np.random.rand() > 0.99 else []
        
        # could the new tile be routed without chaning the previous config?
        if len(failed_routing) != 0:
            if fail_counter < 20:
                fail_counter += 1
                print(f"\tFail {fail_counter}/20", end=',')
                continue
            
            # finalize the current config, save it and start a new one
            else:
                array = try_routing(config_routed_els, stim_electrodes=config_routed_els,
                                    return_array=True)[2]
                print("Failed to route 20 times in a row. Stopping.")
                # new config, save current config
                fname = f"el_config_{config_i:03}_n{len(config_routed_els):02}_impedance.cfg"
                config_fullfname = os.path.join(output_dirname, fname)
                print(f"Saving config number {config_i:03} with {len(config_routed_els):02} "
                      f"electrodes as {config_fullfname}. {len(mea1k_stim_els_left)} left\n")
                
                # csv of config
                config_mapping = array_config2df(array)
                if config_mapping.empty:
                    print("Couldn't route any more electrodes. Done.")
                    break
                print(config_mapping)
                config_mapping['stim_unit'] = [array.query_stimulation_at_electrode(el) 
                                               for el in config_mapping.electrode]
                print(config_mapping)

                config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
                # save config in mea1k specific format
                array.save_config(config_fullfname)
                array.close()
                
                # update for next config
                config_routed_els, config_el2tile_map = [], {}
                fail_counter = 0
                config_i += 1

        # success, append the new electrode to the config
        else:
            config_routed_els.append(new_config_el)
            fail_counter = 0 # reset fail counter
            # drop sampled electrode from the pool
            mea1k_stim_els_left = np.setdiff1d(mea1k_stim_els_left, [new_config_el])
            if len(mea1k_stim_els_left) == 0:
                print("Done.")
                break
        
    
    
def main():
    L = Logger()
    L.init_logger(None, None, "DEBUG")
    nas_dir = C.device_paths()[0]
    
    seed = 42
    np.random.seed(seed)
    
    implant_name = "250917_MEA1K12_H1628pad1shankB5"
    animal_name = None
    # make_bonding_config(animal_name=animal_name, implant_name=implant_name)

    output_dirname = os.path.join(nas_dir, "mea1k_configs", "single_el2stimunit_configs2",)
    make_single_el2stimunit_configs(output_dirname=output_dirname,
                                   animal_name=animal_name,
                                   implant_name=implant_name)

    # implant_name = "250205_MEA1K03_H1278pad4shankB5"
    # implant_name = "241211_MEA1K06_H1278pad4shankB5"
    # implant_name = "250917_MEA1K12_H1628pad1shankB5"
    # make_whole_bonded_pad_stim_config(implant_name=implant_name,
    #                                   config_dirname="imp_rec_configs")
    
    # make_invivo_stim_config(implant_name=C.DEVICE_NAME_RAT006, 
    #                         config_dirname="invivo_localstim_configs")

    # seed = 1
    # np.random.seed(seed)
    # make_9x3x16_meshgrid_config(config_dirname=f"9x3x16_meshgrid_S{seed}")
    # seed = 2
    # np.random.seed(seed)
    # make_9x3x16_meshgrid_config(config_dirname=f"9x3x16_meshgrid_S{seed}")
    # seed = 3
    # np.random.seed(seed)
    # make_9x3x16_meshgrid_config(config_dirname=f"9x3x16_meshgrid_S{seed}")
    
    # # TODO check that this still runs fine, then detele try_bonded_routing
    # seed = 1
    # np.random.seed(seed)
    # make_full_padlayout_configs(device_name=C.DEVICE_NAMfE_RAT006)
    
    # seed = 42
    # np.random.seed(seed)
    # make_9x3x16_meshgrid_config(config_dirname=f"9x3x16_meshgrid_seed{seed}")
    
    # seed = 42
    # np.random.seed(seed)
    # make_3x3_stim_config(config_dirname=f"3x3_stim_seed{seed}")
    

    
if __name__ == "__main__":
    main()