import os
from glob import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ephys_constants as C
from mea1k_ephys import get_implant_mapping
from mea1k_utils import start_saving, stop_saving, try_routing

from mea1k_utils import setup_array, array_config2df

def make_bonding_config(device_name, date_str):
    # def try_bonded_routing(els):
    #     print(f"\n\nMaking config for mapping {len(els)} shank-connected-pads to"
    #             f" electrodes on MEA1K  chip...")
        
    #     array = setup_array(els, randomize_routing=True)
    #     # array.download()
    #     succ_routed = [m.electrode for m in array.get_config().mappings]
    #     failed_routing = [el for el in els if el not in succ_routed]
    #     print(f"Failed routing {len(failed_routing)}: {failed_routing}")
    #     array.close()
    #     return succ_routed, failed_routing

    implant_mapping = get_implant_mapping(C.NAS_DIR, device_name)
    # slice to electrodes under pads that are routed to a shank PI electrode
    implant_mapping = implant_mapping[implant_mapping.shank_id.notna()]

    sel_which_rank = 1
    els = implant_mapping[(implant_mapping.connectivity_order==sel_which_rank) & 
                          (implant_mapping.mea1k_connectivity>20)].mea1k_el.values
    while True:
        # succ_routed, failed_routing = try_bonded_routing(els)
        succ_routed, failed_routing, array = try_routing(els, randomize_routing=True)

        if len(failed_routing) == 0:
            print("Done.")
            break
        sel_which_rank += 1
        print(f"Trying alternative electrodes with connectivity rank {sel_which_rank}")
        
        # check which pad the failed electrodes are associated with
        missing_pads = implant_mapping[implant_mapping.mea1k_el.isin(failed_routing)].pad_id
        missing_pads = implant_mapping[implant_mapping.pad_id.isin(missing_pads)]
        
        alt_els = missing_pads[missing_pads.connectivity_order==sel_which_rank].mea1k_el
        good_enough_connec_mask = missing_pads[missing_pads.connectivity_order==sel_which_rank].mea1k_connectivity > 20
        print(f"{sum(good_enough_connec_mask)} / {len(good_enough_connec_mask)} alternative electrodes have good enough connectivity")
        alt_els = alt_els[good_enough_connec_mask].values
        els = succ_routed + alt_els.tolist()

    # array = setup_array(els)
    # array.download()
    config_fullfname = os.path.join(C.NAS_DIR, "implant_devices", C.DEVICE_NAME, 
                                    f"bonded_{C.DEVICE_NAME}_{date_str}_{len(els)}chnls.cfg")
    # csv of config
    config_mapping = array_config2df(array)
    config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
    
    # save config in mea1k specific format
    array.save_config(config_fullfname)
    array.close()

# def try_routing(els):
#     array = setup_array(els, randomize_routing=False)
#     array.download()
#     succ_routed = [m.electrode for m in array.get_config().mappings]
#     failed_routing = [el for el in els if el not in succ_routed]
#     print(f"Failed routing {len(failed_routing)}: {failed_routing}")
#     array.close()
#     return succ_routed, failed_routing

def make_full_padlayout_configs(device_name, ):
    implant_mapping = get_implant_mapping(C.NAS_DIR, device_name)
    # slice to electrodes under pads that are routed to a shank PI electrode
    implant_mapping = implant_mapping[implant_mapping.shank_id.notna()]

    configured_pads = []
    all_pads = implant_mapping.pad_id.unique()
    el_counter = 0 # for tracking progress only
    config_i =  0
    while True:
        config_i_els = []
        np.random.shuffle(all_pads)
        pad_counter = 0
        for pad_i in all_pads:
            if pad_i in configured_pads:
                continue
            pad_els = implant_mapping[implant_mapping.pad_id==pad_i].mea1k_el.values
            
            _, failed_routing, array = try_routing([*config_i_els, *pad_els], return_array=True)
            if len(failed_routing) != 0:
                # array = setup_array(config_i_els, randomize_routing=False)
                # array.download()
                
                config_fullfname = os.path.join(C.NAS_DIR, "implant_devices", 
                                                C.DEVICE_NAME, "full_padlayout_configs", 
                                                f"el_config_{config_i:03}_{pad_counter:03}pads.cfg")
                print(f"Saving config number {config_i:03} with {pad_counter:03} "
                      f"pads as {config_fullfname}")
                
                # csv of config
                config_mapping = array_config2df(array)
                config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
                # save config in mea1k specific format
                array.save_config(config_fullfname)
                array.close()
                config_i += 1
                el_counter += len(config_i_els)
                print(f"{el_counter}/26400 electrodes\n")
                break
            # continue with increased set of els
            config_i_els.extend(pad_els)
            configured_pads.append(pad_i)
            pad_counter += 1            
        
        if len(configured_pads) == len(all_pads):
            print(f"Done. Routing all pads in {config_i} configs.")
            break      


def sample_3x3_tiles(center_el):
    mea1k = np.arange(26400).reshape(120, 220)
    # excl_mask = np.isin(mea1k, exclude_els)
    # tile_size = 3
    
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


def get_all_9x3x16_meshgrid_electrodes():
    def new_tile_config(n_xtiles, n_ytiles, tile_size, overlap, xstep, ystep, 
                        debug_canvas=None):
        # Calculate the distance between tiles in the x and y directions
        xtile_dist = 220 // n_xtiles
        ytile_dist = 120 // n_ytiles
        
        # Generate x indices for the tiles
        x_indices = []
        for i in range(0, 220, xtile_dist):
            indices = np.array(range(i, i + tile_size)) + xstep * (tile_size - overlap)
            if any(indices >= 220):
                continue
            x_indices.append(indices)
            if xstep > 11:  # 11 for n_tiles = 9
                break

        # Generate y indices for the tiles
        y_indices = []
        for i in range(0, 120, ytile_dist):
            indices = np.array(range(i, i + tile_size)) + ystep * (tile_size - overlap)
            if any(indices >= 120):
                continue
            if ystep > 19:  # 18 for n_tiles = 3
                break
            y_indices.append(indices)
            
        # Initialize the configuration matrix and electrode mapping
        mea1k = np.arange(26400).reshape(120, 220)
        electrodes = {}
        tile_i = 0
        for y_idx in (y_indices):
            for x_idx in (x_indices):
                square_mesh_idx = np.meshgrid(y_idx, x_idx)
                electrodes[tile_i] = mea1k[square_mesh_idx[0], square_mesh_idx[1]].flatten()
                tile_i += 1
                if debug_canvas is not None:
                    debug_canvas[square_mesh_idx[0], square_mesh_idx[1]] += 1
        return electrodes

    # Square size: 4, ntiles: 9x3, 27, Overlap: 2, Steps: 11.0x15.0
    tile_size = 4
    overlap = 2
    n_xtiles = 9
    n_ytiles = 3
    
    debug_canvas = np.zeros((120, 220))
    all_tile_sets_els = []
    tile_set_i = 0
    while True:
        # Calculate the current step in the x and y directions
        xstep = tile_set_i % 12
        ystep = tile_set_i // 12
        
        # Break the loop if the ystep exceeds the maximum allowed value
        if ystep > 19:  # specific to the current tile configuration, last row
            # plt.imshow(debug_canvas) # debugging
            # plt.show()
            break
        
        # Generate the new tile configuration
        el_tile_config = new_tile_config(n_xtiles = n_xtiles, 
                                         n_ytiles = n_ytiles, 
                                         tile_size = tile_size, 
                                         overlap = overlap, 
                                         xstep = xstep, 
                                         ystep = ystep,
                                         debug_canvas = debug_canvas)
        
        # aggregate the electrode indices into a dataframe
        tile_set_els = []
        for tile_i, els in el_tile_config.items():
            tile_set_els.append(pd.Series(els, name=(tile_set_i,tile_i)))
        tile_set_els = pd.concat(tile_set_els, axis=1).T
        all_tile_sets_els.append(tile_set_els)

        tile_set_i += 1
    all_tile_sets_els = pd.concat(all_tile_sets_els)
    all_tile_sets_els.index.set_names(('set', 'tile'), inplace=True)
    return all_tile_sets_els
    
def make_9x3x16_meshgrid_config(config_dirname):
    fulldirname = os.path.join(C.NAS_DIR, "mea1k_configs", config_dirname)
    if not os.path.exists(fulldirname):
        os.makedirs(fulldirname)
        
    all_tile_sets_els = get_all_9x3x16_meshgrid_electrodes()
    print(all_tile_sets_els)
    
    configured_tile_indices = []
    all_tile_indices = np.arange(all_tile_sets_els.shape[0])
    config_i =  0
    el_counter = 0
    while True:
        tile_set_i_els = []
        matching_tile_indices = []
        np.random.shuffle(all_tile_indices)
        tile_counter = 0
        for tile_i in all_tile_indices:
            if tile_i in configured_tile_indices:
                continue
            
            # index the electrodes in the current tile
            tile_els = all_tile_sets_els.iloc[tile_i].values
            
            # enusre that the current set of electrodes do not overlap with 
            # the previous set, some tiles share electrodes (stride=2)
            if np.isin(tile_els, tile_set_i_els).any():
                continue
            
            # try routing the current set of electrodes
            _, failed_routing, array = try_routing([*tile_set_i_els, *tile_els], 
                                                   return_array=True)
            
            # if any electrodes failed routing, stop appending to the current set
            if len(failed_routing) != 0:
                # reroute to not have partial tile in config
                array = setup_array(tile_set_i_els, randomize_routing=False)
                
                fname = f"el_config_{config_i:03}_{tile_counter:03}tiles.cfg"
                config_fullfname = os.path.join(fulldirname, fname)
                print(f"Saving config number {config_i:03} with {tile_counter:03} "
                      f"tiles as {config_fullfname}")
                
                # csv of config
                config_mapping = array_config2df(array)
                config_mapping["tile"] = matching_tile_indices
                config_mapping.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
                # save config in mea1k specific format
                array.save_config(config_fullfname)
                array.close()
                config_i += 1
                el_counter += len(tile_set_i_els)
                print(f"{el_counter}/26400 electrodes routed\n")
                break
            # continue with increased set of els
            tile_set_i_els.extend(tile_els)
            matching_tile_indices.extend([tile_i]*len(tile_els))
            configured_tile_indices.append(tile_i)
            tile_counter += 1
    
        if len(configured_tile_indices) == len(all_tile_indices):
            print(f"Done. Routing all pads in {config_i} configs.")
            break 
    
def make_3x3_stim_config(config_dirname):
    fulldirname = os.path.join(C.NAS_DIR, "mea1k_configs", config_dirname)
    canvas = np.zeros((120, 220))
    mea1k_stim_els_left = np.arange(26400)
    
    config_routed_els, config_stim_els, config_el2tile_map = [], [], {}
    config_i, tile_i, fail_counter = 0, 0, 0
    while True:
        # attampt to route another 3x3 tile, with stim electrode in the center
        stim_el = np.random.choice(np.setdiff1d(mea1k_stim_els_left, config_routed_els))
        # print(len(mea1k_stim_els_left), end=' ')
        tile_els = sample_3x3_tiles(stim_el)
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
    
    
def main():
    # seed = 46
    # np.random.seed(seed)
    # make_bonding_config(device_name=C.DEVICE_NAME_RAT006)
    
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
    
    seed = 42
    np.random.seed(seed)
    make_3x3_stim_config(config_dirname=f"3x3_stim_seed{seed}")
    

    
if __name__ == "__main__":
    main()