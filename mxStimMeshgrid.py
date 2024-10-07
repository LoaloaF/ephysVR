import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import maxlab

def reset_MEA1K():
    print("Resetting MEA1K...", end='', flush=True)
    maxlab.util.initialize()
    maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
    maxlab.send(maxlab.chip.Amplifier().set_gain(7))
    print("Done.")

def setup_array(electrodes, stim_electrodes=None, config_name="default_name"):
    print("Setting up array (reset,route&download)...", end='', flush=True)
    # array = maxlab.chip.Array("offline")
    array = maxlab.chip.Array()
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(electrodes)
    array.connect_all_floating_amplifiers()
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
        # stim.set_current_mode()
        # stim.set_voltage_mode()
        # stim.set_small_current_range()
        # stim.set_large_current_range()
        stim.dac_source(0)
        maxlab.send(stim)
        time.sleep(.1)
    print("Done.")

def create_stim_sine_sequence(dac=0, amplitude=25, f=1000, ncycles=300, nreps=1):
	seq = maxlab.Sequence()
  	
	# 50 us * 20kHz = 1000 samples, 1 khz exmaple
	sampling_rate = 20000  # 20 kHz
	# Create a time array
	t = np.linspace(0,1, int(sampling_rate/f))
	# Create a sine wave with a frequency of 1 kHz
	sine_wave = (amplitude * np.sin(t*2*np.pi)).astype(int) +512
	# debug = []
	for i in range(nreps):
		seq.append(maxlab.system.DelaySamples(40))
		for j in range(ncycles):
			for ampl in sine_wave:
				seq.append(maxlab.chip.DAC(dac, ampl))
				# debug.append(ampl)
	# np.save("sine_wave.npy", debug)
	# plt.show()

	return seq

# def create_stim_sequence(dac=0, amplitude=25, npulses=10, nreps=3, inter_pulse_interval=100, rep_delay_s=.1):
#     def append_stimulation_pulse(seq, amplitude):
#         seq.append(maxlab.chip.DAC(0, 512-amplitude))
#         seq.append(maxlab.system.DelaySamples(4))
#         seq.append(maxlab.chip.DAC(0, 512+amplitude))
#         seq.append(maxlab.system.DelaySamples(4))
#         seq.append(maxlab.chip.DAC(0, 512))
#         return seq

#     seq = maxlab.Sequence()
#     for i in range(nreps):
#         for j in range(npulses):
#             append_stimulation_pulse(seq, amplitude) # 25 *2.83mV - current mode?
#             seq.append(maxlab.system.DelaySamples(inter_pulse_interval)) #5ms
#         time.sleep(rep_delay_s)
#     return seq

# def connect_el2stim_units(array, stim_electrodes):
#     # stim_els collects electrodes that are sucessfully connected    
#     stim_els, stim_units = [], []
#     # failed_stim_els collects electrodes where no stimulation units could be connected to
#     failed_stim_els = []
#     for el in stim_electrodes:
#         array.connect_electrode_to_stimulation(el)
#         stim_unit = array.query_stimulation_at_electrode(el)
        
#         # unknown error case, could not find routing?
#         if not stim_unit:
#             print(f"Warning - Could not connect El{el} to a stim unit.")
#             failed_stim_els.append(el)
        
#         # stim unit not used yet, 
#         elif int(stim_unit) not in stim_units:
#             stim_units.append(int(stim_unit))
#             stim_els.append(el)
            
#             if len(stim_units) == 32:
#                 print("Used up all 32 stim units.")
#                 break
        
#         # stim unit already assigned case        
#         else:
#             array.disconnect_electrode_from_stimulation(el)
#     return stim_els, stim_units, failed_stim_els

def base_setup_saving(s, PATH, ):
    s.set_legacy_format(True)
    s.open_directory(PATH)
    s.start_file("all_channels_recording")
    s.group_delete_all()
    for i in range(26400):
        s.group_define(i, f"electrode_{i:05d}", [i])

# def start_el_saving(s, el_i):
    # s.group_define(0, "one_channel", [chnl])
    # s.group_define(0, "all_channels", list(range(1024)))
    # print(f"Successfully opened file and defined group. Starting recording {dir_name}/{fname}")
    # s.start_recording([el_i])
    
def start_saving(s, PATH, fname):
    # dir_name = f"{PATH}/{el_i//1000:04d}_configs/"
    # fname = f"el_{el_i:05d}_{chnl:04d}"
    
    s.set_legacy_format(True)
    s.open_directory(PATH)
    s.start_file(fname)
    s.group_delete_all()
    # s.group_define(0, "one_channel", [chnl])
    s.group_define(0, "all_channels", list(range(1024)))
    print(f"Successfully opened file and defined group. Starting recording {fname}")
    s.start_recording([0])

def stop_saving(s):
    print("Stopping recording...")
    s.stop_recording()
    s.stop_file()
    s.group_delete_all()

def attampt_connect_el2stim_unit(el, array, used_up_stim_units=[]):
        array.connect_electrode_to_stimulation(el)
        stim_unit = array.query_stimulation_at_electrode(el)
        success = False
        
        # unknown error case, could not find routing?
        if not stim_unit:
            print(f"Warning - Could not connect El{el} to a stim unit.")
            success = False
        
        # stim unit not used yet, 
        elif int(stim_unit) not in used_up_stim_units:
            used_up_stim_units.append(int(stim_unit))
            # print("connected", el, stim_unit)
            success = True
            
            if len(used_up_stim_units) == 32:
                print("Used up all 32 stim units.")
                success = False

        return success, used_up_stim_units














def new_tile_config(n_xtiles, n_ytiles, tile_size, overlap, xstep, ystep):
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
        if ystep > 18:  # 18 for n_tiles = 3
            break
        y_indices.append(indices)
        
    # Initialize the configuration matrix and electrode mapping
    el_matrix_config = np.zeros((120, 220))
    mea1k = np.arange(26400).reshape(120, 220)
    electrodes = {}
    tile_i = 0
    for y_idx in (y_indices):
        for x_idx in (x_indices):
            square_mesh_idx = np.meshgrid(y_idx, x_idx)
            el_matrix_config[square_mesh_idx[0], square_mesh_idx[1]] += 1
            electrodes[tile_i] = mea1k[square_mesh_idx[0], square_mesh_idx[1]]
            tile_i += 1
    
    # # Print the number of electrodes and display the configuration matrix
    print(len(electrodes))
    plt.imshow(el_matrix_config)
    plt.colorbar()
    plt.show()
    return electrodes

def stimulate(PATH):
    s = maxlab.Saving()
    seq = create_stim_sine_sequence()
    
    # Square size: 4, ntiles: 9x3, 27, Overlap: 2, Steps: 11.0x15.0
    tile_size = 4
    overlap = 2
    n_xtiles = 9
    n_ytiles = 3
    
    electrode_mapping = pd.Series([])
    stim_mapping = pd.Series([])
    config_i = 0
    
    while True:
        # Calculate the current step in the x and y directions
        xstep = config_i % 12
        ystep = config_i // 12
        
        # Break the loop if the ystep exceeds the maximum allowed value
        if ystep > 18:  # specific to the current tile configuration, last row
            break
        
        # Generate the new tile configuration
        el_tile_config = new_tile_config(n_xtiles = n_xtiles, 
                                         n_ytiles = n_ytiles, 
                                         tile_size = tile_size, 
                                         overlap = overlap, 
                                         xstep =xstep, 
                                         ystep =ystep)
        
        # el_config = np.array(list(el_tile_config.values())).reshape(30, -1)
        config_name = f"config_{config_i:03d}"
        reset_MEA1K()
        turn_on_stimulation_units(list(range(32)))

        array = setup_array(np.array(list(el_tile_config.values())).flatten(), None, config_name)
        connected_els = [m.electrode for m in array.get_config().mappings]
        connected_chnls = np.array([m.channel for m in array.get_config().mappings])
        # [print(m.electrode, m.channel) for m in array.get_config().mappings]
        # print(el_config)
        # print(el_tile_config)
        
        # update to connected electrodes
        connected_el_tile_config = {}
        for el_tile_name, el_tile in el_tile_config.items():
            # print(el_tile_name, el_tile)
            connected_el_tile = []
            for el_i in el_tile.flatten():
                if el_i in connected_els:
                    connected_el_tile.append(el_i)
            connected_el_tile_config[el_tile_name] = np.array(connected_el_tile)
            # print(connected_el_tile_config[el_tile_name])
            # print()
        el_tile_config = connected_el_tile_config
        # el_config = np.array(list(el_tile_config.values())).reshape(30, -1)
        # print("====")
        # print(el_config)
        start_saving(s, PATH, fname=config_name)
        
        # create a multiindex for the electrode mapping, to identify later
        tuples_midx = [(config_i, el_tile_name, el_i) 
                       for el_tile_name, el_tile in el_tile_config.items() 
                       for el_i in el_tile.flatten()]
        midx = pd.MultiIndex.from_tuples(tuples_midx).set_names(['config', 'tile', 'el'])
        # channels = array.get_config().get_channels_for_electrodes(el_config.flatten())
        el_config_mapping = pd.Series(connected_chnls, index=midx)
        # print(el_tile_config)
        
        electrode_mapping = pd.concat([electrode_mapping, el_config_mapping])
        # print(electrode_mapping.size)
        print(f"\n\n{config_name}... {electrode_mapping.size/(26400*4)*100:.3f}%")
        

        # n_tiles = 30
        # n_el_tile = 16
        # iterate over set of 32-set stimulation electrodes
        stim_indexer = {tile_i: np.arange(els.size) for tile_i,els in el_tile_config.items()}

        stim_counter = 0
        stim_set_i = 0
        while True:
            stim_set_electrodes = []
            used_up_stim_units = []
            for tile_i in range(len(el_tile_config)):
                # print()
                el_left_in_tile = stim_indexer[tile_i].size
                if el_left_in_tile == 0:
                    continue
                stim_el_idx = np.random.choice(np.arange(el_left_in_tile), 1).item()
                # print(stim_el_idx, el_tile_config[tile_i])
                stim_el = el_tile_config[tile_i][stim_indexer[tile_i][stim_el_idx]]
                # print(stim_el)
                # # try to connect here
                success, used_up_stim_units = attampt_connect_el2stim_unit(stim_el, 
                                                                           array, 
                                                                           used_up_stim_units)
                # array.select_stimulation_electrodes(stim_els.flatten())
                if not success:
                    continue
                
                stim_set_electrodes.append(stim_el)
                stim_indexer[tile_i] = np.delete(stim_indexer[tile_i], stim_el_idx)
                stim_counter += 1
                
                # print('tile_i:', tile_i, 'el_left_in_tile:', el_left_in_tile, 'sample:', stim_el)
                # [print(si, len(si[1])) for si in stim_indexer.items()]
            print(stim_set_electrodes)
                
            midx = pd.MultiIndex.from_product([[config_name], [stim_set_i], stim_set_electrodes])
            stim_mapping = pd.concat([stim_mapping, pd.Series(stim_set_electrodes, index=midx)])
            
            array.download() #required
            # time.sleep(3)
                
            print(f"Stimulating ~ ~ ~ ~ ~ ~ ~ ~ set {stim_set_i:02d}, () electrodes ", end="\r")
            seq.send()
            time.sleep(.3)
            # print("Done.")
            
            print(f"Disconnecting {len(stim_set_electrodes)} stimulation electrodes...", end="")
            for stim_el in stim_set_electrodes:
                # print(stim_el, end="...")
                array.disconnect_electrode_from_stimulation(stim_el)
            print("Done.")
            
            # print(stim_counter)
            stim_set_i += 1
            if stim_counter >= len(connected_els):
                break
            
        stop_saving(s)
        print("Config done.")
        config_i += 1
        electrode_mapping.to_csv(f"{PATH}/recording_mapping.csv")
        stim_mapping.to_csv(f"{PATH}/stim_mapping.csv")
        
        
                
            
            
        
        
    #     for stim_row_i in range(6):
    #         for stim_col_i in range(6):
    #             stim_set_name = f"stim_set_{stim_row_i*6+stim_col_i:02d}"
    #             stim_els = el_matrix_config[stim_row_i::6,stim_col_i::6]
    #             print(f"{stim_set_name}: Connect {stim_els.size} electrodes to stim DAC:\n",)# stim_el)
                
    #             midx = pd.MultiIndex.from_product([[config_name], [stim_set_name], range(stim_els.size)])
    #             stim_mapping = pd.concat([stim_mapping, pd.Series(stim_els.flatten(), index=midx)])
                
    #             array.select_stimulation_electrodes(stim_els.flatten())
    #             stim_els, stim_units, failed_stim_els = connect_el2stim_units(array, stim_el)
    #             print(f"Failed to connect these electrodes: {failed_stim_els}")
    #             print(f"Downloading (actually connecting) + 1s sleep")
    #             array.download() #required
    #             time.sleep(post_connection_sleep_time)
                
    #             channels = array.get_config().get_channels_for_electrodes(stim_els)
    #             channel_el_stimunit_map = np.array([channels,stim_els, stim_units]).T                
            
    #             print("Stimulating ~ ~ ~ ~ ~ ~ ~ ~ ", end="")
    #             stim_seq.send()
    #             print("Done.")
                
    #             time.sleep(.1)
    #             print(f"Disconnecting {len(stim_els)} stimulation electrodes...", end="")
    #             for stim_el in stim_els:
    #                 print(stim_el, end="...")
    #                 array.disconnect_electrode_from_stimulation(stim_el)
    #             print("Done.")
            
    #             print(stim_mapping)
    #             el_matrix_config_debug = el_matrix_config.copy()
    #             el_matrix_config_debug[stim_row_i::6,stim_col_i::6] = -1
    #             plt.figure()
    #             plt.imshow(el_matrix_config_debug, vmin=el_matrix_config_debug.min()-200, vmax=el_matrix_config_debug.max())
    #             plt.show()
        
    #     # update the current configuration, move to next square/rectangle at edges
    #     cur_config_fromX += sqaure_size - overlap
    #     if cur_config_fromX >= 220:
    #         cur_config_fromX = 0
    #         cur_config_fromY += sqaure_size - overlap
    #         if cur_config_fromY >= 120:
    #             break            
        
    #     # stop_saving(s)
    #     config_i += 1
    #     plt.imshow(el_matrix_config, vmin=0, vmax=26400)
    #     if config_i > 3:
    #         plt.show()
        
    # # print(electrode_mapping)
    # # print(stim_mapping)
    # # print(stim_mapping.sort_values())
    # return electrode_mapping, stim_mapping
    
if __name__ == "__main__":
    PATH = "/mnt/SpatialSequenceLearning/Simon/impedance/device_headmount_new2EpoxyWalls/impedance_bonded_neighbours2/"
    log2file = True
    post_connection_sleep_time = .6
    if log2file:
        logfile = open(f"{PATH}/mxstimpy.log", "w")
        sys.stdout = logfile
    
    # stim_seq = create_stim_sequence()
    stim_seq = None
    
    sqaure_size = 30
    overlap = 3
    
    electrode_mapping, stim_mapping =  stimulate(PATH, )
    # save the mappings
    # electrode_mapping.to_csv(f"{PATH}/recording_mapping.csv")
    # stim_mapping.to_csv(f"{PATH}/stim_mapping.csv")