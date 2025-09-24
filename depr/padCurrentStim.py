import pandas as pd
import numpy as np
import time
# import matplotlib.pyplot as plt
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
        stim.set_voltage_mode()
        # stim.set_small_current_range()
        # stim.set_large_current_range()
        stim.dac_source(0)
        maxlab.send(stim)
        time.sleep(.1)
    print("Done.")

def create_stim_sine_sequence(dac=0, amplitude=100, f=1000, ncycles=100, nreps=10):
	seq = maxlab.Sequence()
  	
	# 50 us * 20kHz = 1000 samples, 1 khz exmaple
	sampling_rate = 20000  # 20 kHz
	# Create a time array
	t = np.linspace(0,1, int(sampling_rate/f))
	# Create a sine wave with a frequency of 1 kHz
	sine_wave = (amplitude * np.sin(t*2*np.pi)).astype(int) +512
	# debug = []
	for i in range(nreps):
		for j in range(ncycles):
			for ampl in sine_wave:
				seq.append(maxlab.chip.DAC(dac, ampl))
				seq.append(maxlab.system.DelaySamples(1))
                
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
            print("connected", el, stim_unit)
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
    # print(len(electrodes))
    # plt.imshow(el_matrix_config)
    # plt.colorbar()
    # plt.show()
    return electrodes

def stimulate(PATH, PRVPATH):

    pad_mapping = pd.read_pickle(f"{PRVPATH}/final_pads.pkl").sort_values("pad")
    print(pad_mapping)
    connectivity = pd.read_pickle(f"{PRVPATH}/extracted_signal.pkl")
    print(connectivity)

    s = maxlab.Saving()
    seq = create_stim_sine_sequence()
    
    reset_MEA1K()
    turn_on_stimulation_units(list(range(32)))

    array = setup_array(pad_mapping.index.values, None)
    connected_els = [m.electrode for m in array.get_config().mappings]
    print(f"Connected {len(connected_els)}, tried {(pad_mapping.shape[0])}")

    # start_saving(s, PATH, fname="recording")

    all_stimulated = []
    for stim_el in connected_els:
        config_name = f"config_set_{stim_el:05d}"
        
        success, _ = attampt_connect_el2stim_unit(stim_el, array, [])
        if not success:
            continue
        start_saving(s, PATH, fname=config_name)
        
        all_stimulated.append(stim_el)
        # print(stim_set_electrodes)
        # midx = pd.MultiIndex.from_tuples(stim_set_midx).set_names(['config', 'stim_set', 'tile', 'el'])
        # stim_mapping = pd.concat([stim_mapping if stim_mapping is not None else None, 
        #                           pd.Series(stim_set_electrodes, index=midx)])
        
        print("Downloading config....", end='')
        array.download() #required
        print("Done")
            
        print(f"Stimulating ~ ~ ~ ~ ~ ~ ~ ~ el {stim_el}")
        seq.send()
        time.sleep(1.2)
        
        print(f"Disconnecting stimulation electrode...", end='')
        array.disconnect_electrode_from_stimulation(stim_el)
        print("Done.")
        
        array.close()
        stop_saving(s)
        print(f"Config done ({len(all_stimulated)}).\n\n")
        # time.sleep(.5)
    np.save(f"{PATH}/stimulated_order.npy", np.array(all_stimulated, dtype=int))

        # if len(all_stimulated) >5:
        #     break
        
    time.sleep(5)
    array.close()
    stop_saving(s)
    # config_i += 1
    # stim_mapping.to_csv(f"{PATH}/stim_mapping.csv")
    # stim_mapping.to_pickle(f"{PATH}/stim_mapping.pkl")
        
                
if __name__ == "__main__":
    PATH = "/mnt/SpatialSequenceLearning/Simon/impedance/device_headmount_new2EpoxyWalls/impedance_bonded_neighbours4/"
    PATH = "/run/user/1000/gvfs/smb-share:server=yaniklab-data.local,share=large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance/"
    # PATH += "/device_headmount_new3EpoxyWalls/impedance_bonded_try2_rec1/"
    PRVPATH = PATH + "device_headmount_new3EpoxyWalls/impedance_bonded2_D1_1KHz_1024_rec1"
    PATH += "device_headmount_new3EpoxyWalls/impedance_bonded2_D1_singlePadStim_100mV_rec3"
    log2file = False
    post_connection_sleep_time = .6
    if log2file:
        logfile = open(f"{PATH}/mxstimpy.log", "w")
        sys.stdout = logfile
    
    stimulate(PATH, PRVPATH)