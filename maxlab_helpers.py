import maxlab

def reset_MEA1K():
    print("Resetting MEA1K...", end='', flush=True)
    maxlab.util.initialize()
    # maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
    maxlab.send(maxlab.chip.Amplifier().set_gain(112))
    print("Done.")

def setup_array(electrodes, stim_electrodes=None, config_name="default_name"):
    print("Setting up array (reset,route&download)...", end='', flush=True)
    # array = maxlab.chip.Array("offline")
    array = maxlab.chip.Array()
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(electrodes)
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

def base_setup_saving(s, PATH, ):
    s.set_legacy_format(True)
    s.open_directory(PATH)
    s.start_file("all_channels_recording")
    s.group_delete_all()
    for i in range(26400):
        s.group_define(i, f"electrode_{i:05d}", [i])

# def start_el_saving(s, el_i):
#     s.group_define(0, "one_channel", [chnl])
#     s.group_define(0, "all_channels", list(range(1024)))
#     print(f"Successfully opened file and defined group. Starting recording {dir_name}/{fname}")
#     s.start_recording([el_i])
    
def start_saving(s, PATH, el_i, chnl):
    dir_name = f"{PATH}/{el_i//1000:04d}_configs/"
    fname = f"el_{el_i:05d}_{chnl:04d}"
    
    s.set_legacy_format(True)
    s.open_directory(dir_name)
    s.start_file(fname)
    s.group_delete_all()
    # s.group_define(0, "one_channel", [chnl])
    s.group_define(0, "all_channels", list(range(1024)))
    print(f"Successfully opened file and defined group. Starting recording {dir_name}/{fname}")
    s.start_recording([0])
    
    
    
    
    
# def calc_tileparams(sq_size, xntiles, yntiles, overlap, ):
#     x_step_sum = (sq_size*xntiles - (xntiles-1)*overlap)
#     y_step_sum = (sq_size*yntiles - (yntiles-1)*overlap)
#     if y_step_sum == 0 or x_step_sum == 0:
#         return
    
#     xsteps = 220/x_step_sum
#     ysteps = 120/y_step_sum
#     if xsteps % 1 == 0 and ysteps % 1 == 0:
#         print(f"Square size: {sq_size}, ntiles: {xntiles}x{yntiles}, {xntiles*yntiles}, Overlap: {overlap}, Steps: {xsteps}x{ysteps}")
#         # print(f"Square size: {sq_size}, ntiles: {ntiles}, Overlap: {overlap}, Steps: {xsteps}")
    
        
#     # n_tiles = (size - sq_size) / (sq_size - overlap) + 1
#     # if n_tiles % 1 != 0:
#     #     pass
#     #     # print(f"Invalid size: {size}")
#     # else:
#     #     print(f"Square size: {size}")
#     #     print(n_tiles)
        
#     #     mesh = np.zeros((1,size))
#     #     for i in range(0, size, sq_size-overlap):
#     #         print(i, i+sq_size)
#     #         mesh[0,i:i+sq_size] += 1
#     #         # print(mesh)
#     #     plt.imshow(mesh, aspect='auto')
#     #     plt.show()
#     # exit()
# # size = 120
# # calc_tileparams(6, 7, 2, 2, size)
# for sq_size in range(4, 8):
#     for xntiles in range(2, 24):
#         for yntiles in range(2, 24):
#             for overlap in range(2,3):
#                 if xntiles*yntiles <= 32:
#                     calc_tileparams(sq_size, xntiles, yntiles, overlap)
# exit()