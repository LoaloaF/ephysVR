import maxlab
import pandas as pd
import ephys_constants as C
import numpy as np

def reset_MEA1K(gain, enable_stimulation_power=False):
    print(f"Resetting MEA1K with gain of {gain}...", end='', flush=True)
    maxlab.util.initialize()
    if enable_stimulation_power:
        maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
    maxlab.send(maxlab.chip.Amplifier().set_gain(gain))
    print("Done.")

def setup_array(electrodes, stim_electrodes=None, randomize_routing=False):
    print(f"Setting up array with {len(electrodes)} els (reset,route&download)...", 
          end='', flush=True)
    array = maxlab.chip.Array()
    array.reset()
    array.clear_selected_electrodes()
    
    if not randomize_routing:
        array.select_electrodes(electrodes)
        array.connect_all_floating_amplifiers()
        array.connect_amplifier_to_ringnode(0)

    else:
        print("Randomizing routing...", end="", flush=True)
        # split the electrodes into 10 groups
        np.random.shuffle(electrodes)
        el_groups = np.array_split(electrodes, 10)
        for i, el_group in enumerate(el_groups):
            array.select_electrodes(el_group, weight=i+1)

    if stim_electrodes is not None:
        array.select_stimulation_electrodes(stim_electrodes)
    
    array.route()
    array.download()
    # maxlab.offset()
    print("Done.")
    return array

def try_routing(els, return_array=False, stim_electrodes=None, randomize_routing=False):
    array = setup_array(els, stim_electrodes=stim_electrodes, randomize_routing=randomize_routing)
    failed_routing = []
    if stim_electrodes:
        # print(f"Stimulation electrodes: {stim_electrodes}")
        res = [attampt_connect_el2stim_unit(el, array, with_download=True)[0]
               for el in stim_electrodes]
        failed_routing = [el for i, el in enumerate(stim_electrodes) if not res[i]]
        
    succ_routed = [m.electrode for m in array.get_config().mappings]
    failed_routing.extend([el for el in els if el not in succ_routed])
    if failed_routing:
        pass
        # print(f"Failed routing {len(failed_routing)}: {failed_routing}")
    if return_array:
        return succ_routed, failed_routing, array
    array.close()
    return succ_routed, failed_routing

def turn_on_stimulation_units(stim_units, dac_id=0, mode='voltage'):
    print(f"Setting up stim units {len(stim_units)}...", end="", flush=True)
    for stim_unit in stim_units:
        stim = maxlab.chip.StimulationUnit(str(stim_unit))
        stim.power_up(True)
        stim.connect(True)
        if mode == 'voltage':
            stim.set_voltage_mode()
        elif mode == 'small_current':
            stim.set_current_mode()
            stim.set_small_current_range()
        elif mode == 'large_current':
            stim.set_current_mode()
            stim.set_large_current_range()
        stim.dac_source(dac_id)
        maxlab.send(stim)
    print("Done.")

# def create_stim_sequence(dac=0, amplitude=25, npulses=10, nreps=3, inter_pulse_interval=100, rep_delay_s=.1):
#     def append_stimulation_pulse(seq, amplitude):
#         seq.append(maxlab.chip.DAC(0, 512-amplitude))
#         seq.append(maxlab.system.DelaySamples(4))
#         seq.append(maxlab.chip.DAC(0, 512+amplitude))
#         seq.append(maxlab.system.DelaySamples(4))
#         seq.append(maxlab.chip.DAC(0, 512))
#         return seq

#     seq = maxlab.Sequence()config_fullfname
#     for i in range(nreps):
#         for j in range(npulses):
#             append_stimulation_pulse(seq, amplitude) # 25 *2.83mV - current mode?
#             seq.append(maxlab.system.DelaySamples(inter_pulse_interval)) #5ms
#         time.sleep(rep_delay_s)
#     return seq

def create_stim_sine_sequence(dac_id=0, amplitude=25, f=1000, ncycles=100, nreps=1, voltage_conversion=False):
    if voltage_conversion:
        daq_lsb = maxlab.query_DAC_lsb_mV()
        # daq_lsb = maxlab.system.query_DAC_lsb()
        print(f"DAQ LSB: {daq_lsb}")
        amplitude = int(amplitude / daq_lsb)
    
    seq = maxlab.Sequence()
    # Create a time array, 50 us * 20kHz = 1000 samples, 1 khz exmaple
    t = np.linspace(0,1, int(C.SAMPLING_RATE/f))
    # Create a sine wave with a frequency of 1 kHz
    sine_wave = (amplitude * np.sin(t*2*np.pi)).astype(int) +512
    for i in range(nreps):
        for j in range(ncycles):
            for ampl in sine_wave:
                seq.append(maxlab.chip.DAC(dac_id, ampl))
                seq.append(maxlab.system.DelaySamples(1))
    return seq

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

def attampt_connect_el2stim_unit(el, array, used_up_stim_units=[], with_download=False):
    config_before = array_config2df(array)

    used_up_stim_units = []
    array.connect_electrode_to_stimulation(el)
    stim_unit = array.query_stimulation_at_electrode(el)
    success = False
    # print(f"Trying to connect El{el} to stim unit {stim_unit}, used {used_up_stim_units}", flush=True)
    
    # unknown error case, could not find routing?
    if not stim_unit:
        print(f"Warning - Could not connect El{el} to a stim unit.")
        success = False
    
    # stim unit not used yet, 
    elif int(stim_unit) not in used_up_stim_units:
        used_up_stim_units.append(int(stim_unit))
        success = True
        # print("connected", el, stim_unit)
        if with_download:
            array.download()
            if not config_before.equals(array_config2df(array)):
                success = False
            readoutchannel = maxlab.chip.StimulationUnit(stim_unit).get_readout_channel()             
            print(f"Connected El{el} to stim unit {stim_unit} (readout channel {readoutchannel}).")
        
        if len(used_up_stim_units) == 32:
            print("Used up all 32 stim units.")
            success = False
    
    return success, used_up_stim_units





def start_saving(s, dir_name, fname, channels=list(range(1024))):
    s.set_legacy_format(True)
    s.open_directory(dir_name)
    s.start_file(fname)
    s.group_delete_all()
    s.group_define(0, "all_channels", channels)
    print(f"Successfully opened file and defined group. Starting recording {dir_name}/{fname}...")
    s.start_recording([0])

def stop_saving(s):
    print("Stopping recording...")
    s.stop_recording()
    s.stop_file()
    s.group_delete_all()
    
def array_config2df(array):
    rows = [(m.channel, m.electrode, m.x, m.y) for m in array.get_config().mappings]
    config_df = pd.DataFrame(rows, columns=["channel", "electrode", "x", "y"])
    return config_df