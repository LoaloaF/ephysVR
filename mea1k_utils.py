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
        print(f"Stimulation electrodes: {stim_electrodes}")
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
    
def turn_off_stimulation_units(stim_units):
    print(f"Turning off stim units {len(stim_units)}...", end="", flush=True)
    for stim_unit in stim_units:
        stim = maxlab.chip.StimulationUnit(str(stim_unit))
        stim.power_up(False)
        stim.connect(False)
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
        daq_lsb = float(maxlab.query_DAC_lsb_mV())
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

def create_stim_pulse_sequence(dac_id=0, amplitude=25, pulse_duration=167e-6, 
                               inter_phase_interval=67e-6, frequency=50, 
                               burst_duration=400e-3, nreps=1,
                               voltage_conversion=False):
    """
    Create a sequence of biphasic, charge-balanced stimulation pulses.

    Parameters:
        dac_id (int): DAC ID for stimulation.
        amplitude (int): Peak amplitude of the stimulation pulses (in arbitrary units).
        pulse_duration (float): Duration of each phase of the pulse in seconds.
        inter_phase_interval (float): Interval between the cathodic and anodic phases in seconds.
        frequency (float): Frequency of the pulse train in Hz.
        burst_duration (float): Duration of each burst in seconds.
        nreps (int): Number of burst repetitions.

    Returns:
        seq: A sequence object containing the stimulation pulses.
    """
    if voltage_conversion:
        daq_lsb = float(maxlab.query_DAC_lsb_mV())
        # daq_lsb = maxlab.system.query_DAC_lsb()
        print(f"DAQ LSB: {daq_lsb}")
        amplitude = int(amplitude / daq_lsb)
    
    
    seq = maxlab.Sequence()
    
    # Parameters
    frequency = 50  # Hz
    burst_duration = 400e-3  # seconds
    n_pulses = int(burst_duration * frequency)
    burst_interval = 1 / frequency  # seconds
    pulse_duration = 167e-6  # seconds per phase
    inter_phase_interval = 67e-6  # seconds
    sampling_rate = 20000  # Hz (20 kHz)
    samples_per_us = sampling_rate / 1e6

    # print all of thse values
    print(f"n_pulses: {n_pulses}, burst_interval: {burst_interval}, pulse_duration: {pulse_duration}, inter_phase_interval: {inter_phase_interval}, sampling_rate: {sampling_rate}, samples_per_us: {samples_per_us}")
    


    for _ in range(nreps):
        for _ in range(n_pulses):
            # Cathodic phase
            seq.append(maxlab.chip.DAC(dac_id, 512 - amplitude))
            seq.append(maxlab.system.DelaySamples(int(pulse_duration * samples_per_us * 1e6)))
            print(int(pulse_duration * samples_per_us * 1e6))
            # Inter-phase interval
            seq.append(maxlab.chip.DAC(dac_id, 512))  # Zero amplitude for interval
            seq.append(maxlab.system.DelaySamples(int(inter_phase_interval * samples_per_us * 1e6)))
            print(int(inter_phase_interval * samples_per_us * 1e6))
            # Anodic phase
            seq.append(maxlab.chip.DAC(dac_id, 512 + amplitude))
            seq.append(maxlab.system.DelaySamples(int(pulse_duration * samples_per_us * 1e6)))
            print(int(pulse_duration * samples_per_us * 1e6))
            print()
        # Burst interval
        # inter_burst_delay = int((burst_interval - burst_duration) * sampling_rate)
        # print(inter_burst_delay)
        # if inter_burst_delay > 0:
        #     seq.append(maxlab.system.DelaySamples(inter_burst_delay))
            seq.append(maxlab.system.DelaySamples(13+20*20))

    return seq


def create_stim_onoff_sequence(dac_id=0, amplitude=25, pulse_duration=5_000_000, 
                               voltage_conversion=True):
    seq = maxlab.Sequence()
    
    if voltage_conversion:
        daq_lsb = float(maxlab.query_DAC_lsb_mV())
        # daq_lsb = maxlab.system.query_DAC_lsb()
        print(f"DAQ LSB: {daq_lsb}")
        amplitude = int(amplitude / daq_lsb)
        
    seq.append(maxlab.chip.DAC(dac_id, 512 + amplitude))
    seq.append(maxlab.system.DelaySamples(int(pulse_duration/50)))
    seq.append(maxlab.chip.DAC(dac_id, 512))
    seq.append(maxlab.system.DelaySamples(1))
    seq.append(maxlab.chip.DAC(dac_id, 512 - amplitude))
    seq.append(maxlab.system.DelaySamples(int(pulse_duration/50)))
    seq.append(maxlab.chip.DAC(dac_id, 512))
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