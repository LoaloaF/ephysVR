import time
import maxlab
import pandas as pd
import ephys_constants as C
import numpy as np
import math

def reset_MEA1K(gain, enable_stimulation_power=False):
    print(f"Resetting MEA1K with gain of {gain}, then offset...", end='', flush=True)
    maxlab.util.initialize()
    if enable_stimulation_power:
        maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
    maxlab.send(maxlab.chip.Amplifier().set_gain(gain))
    maxlab.offset()
    print("Done.")

def get_maxlab_array():
    return maxlab.chip.Array()

def get_maxlab_saving():
    return maxlab.Saving()

def start_saving(s, dir_name, fname, channels=list(range(1024)), legacy_mode=False):
    if legacy_mode:
        s.set_legacy_format(legacy_mode)
    s.open_directory(dir_name)
    s.start_file(fname)
    s.group_delete_all()
    s.group_define(0, "all_channels", channels)
    print(f"Successfully opened file and defined group. Starting recording {dir_name}/{fname}...")
    s.start_recording([0])

def stop_saving(s):
    print("Stopping recording...", end='', flush=True)
    s.stop_recording()
    s.stop_file()
    s.group_delete_all()
    print("Recording stopped.")
    
def array_config2df(array):
    rows = [(m.channel, m.electrode, m.x, m.y) for m in array.get_config().mappings]
    config_df = pd.DataFrame(rows, columns=["channel", "electrode", "x", "y"])
    return config_df

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
    array = setup_array(els, stim_electrodes=stim_electrodes, 
                        randomize_routing=randomize_routing)
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
        else:
            raise ValueError(f"Unknown mode: {mode}")
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

def attampt_connect_el2stim_unit(el, array, used_up_stim_units=[], 
                                 with_download=False, allow_multiple_connections=False):
    config_before = array_config2df(array)

    # used_up_stim_units = []
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
    elif int(stim_unit) in used_up_stim_units and not allow_multiple_connections:
        readoutchannel = maxlab.chip.StimulationUnit(stim_unit).get_readout_channel()             
        print(f"Warning - Stim unit {stim_unit} already used. Connected to amplifier {readoutchannel}.")
        success = False
    
    return success, used_up_stim_units

def create_stim_sine_sequence(dac_id=0, amplitude=25, f=1000, ncycles=100, 
                              nreps=1, voltage_conversion=False, adjust_offset_for_stimunit=None):
    if voltage_conversion:
        daq_lsb = float(maxlab.query_DAC_lsb_mV())
        print(f"DAQ LSB: {daq_lsb}")
        amplitude = int(amplitude / daq_lsb)
        
    offset = 0
    if adjust_offset_for_stimunit is not None:
        # manually measured for device 4983:
        offset_map = {0: 0, 1: 0, 2: 427, 3: 0, 4: 492, 5: 0, 6: 347, 7: 703, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 895, 14: 0, 15: 0, 16: 959, 17: 634, 18: 0, 19: 0, 20: 325, 21: 1023, 22: 1023, 23: 255, 24: 0, 25: 1023, 26: 1023, 27: 630, 28: 1023, 29: 245, 30: 0, 31: 895}
        offset = offset_map[adjust_offset_for_stimunit]
    
    seq = maxlab.Sequence()
    # Create a time array, 50 us * 20kHz = 1000 samples, 1 khz exmaple
    t = np.linspace(0,1, int(C.SAMPLING_RATE/f))
    # Create a sine wave with a frequency of 1 kHz
    sine_wave = (amplitude * np.sin(t*2*np.pi)).astype(int) +512
    for i in range(nreps):
        for j in range(ncycles):
            for ampl in sine_wave:
                seq.append(maxlab.chip.DAC(dac_id, np.clip(ampl+offset, 0, 1023)))
                seq.append(maxlab.system.DelaySamples(1))
    return seq

def init_fpga_sine_stim(t_period, amp_in_bits, periods=1):
    """Create an FPGA loop on MidSupply"""
    sineStr = ""
    n_samples = 20
    for i in range(0, n_samples):
        v = int(-amp_in_bits * math.sin(periods * 2 * math.pi / n_samples * i))
        s = int(20e3 * t_period / n_samples)
        factor = 128  # 1.65/(3.0/1024)
        sineStr += str(v + factor) + "/" + str(s) + " "
        print(sineStr)
    maxlab.send_raw("system_loop_sine_onVRef " + sineStr)

def begin_fpga_sine_stim():
    maxlab.send(maxlab.system.Switches(sw_0=1, sw_1=0, sw_2=0, sw_3=1, sw_4=0, sw_5=0, sw_6=0, sw_7=0))
    time.sleep(0.5)
    maxlab.send_raw("system_loop_start")

def end_fpga_sine_stim():
    maxlab.send_raw("system_loop_stop")
    maxlab.send(maxlab.system.Switches(sw_0=0, sw_1=0, sw_2=0, sw_3=0, sw_4=0, sw_5=0, sw_6=0, sw_7=0))
    time.sleep(0.5)















#### NOT TESTED YET ##############################

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
            seq.append(maxlab.system.DelaySamples(13+20))

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