import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert
from multiprocessing import Pool, cpu_count
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom

from ephys_constants import SAMPLING_RATE, MAX_AMPL_mV, ADC_RESOLUTION


def filter_trace(signal, sampling_rate, lowcut=None, highcut=None, order=4, btype='band'):
    nyquist = 0.5 * sampling_rate  # Nyquist frequency 
    if highcut is not None:
        high = highcut / nyquist
    if lowcut is not None:
        low = lowcut / nyquist
    
    if btype == 'band':    
        b, a = butter(order, [low, high], btype=btype)
    elif btype == 'low':
        b, a = butter(order, low, btype=btype)
    elif btype == 'high':
        b, a = butter(order, high, btype=btype)
    
    # Apply the filter to the signal using filtfilt for zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def extract_average_amplitude(signal):
    # Compute the analytic signal using the Hilbert transform
    analytic_signal = hilbert(signal)
    # Compute the envelope (magnitude of the analytic signal)
    amplitude_envelope = np.abs(analytic_signal)
    # Compute the average amplitude of the envelope
    average_amplitude = np.mean(amplitude_envelope)
    return average_amplitude

def estimate_frequency_power(signal, min_band, max_band, agg=np.mean):
    # Compute the FFT of the signal
    fft_result = np.fft.fft(signal)
    # Compute the power spectrum
    power_spectrum = np.abs(fft_result) ** 2
    # Compute the corresponding frequencies
    freqs = np.fft.fftfreq(len(signal), 1 / SAMPLING_RATE)
    # Only keep the positive frequencies
    positive_freqs = freqs[freqs >= 1]
    positive_power_spectrum = power_spectrum[freqs >= 1]

    min_band = min_band if min_band is not None else 0
    max_band = max_band if max_band is not None else np.inf
    within_band_power = positive_power_spectrum[(positive_freqs > min_band) & 
                                                (positive_freqs < max_band)]
    agg_band_power = agg(within_band_power)
    return agg_band_power, positive_freqs, positive_power_spectrum

def get_recording_gain(path, fname):
    fmt = get_recording_version(path, fname)
    if fmt == 'legacy':
        with h5py.File(os.path.join(path, fname), 'r') as file:
            gain = file['settings']['gain'][:][0].item()
    elif fmt in ('routed', 'all_channels'):
        with h5py.File(os.path.join(path, fname), 'r') as file:
            gain = file['data_store/data0000/settings/gain'][:][0].item()
    print(f"Recording gain: {gain}")  
    return gain

def get_recording_sampling_rate(path, fname):
    fmt = get_recording_version(path, fname)
    if fmt == 'legacy':
        with h5py.File(os.path.join(path, fname), 'r') as file:
            sr = file['settings']['sampling_rate'][:][0].item()
    elif fmt == 'routed':
        with h5py.File(os.path.join(path, fname), 'r') as file:
            sr = file['data_store/data0000/settings/sampling_rate'][:][0].item()
    elif fmt == 'all_channels':
        with h5py.File(os.path.join(path, fname), 'r') as file:
            sr = file['data_store/data0000/settings/sampling_rate'][:][0].item()
    print(f"Recording sampling rate: {sr}")
    return sr

def get_recording_resolution(gain):
    return (MAX_AMPL_mV/ADC_RESOLUTION) /gain

def convert_to_vol(data, path, fname):
    # scale data to mv
    gain = get_recording_gain(path, fname)
    # print(data)
    data = data.astype(np.float32)
    # print(data)
    data = (data-ADC_RESOLUTION/2) *get_recording_resolution(gain)
    # print(data)
    # floating/ offset
    data += MAX_AMPL_mV/2
    # print(data) 
    return data

# def process_data_chunk(data_chunk, resolution, shift_to_real_potential):
#     # Perform in-place operations to save memory
#     np.subtract(data_chunk, ADC_RESOLUTION / 2, out=data_chunk)
#     np.multiply(data_chunk, resolution, out=data_chunk)
#     if shift_to_real_potential:
#         np.add(data_chunk, MAX_AMPL_mV / 2, out=data_chunk)
#     np.multiply(data_chunk, 1000, out=data_chunk)
#     np.rint(data_chunk, out=data_chunk)
#     # data_chunk -= data_chunk[:,0][:, np.newaxis]
#     # assert np.max(data_chunk) < 2**15 and np.min(data_chunk) > -2**15, "Data is not in int16 range"
#     # data_chunk = data_chunk.astype(np.int16)
    
#     return data_chunk


# def convert_to_vol_chunk(data_chunk, resolution, shift_to_real_potential):
#     # Perform in-place operations to save memory
#     np.subtract(data_chunk, ADC_RESOLUTION / 2, out=data_chunk)
#     np.multiply(data_chunk, resolution, out=data_chunk)
#     if shift_to_real_potential:
#         np.add(data_chunk, MAX_AMPL_mV / 2, out=data_chunk)
#     return data_chunk

# def convert_to_vol(data, path, fname, shift_to_real_potential=True):
#     # scale data to mv
#     gain = get_recording_gain(path, fname)
#     resolution = get_recording_resolution(gain)
    
#     # Convert data to float16 to save space
#     if data.dtype != np.float16:
#         data = data.astype(np.float16, copy=False)
    
#     # Determine the number of chunks and the chunk size
#     num_chunks = cpu_count()
#     chunk_size = len(data) // num_chunks
    
#     # Create a list of data chunks
#     data_chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    
#     # Create a pool of workers
#     with Pool(processes=num_chunks) as pool:
#         # Process each chunk in parallel
#         results = pool.starmap(process_data_chunk, [(chunk, resolution, shift_to_real_potential) for chunk in data_chunks])
    
#     # Combine the results back into a single array
#     data = np.concatenate(results)
    
#     return data

# def convert_to_vol(data, path, fname, shift_to_real_potential=True):
#     # scale data to mv
#     gain = get_recording_gain(path, fname)
#     resolution = get_recording_resolution(gain)
#     # Convert data to float16 to save space
#     if data.dtype != np.float16:
#         data = data.astype(np.float16, copy=False)
    
#     # Perform in-place operations to save memory
#     np.subtract(data, ADC_RESOLUTION / 2, out=data)
#     np.multiply(data, resolution, out=data)
#     if shift_to_real_potential:
#         np.add(data, MAX_AMPL_mV / 2, out=data)
#     return data

def get_recording_version(path, fname):
    with h5py.File(os.path.join(path, fname), 'r') as file:
        if 'sig' in file:
            fmt = 'legacy'
        elif 'data_store/data0000/groups/routed/raw' in file:
            fmt = 'routed'
        elif 'data_store/data0000/groups/all_channels/raw' in file:
            fmt = 'all_channels'
    return fmt

def get_config_rec_fname(path, config_i):
    fname = f"config_{config_i:03d}.raw.h5"
    if fname not in os.listdir(path):
        fname = f'config_set_{config_i:02d}_0.raw.h5'
        if fname not in os.listdir(path):
            fname = f'config_set_{config_i:05}.raw.h5'
            if fname not in os.listdir(path):
                fname = f'el_config_{config_i:03}.raw.h5'
                if fname not in os.listdir(path):
                    print(f"Could not find .raw.h5 recording file for config_i:{config_i}")
                    exit(1)
                
def read_raw_data(path, fname, convert2vol=False, to_df=True, subtract_dc_offset=False,
                  convert2uVInt=False, row_slice=slice(None), col_slice=slice(None)):
    rec_file_fmt = get_recording_version(path, fname)
    if rec_file_fmt == 'legacy':
        data_key = 'sig'
    elif rec_file_fmt == 'routed':
        data_key = 'data_store/data0000/groups/routed/raw'
    elif rec_file_fmt == 'all_channels':
        data_key = 'data_store/data0000/groups/all_channels/raw'
        
    print(f"Reading data {fname} in {path} with format "
          f"`{rec_file_fmt}`", flush=True, end='... ')
    
    start_time = time.time()
    with h5py.File(os.path.join(path, fname), 'r') as file:
        # raw_data = np.array(file[data_key][row_slice, col_slice]).astype(np.float16)
        raw_data = np.array(file[data_key][row_slice, col_slice]).astype(np.int16)
        # raw_data = raw_data.astype(np.float16)
        print(f"Done: {raw_data.shape}", flush=True)
          
    end_time = time.time()
    # print(f"Decompressing time: {end_time - start_time} seconds")
    
    start_time = time.time()
    if convert2vol:
        raw_data = convert_to_vol(raw_data, path, fname, )
    #     if convert2uVInt:
    #         # raw_data = (raw_data*1000).round() #.astype(np.int32) # max ampl 3.3V -> 3.3 million uV -> int32
    #         np.multiply(raw_data, 1000, out=raw_data)
    #         # round inplace
    #         np.rint(raw_data, out=raw_data)
    #         print(raw_data)
    if subtract_dc_offset:
        raw_data -= raw_data[:,0][:, np.newaxis]
        if convert2uVInt and convert2vol:
            assert np.max(raw_data) < 2**15 and np.min(raw_data) > -2**15, "Data is not in int16 range"
            raw_data = raw_data.a   #     if convert2uVInt:
    #         # raw_data = (raw_data*1000).round() #.astype(np.int32) # max ampl 3.3V -> 3.3 million uV -> int32
    #         np.multiply(raw_data, 1000, out=raw_data)
    #         # round inplace
    #         np.rint(raw_data, out=raw_data)
    #         print(raw_data)stype(np.int16)
    end_time = time.time()
    # print(f"Processing time: {end_time - start_time} seconds")
    if to_df:
        raw_data_mapping, _ = get_recording_mapping(path, fname)
        raw_data = pd.DataFrame(raw_data).reindex(raw_data_mapping.values)
        raw_data.index = raw_data_mapping.index
        raw_data.sort_index(inplace=True)
        raw_data.index.name = 'el'

        if rec_file_fmt != 'legacy':
            index_key = data_key.replace('raw', 'frame_nos')
            with h5py.File(os.path.join(path, fname), 'r') as file:
                frame_nos = np.array(file[index_key][col_slice])*(1_000_000//SAMPLING_RATE)
                frame_nos -= frame_nos[0]
        else:
            frame_nos = np.arange(raw_data.shape[1]) *(1_000_000//SAMPLING_RATE)
        raw_data.columns = frame_nos
        
    return raw_data

def get_recording_mapping(path, fname):
    rec_file_fmt = get_recording_version(path, fname)
    if rec_file_fmt == 'legacy':
        mapping_key = 'mapping'
    elif rec_file_fmt in ('routed', 'all_channels'):
        mapping_key = 'data_store/data0000/settings/mapping'
    with h5py.File(os.path.join(path, fname), 'r') as file:
        mapping = np.array([list(m) for m in np.array(file[mapping_key])])
    if mapping.shape[0] == 0:
        print("Mapping is empty, using default bonded mapping")
        mapping = pd.read_csv("assets/default_bonded_mapping.csv", index_col=0).iloc[:,0]
        print(mapping)
        return mapping, None
        
    channels = mapping[:, :2].astype(int)
    channels = pd.Series(channels[:,0], index=channels[:,1]).sort_values()
    el_xy = mapping[:, 2:4]
    return channels, el_xy
    
def get_implant_mapping(nas_dir, device_name):
    fullfname = os.path.join(nas_dir, 'implant_devices', device_name, 
                            #  'bonding_electrode_map.csv')
                             f'bonding_electrode_map_46pad4shank.csv')
    bonding_electrode_map = pd.read_csv(fullfname, index_col=0)
    return bonding_electrode_map

def create_neuroscope_xml_from_template(template_filename, output_filename, 
                                        channel_groups, channel_colors_dict):
    # Parse the template XML file
    tree = ET.parse(template_filename)
    root = tree.getroot()
    
    # Locate relevant elements to insert channel groups and channels
    channel_groups_elem = root.find(".//anatomicalDescription/channelGroups")
    channels_elem = root.find(".//neuroscope/channels")
    
    # Calculate the total number of channels
    total_channels = sum(len(group) for group in channel_groups)
    
    # Update <nChannels> in the <acquisitionSystem> section
    n_channels_elem = root.find(".//acquisitionSystem/nChannels")
    if n_channels_elem is not None:
        n_channels_elem.text = str(total_channels)
    
    # Insert channel groups
    for group in channel_groups:
        group_elem = ET.SubElement(channel_groups_elem, 'group')
        for channel in group:
            channel_elem = ET.SubElement(group_elem, 'channel', skip="0")
            channel_elem.text = str(channel)
    
    # Insert channels with specified colors
    for channel in range(total_channels):
        # Get color settings for the channel from the dictionary, with default values if not provided
        colors = channel_colors_dict.get(channel, {
            'color': '#0080ff',
            'anatomyColor': '#0080ff',
            'spikeColor': '#0080ff'
        })
        
        # Add <channelColors> with all sub-elements
        channel_colors = ET.SubElement(channels_elem, 'channelColors')
        ET.SubElement(channel_colors, 'channel').text = str(channel)
        ET.SubElement(channel_colors, 'color').text = colors['color']
        ET.SubElement(channel_colors, 'anatomyColor').text = colors['anatomyColor']
        ET.SubElement(channel_colors, 'spikeColor').text = colors['spikeColor']
        
        # Add <channelOffset> with all sub-elements
        channel_offset = ET.SubElement(channels_elem, 'channelOffset')
        ET.SubElement(channel_offset, 'channel').text = str(channel)
        ET.SubElement(channel_offset, 'defaultOffset').text = "0"
    
    # Pretty print and write to file
    xml_str = ET.tostring(root, encoding='utf-8')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent=" ")
    
    with open(output_filename, "w") as f:
        f.write(pretty_xml)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def second_recording():
    # # PATH = '/Users/loaloa/local_data/2024-10-21_16-08_rYL006_P0500_MotorLearning_3min'
    # PATH = '/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Other/mea1k_first'
    # fname = 'ephys_output.raw.h5'
    # data = read_raw_data(PATH, fname, convert2vol=True, col_slice=slice(0, 1000))
    # mapping, _ = get_recording_mapping(PATH, fname)
    
    # # # routed
    # PATH = '/Users/loaloa/local_data'
    # fname = 'Trace_20241025_15_35_38.raw.h5'
    # data = read_raw_data(PATH, fname, convert2vol=True, col_slice=slice(0, 1000))
    # mapping, _ = get_recording_mapping(PATH, fname)
    
    # # all chaannels
    PATH = '/mnt/SpatialSequenceLearning/RUN_rYL006/rYL006_P1100/2024-11-15_15-48_rYL006_P1100_LinearTrackStop_35min/'
    fname = 'ephys_output.raw.h5'
    data = read_raw_data(PATH, fname, convert2vol=True, col_slice=slice(0, 1000))
    mapping, _ = get_recording_mapping(PATH, fname)

def main():
    second_recording()

if __name__ == "__main__":
    main()
    
    
    