import h5py
import os
import time

import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET
from xml.dom import minidom

from ephys_constants import SAMPLING_RATE, MAX_AMPL_mV, ADC_RESOLUTION, device_paths
from CustomLogger import CustomLogger as Logger

def _get_recording_gain(path, fname):
    fmt = _get_recording_version(path, fname)
    if fmt == 'legacy':
        with h5py.File(os.path.join(path, fname), 'r') as file:
            gain = file['settings']['gain'][:][0].item()
    elif fmt in ('routed', 'all_channels'):
        with h5py.File(os.path.join(path, fname), 'r') as file:
            gain = file['data_store/data0000/settings/gain'][:][0].item()
    return gain

def _get_recording_resolution(gain):
    return (MAX_AMPL_mV/ADC_RESOLUTION) /gain

def _get_recording_version(path, fname):
    with h5py.File(os.path.join(path, fname), 'r') as file:
        if 'sig' in file:
            fmt = 'legacy'
        elif 'data_store/data0000/groups/routed/raw' in file:
            fmt = 'routed'
        elif 'data_store/data0000/groups/all_channels/raw' in file:
            fmt = 'all_channels'
    return fmt

    
    

        

def _get_recording_config(path, fname):
    rec_file_fmt = _get_recording_version(path, fname)
    if rec_file_fmt == 'legacy':
        mapping_key = 'mapping'
    elif rec_file_fmt in ('routed', 'all_channels'):
        mapping_key = 'data_store/data0000/settings/mapping'
    with h5py.File(os.path.join(path, fname), 'r') as file:
        mapping = np.array([list(m) for m in np.array(file[mapping_key])])
    if mapping.shape[0] == 0:
        Logger().logger.warning("Mapping is empty in h5, inferring mapping from path...")
        
        # infer the mapping from the path
        session_name = os.path.basename(path)
        date, _, animal_name = session_name.split('_')[:3]
        implant_name = animal_name2implant_device(animal_name)
        config_fullfname = _get_implant_config_fullfname(implant_name, animal_name, date)
        mapping = pd.read_csv(config_fullfname.replace(".cfg", '.csv'), 
                              index_col=None).values
        
    channels = mapping[:, :2].astype(int)
    channels = pd.Series(channels[:,0], index=channels[:,1]).sort_values()
    el_xy = mapping[:, 2:4]
    return channels, el_xy

def _ADC2voltage(data, gain, subtract_dc_offset):
    L = Logger()
    resolution = _get_recording_resolution(gain)
    dtype = np.int16 if gain in (1024,512,112) else np.int64
    max_ampl_uV = int(resolution*ADC_RESOLUTION*1000)
    
    L.logger.debug(f"Converting ADC values to mV ({resolution:.4f}mV/ "
                   f"adc step ({ADC_RESOLUTION})) -> range ±{max_ampl_uV:,}"
                   f"uV, casting to {dtype}...")

    # Perform in-place operations to save memory
    np.subtract(data, ADC_RESOLUTION / 2, out=data)
    np.multiply(data, resolution, out=data)
    np.multiply(data, 1000, out=data)
    np.rint(data, out=data)
    # check overflow before casting
    if np.abs(data).max() > max_ampl_uV:
        L.logger.warning(f"Overflow detected in data, casting to {dtype}...")
    data = data.astype(dtype)
    
    if L.logger.level == 10: # DEBUG
        clipped_traces = np.any(np.abs(data)==max_ampl_uV//2, axis=1)
        if np.any(clipped_traces):
            L.logger.warning(f"Clipping detected in {clipped_traces.sum()}/"
                             f"{clipped_traces.size} traces.")

    # if it's passed as a boolean, subtract the first sample from all samples
    if subtract_dc_offset is True:
        data -= data[:,0][:, np.newaxis]
    # one can pass a specific array to subtract the DC offset (when using chunks)
    elif isinstance(subtract_dc_offset, np.ndarray):
        data -= subtract_dc_offset

    L.logger.debug(f"Done.")
    return data
    
def _get_data_key(rec_file_fmt):
    if rec_file_fmt == 'legacy':
        data_key = 'sig'
    elif rec_file_fmt == 'routed':
        data_key = 'data_store/data0000/groups/routed/raw'
    elif rec_file_fmt == 'all_channels':
        data_key = 'data_store/data0000/groups/all_channels/raw'
    return data_key

def _get_recording_shape(path, fname):
    rec_file_fmt = _get_recording_version(path, fname)
    data_key = _get_data_key(rec_file_fmt)
    with h5py.File(os.path.join(path, fname), 'r') as file:
        shape = file[data_key].shape
    return shape

def _read_mea1k_file(path, fname, dtype=np.float16, row_slice=slice(None), 
                    col_slice=slice(None)):
    rec_file_fmt = _get_recording_version(path, fname)
    data_key = _get_data_key(rec_file_fmt)
    Logger().logger.debug(f"Reading MEA1K ephys data (format: {rec_file_fmt}) "
                          f"casting to `{dtype}` with col slice {col_slice}, "
                          f"row slice {row_slice}...")

    # read the MEA1K file
    start_time = time.time()
    with h5py.File(os.path.join(path, fname), 'r') as file:
        raw_data = np.array(file[data_key][row_slice, col_slice], dtype=dtype)
    Logger().logger.debug(f"Done. Decompressed {raw_data.shape} in "
                          f"{time.time() - start_time:.2f} seconds")
    return raw_data

def read_stim_DAC(path, fname, col_slice=slice(None)):
    dac_data = _read_mea1k_file(path, fname, row_slice=1024, 
                                col_slice=col_slice, dtype=np.int16)
    return dac_data

def read_raw_data(path, fname, convert2uV,
                  subtract_dc_offset=False, to_df=False, 
                  row_slice=slice(None), col_slice=slice(None)):
    dtype = np.float16 if convert2uV else np.int16
    raw_data = _read_mea1k_file(path, fname, row_slice=row_slice, 
                                col_slice=col_slice, dtype=dtype)
    
    if not isinstance(row_slice, pd.Index):
        # if data is recorded with format `all_channels`, we use mapping to subset the amplifers
        raw_data_mapping, _ = _get_recording_config(path, fname)
        raw_data = raw_data[raw_data_mapping.values]
    
    if convert2uV:
        gain = _get_recording_gain(path, fname)
        raw_data = _ADC2voltage(raw_data, gain, subtract_dc_offset=subtract_dc_offset,)
    elif subtract_dc_offset:
        raw_data -= raw_data[:,0][:, np.newaxis]
    
    Logger().logger.debug(f"Data:\n{raw_data if raw_data.shape[1]>1 else raw_data.T}\n{raw_data.shape}")
    
    if to_df and not isinstance(row_slice, pd.Index):
        raw_data_mapping, _ = _get_recording_config(path, fname)
        # should already be in this order
        raw_data = pd.DataFrame(raw_data, index=raw_data_mapping.index)
        raw_data.index.name = 'el'

        # if rec_file_fmt != 'legacy':
        #     index_key = data_key.replace('raw', 'frame_nos')
        #     with h5py.File(os.path.join(path, fname), 'r') as file:
        #         frame_nos = np.array(file[index_key][col_slice])*(1_000_000//SAMPLING_RATE)
        #         frame_nos -= frame_nos[0]
        # else:
        #     frame_nos = np.arange(raw_data.shape[1]) *(1_000_000//SAMPLING_RATE)
        # raw_data.columns = frame_nos
    return raw_data

def mea1k_raw2decompressed_dat_file(path, fname, session_name, animal_name, 
                                    chunk_size_s=60, convert2uV_int16=False, 
                                    convert2mV_float16=False, subtract_dc_offset=False, 
                                    exclude_shanks=[]):
    L = Logger()
    
    try:
        _, rec_length = _get_recording_shape(path, fname)
    except OSError as e:
        L.logger.error(f"Error: {e}")
        return
    chunk_indices = np.arange(0, rec_length, chunk_size_s*SAMPLING_RATE)
    chunk_indices = np.append(chunk_indices, rec_length)
    L.logger.info(f"Decompressing {rec_length:,} samples in "
                   f"{len(chunk_indices)} x {chunk_size_s}s chunks.")
    L.spacer("debug")
    
    # get bonding mapping (mea1k el -> polyimide pad / shank electrode)
    # and match with the mea1k el of the recording file
    implant_mapping = get_recording_implant_mapping(path, fname, animal_name=animal_name, 
                                                    exclude_shanks=exclude_shanks)

    # this can be used to reorder the raw data according to physical layout
    shank_order = implant_mapping[['shank_id', 'depth']].values.argsort(axis=0)[:, 0]
    
    # create the output file that we will append to
    out_fullfname = os.path.join(path,f"{session_name}_{len(implant_mapping)}_ephys_traces.dat")
    open(out_fullfname, 'w').close()
    # save the mapping file
    implant_mapping.iloc[shank_order].to_csv(out_fullfname.replace(".dat", "_mapping.csv"))
    
    # calculate the initial DC offset, use for all later chunks
    if subtract_dc_offset:
        subtract_dc_offset = read_raw_data(path, fname, convert2uV_int16=convert2uV_int16,
                                           convert2mV_float16=convert2mV_float16,
                                           col_slice=slice(0,1), row_slice=implant_mapping.index)
    # iterate of time chunks of the data (all channels) and process them
    for i in range(len(chunk_indices)-1):
        col_slice = slice(chunk_indices[i], chunk_indices[i+1])
        data_chunk = read_raw_data(path, fname, convert2uV_int16=convert2uV_int16,
                                   convert2mV_float16=convert2mV_float16,
                                   subtract_dc_offset=subtract_dc_offset,
                                   col_slice=col_slice,
                                   row_slice=implant_mapping.index)
        # reorder to sort by shank and then depth of electrode on that shank
        data = data_chunk[shank_order]
        
        if i == 0 and L.LOGGING_LEVEL == 'DEBUG':
            import matplotlib.pyplot as plt
            # plot traces
            plt.figure()
            for j,row in enumerate(data):
                if j<300:
                    continue
                plt.plot(row[:20_000]+j*1000)
            plt.show()

        # write to file
        with open(out_fullfname, 'ab') as f:
            L.logger.debug(f"Writing {data_chunk.shape} chunk {i+1}/{len(chunk_indices)-1} to file...")
            data_chunk.flatten(order="F").tofile(f)
            L.spacer("debug")
            
def get_recording_implant_mapping(path, mea1k_rec_fname, animal_name=None,
                                  implant_name=None, exclude_shanks=None, 
                                  drop_non_bonded=True):
    L = Logger()
    rec_config, _ = _get_recording_config(path, mea1k_rec_fname)
    rec_mea1k_els = rec_config.index.values
    rec_mea1k_chnls = rec_config.values
    L.logger.debug(f"Recording config:\n{rec_config}")
    
    # infer what implant was used, either from animal or through name itself
    implant_mapping = _get_raw_implant_mapping(animal_name=animal_name,
                                               implant_name=implant_name)
    # get the general mapping from NAS, then reindex this table to fit the recording
    implant_mapping = implant_mapping.set_index('mea1k_el').reindex(rec_mea1k_els)
    implant_mapping.reset_index(inplace=True)
    implant_mapping.index = rec_mea1k_chnls
    implant_mapping.index.name = 'amplifier_id'
    # copying and subseting the implant mapping for the specific config used in this rec
    # implant_mapping.to_csv(os.path.join(path, f"{session_name}_{n_sites}_ephys_traces_mapping.csv"))
    Logger().logger.debug(f"Implant mapping:\n{implant_mapping}")

    # check if all electrodes are bonded    
    n_per_shank_els = implant_mapping.shank_id.value_counts().sort_index()
    n_per_shank_els['no_shank_connection'] = implant_mapping.shank_id.isna().sum()
    L.logger.debug(L.fmtmsg([f"Recorded from {implant_mapping.shape[0]} MEA1K "
                              "electrodes that were bonded to these shanks:",
                              n_per_shank_els.to_dict()]))
    if n_per_shank_els['no_shank_connection'] > 0:
        L.logger.warning(f"Recorded from {n_per_shank_els['no_shank_connection']}"
                         " MEA1K electrodes that were not bonded/ connected to a "
                         "shank.")
        if drop_non_bonded:
            L.logger.warning("Dropping non-bonded electrodes from the mapping")
            implant_mapping = implant_mapping[implant_mapping.shank_id.notna()]
    
    if exclude_shanks is not None and len(exclude_shanks) > 0:
        L.logger.debug(f"Excluding MEA1K electrodes bonded to shanks "
                       f"{exclude_shanks} from the recording...")
        implant_mapping = implant_mapping[~implant_mapping.shank_id.isin(exclude_shanks)]

    Logger().logger.debug(f"Implant mapping after:\n{implant_mapping}")
    return implant_mapping
    
def _get_raw_implant_mapping(implant_name=None, animal_name=None):
    nas_dir = device_paths()[0]
    if animal_name is not None:
        implant_name = animal_name2implant_device(animal_name)
    if implant_name is not None:
        fullfname = os.path.join(nas_dir, 'devices', 'implant_devices', implant_name, 
                                'bonding', f'bonding_mapping_{implant_name}.csv')
    else:
        raise ValueError("Either `implant_name` or `animal_name` must be provided")
    return pd.read_csv(fullfname, index_col=None)

def animal_name2implant_device(animal_name):
    nas_dir = device_paths()[0]
    fullfname = os.path.join(nas_dir, 'devices', 'implant_to_animal_map.csv')
    mapping = pd.read_csv(fullfname, index_col=0, header=0)
    Logger().logger.debug(f"Animal->Implant map:\n{mapping}")
    if animal_name in mapping.index:
        return mapping.loc[animal_name].item()
    else:
        raise ValueError(f"No implant name found for `{animal_name}` Add to the "
                         f"mapping manually: {fullfname}")
    
def _get_implant_config_fullfname(implant_name, animal_name, date):
    nas_dir = device_paths()[0]
    path = os.path.join(nas_dir, 'devices', 'implant_devices', implant_name, 'bonding')
    animal_config = [f for f in os.listdir(path) 
                     if f.startswith(animal_name) and f.endswith('.cfg')]
    Logger().logger.info(f"Implant configurations found for {implant_name}, "
                          f"{animal_name}: {animal_config}")
    if len(animal_config) > 1:
        Logger().logger.warning(f"Multiple implant configurations found for "
                                f"{animal_name}. Should use {date}. For now "
                                "simply most recent.")
    elif len(animal_config) == 0:
        raise ValueError(f"No implant configuration found for {animal_name}")
    return os.path.join(path, animal_config[-1])
    
    
    
    
    
    
    
    
# TODO needs to be updated    

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

def write_probe_file(subdir, fname, pad_size=11, shanks=[1.,2.]):
    data = read_raw_data(subdir, fname, convert2vol=True,  convert2uVInt=True,
                         col_slice=slice(0, 1000), subtract_dc_offset=True)
    implant_mapping = _get_raw_implant_mapping(C.NAS_DIR, C.DEVICE_NAME)
    data, implant_mapping = assign_mapping_to_data(data, implant_mapping)
    
    # subset shanks
    implant_mapping = implant_mapping[implant_mapping.shank_id.isin(shanks)]
    print(implant_mapping)
    
    channels = implant_mapping.index +1
    shanks = implant_mapping.shank_id.astype(int)
    
    # Use `x` and `depth` to populate geometry based on actual positions
    geometry = np.zeros((implant_mapping.shape[0], 2))
    geometry[:, 0] = shanks*1000  # x coo constant for each shank
    geometry[:, 1] = implant_mapping.depth # y coo

    # Write to MATLAB-readable probe file format
    fullfname = os.path.join(subdir, f"ephys_{data.shape[0]}_ss.prb")
    print(fullfname)
    with open(fullfname, "w") as f:
        f.write("% Order of the probe sites in the recording file\n")
        f.write(f"channels = {list(channels)};\n\n")
        
        f.write("% Site location in micrometers (x and y)\n")
        f.write("geometry = [\n")
        for row in geometry:
            f.write(f"    {row[0]}, {row[1]};\n")
        f.write("];\n\n")
        
        f.write("% Shank information\n")
        f.write(f"shank = {list(shanks)};\n\n")
        
        # f.write("% Reference sites to exclude\n")
        # f.write(f"ref_sites = {ref_sites};\n\n")
        
        f.write("% Recording contact pad size (height x width in micrometers)\n")
        f.write(f"pad = [{pad_size} {pad_size}];\n\n")






# TODO DEPRECATED FUNCTIONS

def get_implant_mapping_depr(nas_dir, device_name):
    fullfname = os.path.join(nas_dir, 'implant_devices', device_name, 
                            #  'bonding_electrode_map.csv')
                             f'bonding_electrode_map_46pad4shank.csv')
    bonding_electrode_map = pd.read_csv(fullfname, index_col=0)
    return bonding_electrode_map

def assign_mapping_to_data_depr(data, implant_mapping, 
                                shank_depth_side_multiindex=True):
    # mask to the electrodes in the data (routed in rec config)
    implant_mapping = implant_mapping[np.isin(implant_mapping.mea1k_el, data.index)]
    if implant_mapping.pad_id.isna().any():
        print(f"Warning: {implant_mapping.pad_id.isna().sum()} recorded electrodes were not below a pad")
    
    # include only the electrodes that are routed to a shank
    implant_mapping = implant_mapping[implant_mapping.shank_id.notna()]
    # sort the electrodes by shank and depth
    cols = ['shank_id', 'depth', 'shank_side', 'mea1k_el']
    implant_mapping = implant_mapping.sort_values(cols).reset_index(drop=True)
    # reindex the data according to shank and depth
    data = data.loc[implant_mapping.mea1k_el]
    if shank_depth_side_multiindex:
        levels = implant_mapping[cols]
        levels = pd.concat([levels, pd.Series(range(levels.shape[0]),name='channel')], axis=1)
        data.index = pd.MultiIndex.from_frame(levels)
    implant_mapping.index.name = 'channel'
    return data, implant_mapping