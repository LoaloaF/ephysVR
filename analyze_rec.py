import numpy as np
import os
import h5py
import mea1k_ephys as mea1k
import utils
import time
import ephys_constants as EC
from mea1k_ephys import read_raw_data, get_recording_version

from mea1k_viz import validate_shank_map

def check_data(subdir, fname, n_plots=10, n_samples=10_000):
    
    for i in range(10):
        data = mea1k.read_raw_data(subdir, fname, convert2vol=True, to_df=True,
                                   subtract_dc_offset=True, col_slice=slice(i*10_000, (i+1)*10_000))
        implant_mapping = mea1k.get_implant_mapping(EC.NAS_DIR, EC.DEVICE_NAME)
        
        validate_shank_map(data, implant_mapping, shank_id=2, min_depth=6000, 
                           max_depth=8500, scaler=100)
        validate_shank_map(data, implant_mapping, shank_id=1, min_depth=2000, 
                           max_depth=6000, scaler=100)
        
def convert2neuroscope(subdir, fname):
    data = read_raw_data(subdir, fname, convert2vol=True,  convert2uVInt=True,
<<<<<<< Updated upstream
                         col_slice=slice(0, 10_000*60*2), subtract_dc_offset=True)
=======
                         col_slice=slice(0, None), subtract_dc_offset=True)
>>>>>>> Stashed changes
    implant_mapping = mea1k.get_implant_mapping(EC.NAS_DIR, EC.DEVICE_NAME)
    # mask to the electrodes in the data (routed in rec config)
    implant_mapping = implant_mapping[np.isin(implant_mapping.mea1k_el, data.index)]
    if implant_mapping.pad_id.isna().any():
        print(f"Warning: {implant_mapping.pad_id.isna().sum()} recorded electrodes were not below a pad")
    
    # include only the electrodes that are routed to a shank
    implant_mapping = implant_mapping[implant_mapping.shank_id.notna()]
    # sort the electrodes by shank and depth
    implant_mapping = implant_mapping.sort_values(['shank_id', 'depth']).reset_index(drop=True)
    print(implant_mapping)
    # reindex the data according to shank and depth
    data = data.loc[implant_mapping.mea1k_el]
    
    fullfname = os.path.join(subdir, f"ephys_{len(data)}.dat")
    data.values.flatten(order='F').tofile(fullfname)
    
    # convert electrode id to channel id (iloc basically)
    channel_groups = implant_mapping.groupby('shank_id').apply(lambda x: x.index)
    channel_groups = [group.values for group in channel_groups] # unpack

    # color the channels, left side electrodes are slightly different
    channel_colors_dict = {}
    for channel_i, el_row in implant_mapping.iterrows():
        col = EC.SHANK_BASE_COLORs[el_row.shank_id]
        if el_row.shank_side == 'left':
            col = utils.adjust_saturation(col, EC.METALLIZATION_COLOR_OFFSET)
        # convert to html color
        col = '#%02x%02x%02x' % tuple(int(255 * c) for c in col)
        channel_colors_dict[channel_i] = {'color': col, 'anatomyColor': col, 'spikeColor': col}

    # Create XML file using template
    mea1k.create_neuroscope_xml_from_template("assets/neuroscope_template.xml", 
                                              fullfname.replace('.dat', '.xml'),
                                              channel_groups, channel_colors_dict)


def convert2neuroscope_iter(subdir, fname):
    
    rec_file_fmt = get_recording_version(subdir, fname)
    if rec_file_fmt == 'legacy':
        data_key = 'sig'
    elif rec_file_fmt == 'routed':
        data_key = 'data_store/data0000/groups/routed/raw'
    elif rec_file_fmt == 'all_channels':
        data_key = 'data_store/data0000/groups/all_channels/raw'

    with h5py.File(os.path.join(subdir, fname), 'r') as file:
        data_shape = file[data_key].shape    
    
    chunk_size = 10_000*60*2 # 1 minute of data
    implant_isMapped = False
    
    for i in range(0, data_shape[1], chunk_size):
        print("process minute: ", i//chunk_size)
        data_chunk = read_raw_data(subdir, fname, convert2vol=True,  convert2uVInt=True,
                                   col_slice=slice(i, i+chunk_size), subtract_dc_offset=True)
        
        if not implant_isMapped:
            implant_mapping = mea1k.get_implant_mapping(EC.NAS_DIR, EC.DEVICE_NAME)
            # mask to the electrodes in the data (routed in rec config)
            implant_mapping = implant_mapping[np.isin(implant_mapping.mea1k_el, data_chunk.index)]
            if implant_mapping.pad_id.isna().any():
                print(f"Warning: {implant_mapping.pad_id.isna().sum()} recorded electrodes were not below a pad")
            
            # include only the electrodes that are routed to a shank
            implant_mapping = implant_mapping[implant_mapping.shank_id.notna()]
            # sort the electrodes by shank and depth
            implant_mapping = implant_mapping.sort_values(['shank_id', 'depth']).reset_index(drop=True)
            print(implant_mapping)
            implant_isMapped = True
        
        # reindex the data according to shank and depth
        data_chunk = data_chunk.loc[implant_mapping.mea1k_el]
        
        start_time = time.time()
        fullfname = os.path.join(subdir, f"ephys_{len(data_chunk)}.dat")
        with open(fullfname, 'ab') as f:
            data_chunk.values.flatten(order='F').tofile(f)
        
        end_time = time.time()
        print(f"Running time: {end_time - start_time} seconds")
        print("Down")
    
    # convert electrode id to channel id (iloc basically)
    channel_groups = implant_mapping.groupby('shank_id').apply(lambda x: x.index)
    channel_groups = [group.values for group in channel_groups] # unpack

    # color the channels, left side electrodes are slightly different
    channel_colors_dict = {}
    for channel_i, el_row in implant_mapping.iterrows():
        col = EC.SHANK_BASE_COLORS[el_row.shank_id]
        if el_row.shank_side == 'left':
            col = utils.adjust_saturation(col, EC.METALLIZATION_COLOR_OFFSET)
        # convert to html color
        col = '#%02x%02x%02x' % tuple(int(255 * c) for c in col)
        channel_colors_dict[channel_i] = {'color': col, 'anatomyColor': col, 'spikeColor': col}

    # Create XML file using template
    mea1k.create_neuroscope_xml_from_template("assets/neuroscope_template.xml", 
                                              fullfname.replace('.dat', '.xml'),
                                              channel_groups, channel_colors_dict)



def main():
<<<<<<< Updated upstream
    subdir = os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-11-07_16-13_rYL006_P1000_MotorLearningStop_20min/")
    # /Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/
    fname = 'ephys_output.raw.h5'
    # check_data(subdir, fname, n_plots=10, n_samples=10_000)

    convert2neuroscope_iter(subdir, fname)
=======
    subdirs = [
        # "RUN_rYL006/rYL006_P1000/2024-10-25_15-41_rYL006_P1000_MotorLearningStop_14min",
        # "RUN_rYL006/rYL006_P1000/2024-11-04_16-13_rYL006_P1000_MotorLearningStop_22min",
        # "RUN_rYL006/rYL006_P1000/2024-11-05_16-11_rYL006_P1000_MotorLearningStop_22min",
        # "RUN_rYL006/rYL006_P1000/2024-11-06_15-36_rYL006_P1000_MotorLearningStop_30min",
        # "RUN_rYL006/rYL006_P1000/2024-11-06_16-20_rYL006_P1000_MotorLearningStop_27min",
        # "RUN_rYL006/rYL006_P1000/2024-11-07_16-13_rYL006_P1000_MotorLearningStop_20min",
        # "RUN_rYL006/rYL006_P1000/2024-11-08_15-01_rYL006_P1000_MotorLearningStop_16min",
        # "RUN_rYL006/rYL006_P1000/2024-11-08_15-56_rYL006_P1000_MotorLearningStop_18min",
        # "RUN_rYL006/rYL006_P1000/2024-11-11_16-11_rYL006_P1000_MotorLearningStop_25min",
        "RUN_rYL006/rYL006_P1000/2024-11-12_17-18_rYL006_P1000_MotorLearningStop_20min",
    ]
    for subdir in subdirs:
        subdir = os.path.join(EC.NAS_DIR, subdir)
        fname = 'ephys_output.raw.h5'
        if not os.path.exists(os.path.join(subdir, fname)):
            print(f"Error: {os.path.join(subdir, fname)} does not exist.")
            continue
        convert2neuroscope(subdir, fname)
    
    # subdir = os.path.join(EC.NAS_DIR,"RUN_rYL006/2024-11-01_15-47-09_active")
    # subdir = os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-11-04_16-13_rYL006_P1000_MotorLearningStop_22min")
    # # /Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/
    # fname = 'ephys_output.raw.h5'
    # # check_data(subdir, fname, n_plots=10, n_samples=10_000)

    # if not os.path.exists(os.path.join(subdir, fname)):
    #     print(f"Error: {os.path.join(subdir, fname)} does not exist.")
        
    # convert2neuroscope(subdir, fname)
>>>>>>> Stashed changes
    
if __name__ == "__main__":
    main()