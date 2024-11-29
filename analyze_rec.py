import numpy as np
import pandas as pd
import os

import mea1k_ephys as mea1k
import utils

import ephys_constants as EC
from mea1k_ephys import read_raw_data

from mea1k_viz import validate_shank_map, vis_depth_corr_stabiiity

def check_data(subdir, fname, n_plots=10, n_samples=10_000, shank_id=1, 
               channel_subset=None,
               min_depth=2000, max_depth=6000, scaler=100,  shank_side=None):
    
    implant_mapping = mea1k.get_implant_mapping(EC.NAS_DIR, EC.DEVICE_NAME)
    print(implant_mapping)
    print(implant_mapping[implant_mapping.routed].mea1k_connectivity.sort_values())
    exit()
    for i in range(n_plots):
        data = mea1k.read_raw_data(subdir, fname, convert2vol=True, to_df=True,
                                   subtract_dc_offset=True, convert2uVInt=True, 
                                   col_slice=slice(i*n_samples, (i+1)*n_samples))
        data, _ = mea1k.assign_mapping_to_data(data, implant_mapping)
        
        if channel_subset is not None:
            data = data.iloc[channel_subset]
        
        # slice data to passed parameters (shank_id, min_depth, max_depth, shank_side)        
        side_slice = slice(None) if shank_side is None else slice(shank_side)
        depths = data.index.get_level_values('depth')
        depth_mask = (depths >= min_depth) & (depths <= max_depth)
        data = data.loc[shank_id, depth_mask, side_slice]

        validate_shank_map(data, implant_mapping, scaler=scaler)
        
def convert2neuroscope(subdir, fname, timeslice=20_000*1*1):
    data = read_raw_data(subdir, fname, convert2vol=True,  convert2uVInt=True,
                         col_slice=slice(0, timeslice), subtract_dc_offset=True)
    implant_mapping = mea1k.get_implant_mapping(EC.NAS_DIR, EC.DEVICE_NAME)
    data, implant_mapping = mea1k.assign_mapping_to_data(data, implant_mapping)
    
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

def analyze_depth_corr(subdirs, fname, timeslice=20_000*1*1):
    all_corrs = []
    for i, subdir in enumerate(subdirs):
        if not os.path.exists(os.path.join(subdir, fname)): continue
        data = read_raw_data(subdir, fname, convert2vol=True,  convert2uVInt=True,
                            col_slice=slice(1000000, timeslice+1000000), subtract_dc_offset=True)
        implant_mapping = mea1k.get_implant_mapping(EC.NAS_DIR, EC.DEVICE_NAME)
        
        data, _ = mea1k.assign_mapping_to_data(data, implant_mapping)
        data = data.loc[:,:,'right']
        
        # prepend level to row multiindex
        date = os.path.basename(subdir).split('_')[0]
        data = pd.concat({date: data}, names=['session_date'])
        # all_data.append(data)
    
        # all_data = pd.concat(all_data)
        corrs = []
        for i in range(2, len(data)-2):
            prv_trace = data.iloc[i-1]
            trace = data.iloc[i]
            nxt_trace = data.iloc[i+1]
            
            corr = np.max([np.corrcoef(trace, tr)[0,1] for tr in [prv_trace, nxt_trace]])
            corrs.append(pd.Series(corr, name=data.index[i]))
            
        corrs = pd.concat(corrs, axis=1).T.iloc[:,0]
        corrs.index.names = data.index.names
        all_corrs.append(corrs)
        
    all_corrs = pd.concat(all_corrs)
    vis_depth_corr_stabiiity(all_corrs)
    
    
    
    

    


def main():
    subdirs = [
        os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-10-25_15-41_rYL006_P1000_MotorLearningStop_14min"),
        os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-11-04_16-13_rYL006_P1000_MotorLearningStop_22min"),
        os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-11-05_16-11_rYL006_P1000_MotorLearningStop_22min"),
        os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-11-06_15-36_rYL006_P1000_MotorLearningStop_30min"),
        # os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-11-06_16-20_rYL006_P1000_MotorLearningStop_27min"),
        os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-11-07_16-13_rYL006_P1000_MotorLearningStop_20min"),
        os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-11-08_15-01_rYL006_P1000_MotorLearningStop_16min"),
        os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-11-08_15-56_rYL006_P1000_MotorLearningStop_18min"),
        os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-11-11_16-11_rYL006_P1000_MotorLearningStop_25min"),
        os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-11-12_17-18_rYL006_P1000_MotorLearningStop_20min"),
        os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1000/2024-11-13_16-31_rYL006_P1000_MotorLearningStop_15min"),
        ]
    fname = 'ephys_output.raw.h5'
    
    # check_data(subdirs[-1], fname, n_plots=10, n_samples=10_000, shank_id=1, 
    #            min_depth=0, max_depth=6500, scaler=150, shank_side='right')
    check_data(subdirs[-1], fname, n_plots=10, n_samples=10_000, shank_id=2, 
               min_depth=0, max_depth=11000, scaler=150, channel_subset=[
                   [9, 9, 10, 13, 19, 45, 48, 69, 89, 95, 96, 105, 105, 107, 107, 108, 113, 119,
                    122, 122, 126, 141, 143, 176, 277, 277, 297, 308, 339, 339, 336, 342, 346]
               ])
    # convert2neuroscope(subdir, fname, timeslice=20_000*1*1) # samples * seconds
    # analyze_depth_corr(subdirs, fname)
    
    # mea1k.write_probe_file(subdir, fname)
        
if __name__ == "__main__":
    main()