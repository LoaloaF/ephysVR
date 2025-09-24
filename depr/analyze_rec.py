import numpy as np
import pandas as pd
import os
import scipy.cluster.hierarchy
import seaborn


import mea1k_ephys as mea1k
import utils
import time
import ephys_constants as EC
from mea1k_ephys import read_raw_data, get_recording_version

from mea1k_viz import validate_shank_map, vis_depth_corr_stabiiity, draw_mea1k, compare_2_configs
from mea1k_viz import raster_spikes, view_spikes

import matplotlib.pyplot as plt

def check_data(subdir, fname, n_plots=10, n_samples=10_000, shank_id=1, 
               channel_subset=None,
               min_depth=2000, max_depth=6000, scaler=100,  shank_side=None):
    
    implant_mapping = mea1k.get_implant_mapping(EC.NAS_DIR, EC.DEVICE_NAME)

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
        
def check_spikes(subdir, fname, spikes_fname, n_plots=10, n_samples=10_000, 
                 shank_id=1, channel_subset=None, min_depth=2000, max_depth=6000, 
                 scaler=100,  shank_side=None):
    implant_mapping = mea1k.get_implant_mapping(EC.NAS_DIR, EC.DEVICE_NAME)
    # for ordering
    data = mea1k.read_raw_data(subdir, fname, to_df=True, col_slice=slice(0,1))
    _, implant_mapping = mea1k.assign_mapping_to_data(data, implant_mapping)
    all_spikes = pd.read_csv(os.path.join(subdir,spikes_fname), index_col=0)
    print(all_spikes)

    # first vis    
    raster_spikes(all_spikes, implant_mapping, shank_id)
    all_spikes.reset_index(inplace=True)

    for i in range(n_plots):
        from_smple, to_smple = i*n_samples, (i+1)*n_samples
        data = mea1k.read_raw_data(subdir, fname, convert2vol=True, to_df=True,
                                   subtract_dc_offset=True, convert2uVInt=True, 
                                   col_slice=slice(from_smple, to_smple))
        data, _ = mea1k.assign_mapping_to_data(data, implant_mapping)
        
        if channel_subset is not None:
            data = data.iloc[channel_subset]
        
        # slice data to passed parameters (shank_id, min_depth, max_depth, shank_side)        
        side_slice = slice(None) if shank_side is None else slice(shank_side)
        depths = data.index.get_level_values('depth')
        depth_mask = (depths >= min_depth) & (depths <= max_depth)
        data = data.loc[shank_id, depth_mask, side_slice]

        # validate_shank_map(data, implant_mapping, scaler=scaler)
        spikes = all_spikes[(all_spikes['Timestamp'] >= from_smple) &
                            (all_spikes['Timestamp'] < to_smple) &
                            (all_spikes['Channel'].isin(data.index.unique("channel")))]
        view_spikes(data, implant_mapping, spikes, scaler=scaler)
        
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

def analyze_fullpadlayout_rec(subdir, cache='from'):
    implant_mapping = mea1k.get_implant_mapping(EC.NAS_DIR, EC.DEVICE_NAME)
    if cache == 'from':
        all_data = pd.read_pickle(os.path.join(EC.LOCAL_DATA_DIR, "", "fullpadlayout_rec.pkl"))
    else:
        fnames = os.listdir(subdir)
        all_data = []
        for i,fname in enumerate(fnames):
            print(f"{i}/{len(fnames)}")
            data = read_raw_data(subdir, fname, convert2vol=True,
                                subtract_dc_offset=True)
            data, _ = mea1k.assign_mapping_to_data(data, implant_mapping)
            all_data.append(data)
            # print(data)
        all_data = pd.concat(all_data)
        if cache == 'to':
            all_data.to_pickle(os.path.join(EC.LOCAL_DATA_DIR, "", "fullpadlayout_rec.pkl"))
    print(all_data.sort_index())
    implant_mapping = implant_mapping.set_index('mea1k_el')
    
    # (fig, ax), el_rec = draw_mea1k()
    # for el_i, el_rec in enumerate(els):
    #     if el_i not in all_data.index.get_level_values('mea1k_el'):
    #         continue
    #     el_col = implant_mapping.loc[el_i, ['r', 'g', 'b']].values.flatten()/255
    #     el_rec.set_facecolor(el_col)
    # plt.show()
    
    fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(19,10), width_ratios=(.9,.05))
    scaler = 30
    for shank_id in all_data.index.get_level_values('shank_id').unique():
        if shank_id not in (1,2): 
            continue
        
        data = all_data.loc[shank_id].sort_index(ascending=False).droplevel('shank_side')
        print(data)
        for polyimide_el_i, el_depth in enumerate(data.index.get_level_values('depth').unique()):
            polyimide_el_all_mea1k_els = data.loc[el_depth]            
            el_info = implant_mapping.reindex(polyimide_el_all_mea1k_els.index).sort_values('connectivity_order')
            el_info = el_info[el_info.mea1k_connectivity.notnull()]
            print(el_info)
            polyimide_el_all_mea1k_els = polyimide_el_all_mea1k_els.reindex(el_info.index)
            
            # [el_rec[el_i].set_facecolor('red') for el_i in polyimide_el_all_mea1k_els.index]
            # order = el_info.connnectivity_order.values
            
            connectivity = el_info.mea1k_connectivity.values
            connectivity = np.clip(connectivity, 0, 25)
            # ylocs = polyimide_el_i*200 + connectivity
            traces = (polyimide_el_all_mea1k_els.values[:, :20_000]*scaler)
            print(traces)
            # print(traces.shape)
            # col = el_info[['r', 'g', 'b']].values[0]/255
            # np.random.shuffle(col)
            # print(col)
            
            # get a rnadom color
            col = np.random.rand(3)
            
            alpha = np.ones(traces.shape[0])
            alpha[connectivity < 20] = 0.3
            # alpha = np.clip(alpha, 0.4, 1)
            for i in range(traces.shape[0]):
                axes[polyimide_el_i,0].plot(traces[i], color=col, alpha=alpha[i])
            # ax.plot(traces, color=col)
            # axes.axhline(ylocs[0], color='black', linestyle='--')
            # turn of axis labels
            axes[polyimide_el_i,0].set_yticks([])
            axes[polyimide_el_i,0].set_xticks([])
            
            corr = np.corrcoef(traces)
            
            # cluster
            linkage = scipy.cluster.hierarchy.linkage(corr, method='ward')
            dend = scipy.cluster.hierarchy.dendrogram(linkage, ax=axes[polyimide_el_i,1], no_labels=True)
            
            seaborn.clustermap(corr, row_linkage=linkage, col_linkage=linkage)
            
            
            # axes[polyimide_el_i,1].imshow(corr, aspect='equal')
            # axes[polyimide_el_i,1].set_yticks([])
            # axes[polyimide_el_i,1].set_xticks([])
            
            
            
            if polyimide_el_i >= 1:
                break
        fig.tight_layout()
        plt.show()
    
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
        os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1100/2024-11-15_15-48_rYL006_P1100_LinearTrackStop_35min"),
        os.path.join(EC.NAS_DIR,"RUN_rYL006/rYL006_P1100/2024-11-20_17-46_rYL006_P1100_LinearTrackStop_22min"),
        os.path.join(EC.NAS_DIR,"implant_devices/241016_headstage03_46pad4shank/recordings/20241122_complete_padlayout_rec2"),
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
    
    fname = 'ephys_output.raw.h5'
    
    # compare_2_configs(subdirs[-2], subdirs[-1], fname)
    # plot_pad_alignment()
    
    # check_data(subdirs[-1], fname, n_plots=3, n_samples=10_000, shank_id=1, 
    #            min_depth=0, max_depth=6500, scaler=.4, shank_side='right')
    # check_data(subdirs[-1], fname, n_plots=3, n_samples=10_000, shank_id=2, 
    #            min_depth=4000, max_depth=8000, scaler=.4,)# shank_side='right')
    # check_data(subdirs[-1], fname, n_plots=10, n_samples=10_000, shank_id=2, 
    #            min_depth=0, max_depth=11000, scaler=150, channel_subset=[
    #                [9, 9, 10, 13, 19, 45, 48, 69, 89, 95, 96, 105, 105, 107, 107, 108, 113, 119,
    #                 122, 122, 126, 141, 143, 176, 277, 277, 297, 308, 339, 339, 336, 342, 346]
    #            ])
    # convert2neuroscope(subdir, fname, timeslice=20_000*1*1) # samples * seconds
    # analyze_depth_corr(subdirs, fname)
    
    # mea1k.write_probe_file(subdir, fname)
    
    analyze_fullpadlayout_rec(subdirs[-1])
    
    spikes_fname = 'spike_data.csv'
    check_spikes(subdirs[-1], fname, spikes_fname, n_plots=10, n_samples=10_000, shank_id=2, 
                 min_depth=0, max_depth=10000, scaler=.4, shank_side=None)
        
if __name__ == "__main__":
    main()