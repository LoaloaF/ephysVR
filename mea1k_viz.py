import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
import matplotlib.colorbar as mcolorbar

from scipy.cluster.hierarchy import linkage

import ephys_constants as C
import mea1k_ephys as mea1k
from mea1k_ephys import read_raw_data


def plot_trace(signal, positive_freqs, positive_power_spectrum, agg_power, 
               flt_signal, avg_ampl, implant_mapping=None, label=None, 
               filter_label=None, max_ampl=30):
    fig, ax = plt.subplots(3, 1, figsize=(12, 6))
    fig.subplots_adjust( hspace=.5)
    
    ax[0].set_title(label)
    if implant_mapping is not None:
        ampl = implant_mapping[implant_mapping.mea1k_el == signal.name].ampl
        ax[0].set_title(f"{label} - previous connectivity: {ampl.values[0]:.2f} mV")
    
    t = signal.index/1000
    ax[0].plot(t, signal.values, color='blue', alpha=.8, label='raw signal')
    ax[0].set_ylabel(f'Δ Potential [mV]')
    ax[0].set_yticks(np.array((-max_ampl,0,max_ampl)))
    ax[0].set_xlabel('Time [ms]')
    ax[0].grid(True)
    [ax[0].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
    ax[0].legend()
    
    ax[1].plot(positive_freqs, positive_power_spectrum, color='orange',
                label='Power Spectrum')
    ax[1].scatter([], [], edgecolor='orange', facecolor='none', 
                    label=f'Mean power: {agg_power:.1e} ({filter_label})', s=100)
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Power')
    ax[1].set_xlim(0, 3000)
    ax[1].set_ylim(0, 1e5//2)
    ax[1].grid(True)
    [ax[1].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
    ax[1].legend()
    
    ax[2].plot(t, flt_signal, color='blue', alpha=.5,
                label=f'{filter_label} signal')
    ax[2].plot([t[0]-20,t[-1]+20], [avg_ampl,avg_ampl], color='k', 
                linestyle='dashed', label=f'Average Amplitude: {avg_ampl:.3f} mV')
    ax[2].set_xlabel('Time [ms]')
    ax[2].set_ylabel('Amplitude')
    ax[2].set_ylabel(f'Δ Potential [mV]')
    ax[2].set_yticks(np.array((-max_ampl,0,max_ampl)))
    ax[2].grid(True)    
    ax[2].sharex(ax[0])
    [ax[2].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
    ax[2].legend()
    plt.show()

def plot_implanted_connectivity_freq_bands(path):
    all_powers = pd.read_csv(os.path.join(path, 'connectivity.csv'), index_col=0)
    
    plt.suptitle('Power in two freq bands, in the brain')
    plt.scatter(all_powers['outside_brain_ampl'], all_powers['300-3000Hz'], 
                label='300-3000Hz band', alpha=.5, s=1)
    plt.scatter(all_powers['outside_brain_ampl'], all_powers['<300Hz'], 
                label='<300Hz band', alpha=.5, s=1)
    plt.xlabel('Connectivity before implantation, ext signal ampl [mV]')
    plt.ylabel('Power in band')
    plt.legend()
    plt.ylim(0,500)
    plt.show()

def correlate_traces(config_traces, implant_mapping):
    connected_els = implant_mapping[implant_mapping.ampl > 20].mea1k_el.values
    ticks = [i for i in np.arange(config_traces.shape[0]) if config_traces.index[i] in connected_els]
    
    corr = np.corrcoef(config_traces.values)
    # Replace non-finite values with 0
    corr[~np.isfinite(corr)] = 0
    # Perform hierarchical clustering
    linkage_matrix = linkage(corr, method='ward')
    # Create a clustermap with seaborn
    g = sns.clustermap(corr, cmap='coolwarm', vmin=-1, vmax=1, row_linkage=linkage_matrix, col_linkage=linkage_matrix)
    
    # Adjust the ticks to show connected electrodes
    ax = g.ax_heatmap
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([config_traces.index[i] for i in ticks], rotation=90)
    ax.set_yticklabels([config_traces.index[i] for i in ticks])
    plt.show()

def draw_mea1k(bg='black', el_color='#222222'):
    fig, ax = plt.subplots(figsize=(3850/300, 2100/300), facecolor='none')
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    # fig.patch.set_facecolor('black')

    if isinstance(bg, np.ndarray):
        ax.imshow(bg)
    else:
        ax.set_facecolor(bg)
    
    # draw 26400 colors from hsv colormap
    # cmap = plt.cm.get_cmap('hsv', 26400)
    if el_color == 'hsv':
        cmap = plt.get_cmap('hsv', 26400)
        colors = [list(col) for col in cmap(np.linspace(0, 1, 26400))]
    else:
        colors = [el_color]*26400
    
    i = 0
    recs = []
    # mea1k_yx = []
    for y in np.arange(0+17.5/4, 2100, 17.5):
        for x in np.arange(0+17.5/4, 3850, 17.5):
            recs.append(plt.Rectangle((x, y), 9, 9, facecolor=colors[i], 
                                      edgecolor='none', alpha=.7))
            # mea1k_yx.append((x+4.5,y+4.5))
            i += 1
    # plt.scatter(*zip(*mea1k_yx), c='red', s=10)
            

    [ax.add_patch(rec) for rec in recs]
    ax.set_ylim(2100, 0)
    ax.set_xlim(0,3850)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    [ax.spines[spine].set_visible(False) for spine in ax.spines]
    
    # plt.show()
    # plt.savefig(f"./el_pads_v3.png", dpi=300, transparent=True if bg=='transparent' else False, 
    #             bbox_inches='tight', pad_inches=0, )
    return (fig, ax), recs

def draw_mea1K_colorbar(cmap, norm, lbl, orientation='vertical'):
    if orientation == 'vertical':
        cbar_fig, cbar_ax = plt.subplots(figsize=(1.4, 2100/300))
        cbar_fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
        cbar_fig.subplots_adjust(right=0.25)
    elif orientation == 'horizontal':
        cbar_fig, cbar_ax = plt.subplots(figsize=(2100/300, 1.4))
        # cbar_fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
        cbar_fig.subplots_adjust(top=.95, bottom=.5)
    else:
        raise ValueError("Orientation must be either 'vertical' or 'horizontal'")
    
    cbar = mcolorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation=orientation)
    cbar.set_label(lbl)
    return cbar_fig, cbar
    
def plot_connectivity_parallel_ext_signal(path):
    all_amplitudes = pd.read_csv(os.path.join(path, 'ext_signal_ampl.csv'), index_col=0).ampl
    
    cmap = plt.get_cmap('gray_r')
    vmin, vmax = 0, all_amplitudes.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    (fig, ax), el_pads = draw_mea1k(bg='white', el_color='#eeeeee')
    fig_cb, _ = draw_mea1K_colorbar(cmap, norm, "Amplitude [mV]")

    for i, rec in enumerate(el_pads):
        if i in all_amplitudes.index:
            rec.set_facecolor(cmap(norm(all_amplitudes.loc[i])))
    plt.show()

def plot_connectivity_neighbour_corr(path):
    corr = pd.read_csv(os.path.join(path, 'neighbour_el_corr.csv'), index_col=(0,1))
    
    # make index uniuqe and duplate permutations (el1,el2) == (el2,el1)  
    sorted_index = corr.index.map(lambda x: f"{min(x):05d}-{max(x):05d}")
    corr.index = sorted_index
    corr = corr.groupby(level=[0]).mean().dropna()

    (fig,ax), el_pads = draw_mea1k(bg='white', el_color='#aaaaaa')
    for el2el, pearson_r in corr.iterrows():
        weight = max(.1,pearson_r.item())
        el1, el2 = map(int, el2el.split('-'))
        plt.plot([el_pads[el1].get_x()+4.5, el_pads[el2].get_x()+4.5], 
                 [el_pads[el1].get_y()+4.5, el_pads[el2].get_y()+4.5], 
                 color='k', lw=weight,
                 alpha=weight)
    plt.show()


def compare_2_configs(old_config_subdir, new_config_subdir, fname, timeslice=200*1*1):
    implant_mapping = mea1k.get_implant_mapping(C.NAS_DIR, C.DEVICE_NAME)
    old_config_data = read_raw_data(old_config_subdir, fname,
                         col_slice=slice(0, timeslice), subtract_dc_offset=True)
    old_config_data, _ = mea1k.assign_mapping_to_data(old_config_data, implant_mapping)
    old_config_data = old_config_data.loc[[1,2],:]
    
    new_config_data = read_raw_data(new_config_subdir, fname, 
                            col_slice=slice(0, timeslice), subtract_dc_offset=True)
    new_config_data, _ = mea1k.assign_mapping_to_data(new_config_data, implant_mapping)
    new_config_data = new_config_data.loc[[1,2],:]
    
    old_config_els = old_config_data.index.get_level_values('mea1k_el')
    new_config_els = new_config_data.index.get_level_values('mea1k_el')
    
    (fig, ax), els = draw_mea1k()
    
    pad_circles = []
    for el_i, el_rec in enumerate(els):
        if el_i not in implant_mapping.mea1k_el.values:
            continue # not measured during connectivity analysis
        el_entry = implant_mapping[implant_mapping['mea1k_el'] == el_i].iloc[0]
        el_rec.set_alpha(min(1, el_entry.mea1k_connectivity / 25))

        if pd.isna(el_entry.pad_id):
            continue
        col = implant_mapping[implant_mapping['mea1k_el'] == el_i][['r', 'g', 'b']].values[0] / 255
        # el_rec.set_facecolor(col)
        el_rec.set_facecolor(np.array([1., 1., 1.]) * min(1, el_entry.mea1k_connectivity / 25))
        el_rec.set_zorder(10)
        
        
        if el_i%100 == 0: print(el_i, end='...', flush=True)        
        if (el_i in old_config_els or el_i in new_config_els) and not (el_i in old_config_els and el_i in new_config_els):
            # el_rec.set_linewidth(2)
            # el_rec.set_edgecolor(col)
            
            pad_circles.append(plt.Circle((el_entry.x_aligned, el_entry.y_aligned), 
                                          26, color=col, fill=False, linewidth=.8))
            
            # get the rectangle coordinates
            lowleftx, lowlefty = el_rec.get_xy()
            width = el_rec.get_width()
            height = el_rec.get_height()
            baseline = lowlefty+height/2
            
            if el_i in old_config_els:
                old_trace = old_config_data.loc[pd.IndexSlice[:,:,:,el_i]].values
                old_trace = (old_trace.flatten() /1000) +baseline
                ax.scatter([lowleftx+width/2], [baseline], color='blue', s=20, alpha=.5,  zorder=20)
            if el_i in new_config_els:
                new_trace = new_config_data.loc[pd.IndexSlice[:,:,:,el_i]].values
                new_trace = (new_trace.flatten() /1000) +baseline
                ax.scatter([lowleftx+width/2], [baseline], color='green', s=20, alpha=.5, zorder=20)
            # xindex = np.linspace(lowleftx, lowleftx+width, len(old_trace))
            # ax.plot(xindex, old_trace, color='blue', alpha=.5, linewidth=1, zorder=20)
            # ax.plot(xindex, new_trace, color='red', alpha=.5, linewidth=1, zorder=20)
    
    [ax.add_patch(pc) for pc in pad_circles]
    plt.show()


# def validate_shank_map(data, implant_mapping, shank_id=2, min_depth=0, 
#                        max_depth=8200, scaler=80, single_side=None):
#     print('------------------------------------')
#     print(data)
#     print(implant_mapping)
    
#     # first subset the mapping to the electrodes that are in the data
#     shank_mapping = implant_mapping[implant_mapping.mea1k_el.isin(data.index)].set_index('mea1k_el')
#     shank_mapping = shank_mapping[shank_mapping['shank_id'] == shank_id]
#     # this subset has all pads, but also all electrodes below the pads
#     all_shank_pads = implant_mapping[implant_mapping['shank_id'] == shank_id].sort_values('depth', ascending=False)
    
#     # right side of shank
#     if single_side is not None:
#         all_shank_pads = all_shank_pads[all_shank_pads.shank_side == 'right']
    
#     # make plot
#     fig, (left_ax, right_ax, corr_ax) = plt.subplots(1, 3, figsize=(16, 9), sharey=True, 
#                                             width_ratios=[.005,.9, .1])
#     fig.subplots_adjust(wspace=0, hspace=0, right=.97, left=.03)
#     fig.suptitle(f"{DEVICE_NAME} - {shank_mapping.shank_name.iloc[0]}", fontsize=16)
    
#     el_pads_rects = []
#     prv_trace = None
#     # iterate over all pads in the shank/ polyimide electrodes    
#     for pad_id in all_shank_pads.pad_id.unique():
#         # get info for the pad
#         pad_info = all_shank_pads[all_shank_pads.pad_id == pad_id][['depth', 'metal', 'el_pair']].iloc[0]

#         # draw only depth subset
#         if min_depth <= pad_info['depth'] <= max_depth:
#             # draw the electrode pad with color for metalization layer
#             col = 'purple' if pad_info['metal'] == 2 else 'green'
#             el_pads_rects.append(plt.Rectangle((0-13/2, pad_info['depth']-13/2), 
#                                                13, 13, edgecolor=col, 
#                                                facecolor='none', lw=1))
#             # draw a line bewteen the electrode pairs (on same filber)
#             el_pair_depths = all_shank_pads[all_shank_pads.el_pair == pad_info["el_pair"]].depth
#             el_pair_depths = el_pair_depths.unique()
#             left_ax.plot([8, 8], [min(el_pair_depths), max(el_pair_depths)], 
#                          color='black', lw=1)
            
#             # check if the pad is in the data, eg is one of the electrodes below
#             # the pad is in the data
#             if (pad_id == shank_mapping.pad_id.values).any():
#                 el_i = int(shank_mapping[shank_mapping.pad_id == pad_id].index[0])
                
#                 trace = data.loc[el_i] * scaler * -1 # negative because depth flipped
#                 right_ax.plot(trace+pad_info['depth'], color=col, lw=1)
#                 # right_ax.plot(-1*trace+pad_info['depth'], color='k', lw=1)
#                 el_pads_rects[-1].set_facecolor(col)
#                 left_ax.text(-16/2, pad_info['depth'], f"e{el_i:05d}", 
#                              va='center', ha='right', fontsize=6)
                
#                 if prv_trace is not None and prv_depth is not None:
#                     corr = np.corrcoef(prv_trace, trace)[0,1]
#                     corr_ax.plot(corr, pad_info['depth'], 'o', color='black')
#                     print(corr, pad_info['depth'])
#                 prv_trace = trace.copy()
#                 prv_depth = pad_info['depth']
            
#     # set up x-axis for right plot      
#     right_ax.set_xlim(0, data.columns[-1])
#     right_ax.set_xticklabels([f"{xt/1000:.0f}ms" for xt in right_ax.get_xticks()])
#     # set up y-axis for right plot
#     right_ax.yaxis.set_ticks_position('right')
#     right_ax.yaxis.set_label_position('right')
#     right_ax.set_ylabel('Depth (um)')
    
#     # draw yscale amplitude bar
#     right_ax.plot([data.columns[-1], data.columns[-1]],  
#                   [min_depth+50, min_depth+50+scaler], color='black', lw=4)
#     right_ax.text(data.columns[-1], min_depth+50+scaler/2, f"1mV",
#                     va='center', ha='left', fontsize=9)
    
#     # set up x, y-axis for left plot
#     left_ax.set_xlim(-13/2, 10)
#     left_ax.xaxis.set_visible(False)
#     left_ax.set_ylim(max_depth, min_depth)
#     left_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    
#     # make the canvas for the correlation plot transparent
#     corr_ax.set_facecolor('none')
    

#     # remove spines and add electrode pads
#     [left_ax.add_patch(rect) for rect in el_pads_rects]
#     [spine.set_visible(False) for spine in right_ax.spines.values()]
#     [spine.set_visible(False) for spine in left_ax.spines.values()]
#     [spine.set_visible(False) for spine in corr_ax.spines.values()]

#     plt.show()


def view_spikes(data, implant_mapping, spikes, scaler=80):
    rec_els = data.index.get_level_values('mea1k_el')
    rec_subset_mapping = implant_mapping[implant_mapping.mea1k_el.isin(rec_els)]
    print(spikes)
    print(data)
    print(rec_subset_mapping)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(wspace=0, hspace=0, right=.97, left=.03)
    [spine.set_visible(False) for spine in ax.spines.values()]
    ax.spines['left'].set_visible(True)
    
    for el_identifer, trace in data.iterrows():
        # if el_i not in spikes.index:
        #     continue
        # print(el_identifer)
        # print(trace)
        channel = el_identifer[4]
        depth = el_identifer[1]
        
        
        trace = (trace * scaler * -1) + depth
        ax.plot(trace.values, color='black', lw=1)
        
        if channel in spikes.Channel:
            sp = spikes[spikes.Channel == channel]
            print(trace.iloc[sp.Timestamp])
            print(trace.iloc[sp.Timestamp]*sp.shape[0])
            ax.scatter(sp.Timestamp, trace.iloc[sp.Timestamp], color='red', s=100, alpha=.8, marker='|')
    
    ax.set_ylim(rec_subset_mapping.depth.min(), rec_subset_mapping.depth.max())
    plt.show()

def raster_spikes(spikes, implant_mapping, shank_id, sample_interval_us=50):
    # Make a raster plot of the spike times, each row is a different unit, colored by channel
    print(spikes)
    print(implant_mapping)
    shank_implan_mapping = implant_mapping[implant_mapping.shank_id == shank_id]
    fig, ax = plt.subplots(ncols=2, figsize=[10, 5], sharey=True, width_ratios=[.9, .1])
    fig.subplots_adjust(wspace=0, hspace=0, right=.97, left=.06)
    unique_channels = np.unique(spikes['Channel'])
    colors = plt.cm.get_cmap('tab20', len(unique_channels))

    for idx, chnl in enumerate(unique_channels):
        channel_spikes = spikes[spikes['Channel'] == chnl]
        chnl_depth = implant_mapping.iloc[chnl].depth
        chnl_shank = implant_mapping.iloc[chnl].shank_id
        if chnl_shank != shank_id:
            continue
        # Convert spike times from sample IDs to minutes
        spike_times_minutes = (channel_spikes.index * sample_interval_us) / 1e6 / 60
        ax[0].plot(spike_times_minutes, np.ones_like(spike_times_minutes) * chnl_depth, '|', markersize=2, color=colors(idx))

    from_depth = shank_implan_mapping.depth.min()
    to_depth = shank_implan_mapping.depth.max()
    ax[0].set_ylim(to_depth + 1, from_depth - 1)
    ax[0].set_ylabel('Depth [mm]')
    ax[0].set_yticks(np.arange(from_depth, to_depth, 1000))
    ax[0].set_yticklabels([f"{d//1000}" for d in ax[0].get_yticks()])
    ax[0].set_xlabel('Time [min]')

    for chnl, info in shank_implan_mapping.iterrows():
        col = 'purple' if info.metal == 2 else 'green'
        alpha = 1 if chnl in unique_channels else .1
        x = ((info.depth / to_depth) * (1 if info.shank_side == 'left' else -1)) 
        plt.scatter([x], (info.depth-to_depth)*-1, color=col, s=10, alpha=alpha, marker='s')
        
    ax[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax[0].spines['top'].set_visible(False)
    [ax[1].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
    ax[0].set_title(f"Raster Plot of Spike Times by Channel - Anterior long shank (HPC)")
    plt.show()


def validate_shank_map(data, implant_mapping, scaler):
    print('------------------------------------')
    print(data)
    print(implant_mapping)
    rec_els = data.index.get_level_values('mea1k_el')
    rec_subset_mapping = implant_mapping[implant_mapping.mea1k_el.isin(rec_els)]
    min_depth, max_depth = rec_subset_mapping.depth.min(), rec_subset_mapping.depth.max()
    
    # make plot
    fig, (left_ax, right_ax, corr_ax) = plt.subplots(1, 3, figsize=(16, 9), sharey=True, 
                                            width_ratios=[.005,.9, .1])
    fig.subplots_adjust(wspace=0, hspace=0, right=.97, left=.03)
    shank_id = data.index[0][0]
    shank_name = implant_mapping[implant_mapping.shank_id == shank_id].shank_name.values[0]
    fig.suptitle(f"{C.DEVICE_NAME} - {shank_name}", fontsize=16)
    
    el_pads_rects = []
    # iterate over all pads in the shank/ polyimide electrodes    
    for pad_id in implant_mapping[~implant_mapping.shank_id.isna()].pad_id.unique():
        # get info for the pad
        metal = implant_mapping[implant_mapping.pad_id == pad_id].metal.values[0]
        depth = implant_mapping[implant_mapping.pad_id == pad_id].depth.values[0]
        fiber = implant_mapping[implant_mapping.pad_id == pad_id].el_pair.values[0]
        
        col = 'purple' if metal == 2 else 'green'
        el_pads_rects.append(plt.Rectangle((0-13/2, depth-13/2), 
                                            13, 13, edgecolor=col, 
                                            facecolor='none', lw=1))
        # draw a line bewteen the electrode pairs (on same filber)
        el_pair_depths = implant_mapping[implant_mapping.el_pair == fiber].depth
        el_pair_depths = el_pair_depths.unique()
        left_ax.plot([8, 8], [min(el_pair_depths), max(el_pair_depths)], 
                        color='black', lw=1)
        
        # check if the pad is in the data, eg is one of the electrodes below
        # the pad is in the data
        if (pad_id == rec_subset_mapping.pad_id.values).any():
            el_i = int(rec_subset_mapping[rec_subset_mapping.pad_id == pad_id].mea1k_el)
            idx = np.where(data.index.get_level_values('mea1k_el') == el_i)[0][0]
            trace = data.iloc[idx] * scaler * -1 # negative because depth flipped
            
            # plot correlation of neighboring electrodes
            trace_style = {'linestyle': 'solid', 'linewidth': 1}
            if idx > 0 and idx < len(data)-1:
                prv_trace = data.iloc[idx-1] * scaler * -1
                nxt_trace = data.iloc[idx+1] * scaler * -1
                corr_prv = np.corrcoef(prv_trace, trace)[0,1]
                corr_nxt = np.corrcoef(nxt_trace, trace)[0,1]
                avg_corr = max(corr_prv, corr_nxt)
                corr_ax.scatter(avg_corr, depth, color=C.SHANK_BASE_COLORS[shank_id])
                
                goodtrace = avg_corr > .8
                trace_style = {'alpha': 1 if goodtrace else .5}
            
            # draw the actual trace
            right_ax.plot(trace+depth, color=col, **trace_style)
            # right_ax.plot(-1*trace+pad_info['depth'], color='k', lw=1)
            el_pads_rects[-1].set_facecolor(col)
            left_ax.text(-16/2, depth, f"e{el_i:05d}", 
                            va='center', ha='right', fontsize=6)
        
    # set up x-axis for right plot      
    right_ax.set_xlim(0, data.columns[-1])
    right_ax.set_xticklabels([f"{xt/1000:.0f}ms" for xt in right_ax.get_xticks()])
    # set up y-axis for right plot
    
    corr_ax.yaxis.set_ticks_position('right')
    corr_ax.yaxis.set_label_position('right')
    corr_ax.set_ylabel('Depth (um)')
    corr_ax.set_xlabel('Depth Neighbour Correlation')
    
    # draw yscale amplitude bar
    right_ax.plot([data.columns[-1], data.columns[-1]],  
                  [min_depth+50, min_depth+50+scaler], color='black', lw=4)
    right_ax.text(data.columns[-1], min_depth+50+scaler/2, f"1mV",
                    va='center', ha='left', fontsize=9)
    
    # set up x, y-axis for left plot
    left_ax.set_xlim(-13/2, 10)
    left_ax.xaxis.set_visible(False)
    left_ax.set_ylim(max_depth, min_depth)
    left_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    
    # make the canvas for the correlation plot transparent
    corr_ax.set_facecolor('none')
    

    # remove spines and add electrode pads
    [left_ax.add_patch(rect) for rect in el_pads_rects]
    [spine.set_visible(False) for spine in right_ax.spines.values()]
    [spine.set_visible(False) for spine in left_ax.spines.values()]
    [spine.set_visible(False) for spine in corr_ax.spines.values()]

    plt.show()
    
def vis_depth_corr_stabiiity(data):
    nshanks = len(data.index.get_level_values('shank_id').unique())
    nsessions = len(data.index.get_level_values('session_date').unique())

    # check which depth range each shank has, use for axis size scaling
    minmax_depth = lambda shankd: (shankd.index.get_level_values('depth').min(), 
                                   shankd.index.get_level_values('depth').max())
    depth_limits = np.stack(data.groupby('shank_id').apply(minmax_depth).values)
    depth_ranges = depth_limits[:,1] - depth_limits[:,0]
    height_ratios = depth_ranges / depth_ranges.sum()
    
    fig, axes = plt.subplots(figsize=(14, 9), nrows=nshanks, ncols=nsessions, 
                             sharex=True,
                           gridspec_kw={'height_ratios': height_ratios})
    fig.subplots_adjust(left=.03, right=.99, top=.9, bottom=.1, hspace=.1)
    
    for session_i, session_date in enumerate(data.index.get_level_values('session_date').unique()):
        print(session_date)
        session_data = data.loc[session_date]
        axes[0, session_i].set_title(f"{session_date}") 
        axes[-1, session_i].set_xlabel('Depth Neighbour Correlation')
    
        for shank_id in data.index.get_level_values('shank_id').unique():
            shank_data = session_data.loc[shank_id]
            ax = axes[int(shank_id)-1, session_i]
            
            corr, depth = shank_data.values, shank_data.index.get_level_values('depth').values
            # ax.scatter(corr, depth,
            #             color=C.SHANK_BASE_COLORS[shank_id], s=20, alpha=.7)
            # change to hoiizontal bar plot
            # print(corr, depth)
            # for i, (c,d) in enumerate(zip(corr, depth)):
            #     # ax.axhline(d, c, 0, color=C.SHANK_BASE_COLORS[shank_id], lw=1, alpha=.8)
            #     ax.plot([0, c], [d, d], color=C.SHANK_BASE_COLORS[shank_id], alpha=.5, lw=2)
            
            if session_i != 0:
                prv_session_data = data.loc[data.index.get_level_values('session_date').unique()[session_i-1]]
                prv_session_data_shank = prv_session_data.loc[shank_id]
                
                diff_corr = np.abs(prv_session_data_shank.values-corr)
                # plot median diff correlation
                ax.scatter(np.mean(diff_corr[~np.isnan(diff_corr)]), np.median(depth),
                        color='black', s=200, alpha=1, marker='|')
                for diff_corr, depth in zip(diff_corr, depth):
                    ax.plot([0, diff_corr], [depth, depth], lw=2,
                             alpha=.8, color=C.SHANK_BASE_COLORS[shank_id])    
                
        for i, ax in enumerate(axes[:, session_i]):
            ax.grid("on", axis="x")
            # grid only on x-axis
            [ax.spines[spine].set_visible(False) for spine in ax.spines]
            ax.set_yticks([])
            if session_i == 0:
                ax.set_ylabel(f"Shank {i+1}")
        
                
                
        
    plt.show()