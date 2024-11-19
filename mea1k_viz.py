import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
import matplotlib.colorbar as mcolorbar

from scipy.cluster.hierarchy import linkage

from ephys_constants import DEVICE_NAME

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
    

def validate_shank_map(data, implant_mapping, shank_id=2, min_depth=0, 
                       max_depth=8200, scaler=80):
    print('------------------------------------')
    print(data)
    print(implant_mapping)
    
    # first subset the mapping to the electrodes that are in the data
    shank_mapping = implant_mapping[implant_mapping.mea1k_el.isin(data.index)].set_index('mea1k_el')
    shank_mapping = shank_mapping[shank_mapping['shank_id'] == shank_id]
    # this subset has all pads, but also all electrodes below the pads
    all_shank_pads = implant_mapping[implant_mapping['shank_id'] == shank_id].sort_values('depth', ascending=False)
    
    # right side of shank
    all_shank_pads = all_shank_pads[all_shank_pads.shank_side == 'right']
    
    # make plot
    fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(16, 9), sharey=True, 
                                            width_ratios=[.005,.9])
    fig.subplots_adjust(wspace=0, hspace=0, right=.9, left=.03)
    fig.suptitle(f"{DEVICE_NAME} - {shank_mapping.shank_name.iloc[0]}", fontsize=16)
    
    el_pads_rects = []
    # iterate over all pads in the shank/ polyimide electrodes    
    for pad_id in all_shank_pads.pad_id.unique():
        # get info for the pad
        pad_info = all_shank_pads[all_shank_pads.pad_id == pad_id][['depth', 'metal', 'el_pair']].iloc[0]

        # draw only depth subset
        if min_depth <= pad_info['depth'] <= max_depth:
            # draw the electrode pad with color for metalization layer
            col = 'purple' if pad_info['metal'] == 2 else 'green'
            el_pads_rects.append(plt.Rectangle((0-13/2, pad_info['depth']-13/2), 
                                               13, 13, edgecolor=col, 
                                               facecolor='none', lw=1))
            # draw a line bewteen the electrode pairs (on same filber)
            el_pair_depths = all_shank_pads[all_shank_pads.el_pair == pad_info["el_pair"]].depth
            el_pair_depths = el_pair_depths.unique()
            left_ax.plot([8, 8], [min(el_pair_depths), max(el_pair_depths)], 
                         color='black', lw=1)
            
            # check if the pad is in the data, eg is one of the electrodes below
            # the pad is in the data
            if (pad_id == shank_mapping.pad_id.values).any():
                el_i = int(shank_mapping[shank_mapping.pad_id == pad_id].index[0])
                
                trace = data.loc[el_i] * scaler * -1 # negative because depth flipped
                right_ax.plot(trace+pad_info['depth'], color=col, lw=1)
                # right_ax.plot(-1*trace+pad_info['depth'], color='k', lw=1)
                el_pads_rects[-1].set_facecolor(col)
                left_ax.text(-16/2, pad_info['depth'], f"e{el_i:05d}", 
                             va='center', ha='right', fontsize=6)
            
    # set up x-axis for right plot      
    right_ax.set_xlim(0, data.columns[-1])
    right_ax.set_xticklabels([f"{xt/1000:.0f}ms" for xt in right_ax.get_xticks()])
    # set up y-axis for right plot
    right_ax.yaxis.set_ticks_position('right')
    right_ax.yaxis.set_label_position('right')
    right_ax.set_ylabel('Depth (um)')
    
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

    # remove spines and add electrode pads
    [left_ax.add_patch(rect) for rect in el_pads_rects]
    [spine.set_visible(False) for spine in right_ax.spines.values()]
    [spine.set_visible(False) for spine in left_ax.spines.values()]

    plt.show()