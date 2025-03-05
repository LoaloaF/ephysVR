import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ephys_constants as C

import colorsys

def adjust_saturation(rgb_color, offset):
    # Convert RGB (0-1 scale) to HSV
    r, g, b = rgb_color
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    # Adjust saturation by offset, ensuring it stays within [0, 1]
    s = max(0.0, min(1.0, s - offset))
    # Convert back to RGB
    return colorsys.hsv_to_rgb(h, s, v)

def _n_sequentially_different_colors(n, cmap='tab20'):
    cmap = plt.get_cmap(cmap)
    colors = [list(col) for col in cmap(np.linspace(0, 1, n))]
    return colors

def vis_shank_traces(data, implant_mapping, scaler=80, stim_mea1k_el=None):
    implant_mapping.reset_index(drop=True, inplace=True)
    # print(data)
    # print(implant_mapping)
    # print('------------------------------------')
    assert implant_mapping.shank_id.nunique() == 1, "Only one shank is supported"
    
    # rec_els = data.index.get_level_values('mea1k_el')
    # rec_subset_mapping = implant_mapping[implant_mapping.mea1k_el.isin(rec_els)]
    min_depth, max_depth = implant_mapping.depth.min()-70, implant_mapping.depth.max()+70
    depth_range = max_depth - min_depth
    
    # make plot
    fig, (left_ax, right_ax, corr_ax) = plt.subplots(1, 3, figsize=(16, 9), sharey=True, 
                                            width_ratios=[0.01,.9, .1])
    fig.subplots_adjust(wspace=0, hspace=0, right=.97, left=.04)
    # shank_id = data.index[0][0]
    # shank_id = implant_mapping.shank_id[0]
    # shank_name = implant_mapping[implant_mapping.shank_id == shank_id].shank_name.values[0]
    # fig.suptitle(f"{shank_name}", fontsize=16)
    
    el_pads_rects = []
    pad_colors = _n_sequentially_different_colors(implant_mapping.pad_id.nunique())
    implant_mapping["color"] = None
    bg_rects = []
    
    # iterate over all pads in the shank/ polyimide electrodes    
    for i, pad_id in enumerate(implant_mapping.pad_id.unique()):
    # for i, row in implant_mapping.iterrows():
        # pad_id = row.pad_id
        # get info for the pad
        metal = implant_mapping[implant_mapping.pad_id == pad_id].metal.values[0]
        depth = implant_mapping[implant_mapping.pad_id == pad_id].depth.values[0]
        fiber = implant_mapping[implant_mapping.pad_id == pad_id].el_pair.values[0]
        shank_side = implant_mapping[implant_mapping.pad_id == pad_id].shank_side.values[0]
        
        col = 'purple' if metal == 2 else 'green'
        side = 2 if shank_side == 'right' else -2
        left_ax.scatter(side, depth, color=col, s=8+depth_range**(2/5), marker='s')
        if metal == 1:
            same_fiber_metal2 = implant_mapping[(implant_mapping.el_pair == fiber) &
                                                (implant_mapping.metal == 2)]
            if len(same_fiber_metal2)>=1:
                same_fiber_metal2 = same_fiber_metal2.iloc[0]
                left_ax.plot([side, side], [depth, same_fiber_metal2.depth], 
                             color='black', lw=2, alpha=.2, zorder=0)
        
        # background rectangles indicating value range
        
        
        # if row.mea1k_el == stim_mea1k_el:
        #     col = 'red'
        traces_iloc = np.where(implant_mapping.pad_id == pad_id)[0]
        stim_pad = np.isin(stim_mea1k_el, implant_mapping.iloc[traces_iloc].mea1k_el)
        
        min_potential = depth -scaler*6438/2
        max_potential = depth +scaler*6438/2
        right_ax.text(data.shape[1], max_potential, f"0 mV", va='center', ha='left', fontsize=7)
        right_ax.text(data.shape[1], min_potential, f"6.4 mV", va='center', ha='left', fontsize=7)
        bg_rects.append(plt.Rectangle((0, min_potential), data.shape[1], scaler*6438,
                                      facecolor='black', alpha=.03 if not stim_pad else .15))
        
        for iloc in traces_iloc:
            if implant_mapping.iloc[iloc].mea1k_el == stim_mea1k_el:
                kwargs = {'lw':2, "linestyle":':'}
            else:
                kwargs = {"lw":.5, 'linestyle':'solid'}
            col = pad_colors[i] if not stim_pad else 'black'
            implant_mapping.at[implant_mapping.index[iloc], 'color'] = col
            # print(min(data[iloc]), max(data[iloc]))
            right_ax.plot(-1*data[iloc]*scaler + depth , color=col, **kwargs)
        
        continue
    # draw rectangels for the ackground
    [right_ax.add_patch(rect) for rect in bg_rects]
    
    
    # for trace_i in range(1, len(data)-1):
    #     trace = data[trace_i] * scaler * -1
    #     prv_trace = data[trace_i-1] * scaler * -1
    #     nxt_trace = data[trace_i+1] * scaler * -1
    #     corr_prv = np.corrcoef(prv_trace, trace)[0,1]
    #     corr_nxt = np.corrcoef(nxt_trace, trace)[0,1]
    #     avg_corr = max(corr_prv, corr_nxt)
    #     corr_ax.scatter(avg_corr, depth)
        
        # goodtrace = avg_corr > .8
        # trace_style = {'alpha': 1 if goodtrace else .5}
    
        
        # # check if the pad is in the data, eg is one of the electrodes below
        # # the pad is in the data
        # if (pad_id == rec_subset_mapping.pad_id.values).any():
        #     el_i = int(rec_subset_mapping[rec_subset_mapping.pad_id == pad_id].mea1k_el)
        #     idx = np.where(data.index.get_level_values('mea1k_el') == el_i)[0][0]
        #     trace = data.iloc[idx] * scaler * -1 # negative because depth flipped
            
        #     # plot correlation of neighboring electrodes
        #     trace_style = {'linestyle': 'solid', 'linewidth': 1}
        #     # draw the actual trace
        #     right_ax.plot(trace+depth, color=col, **trace_style)
        #     # right_ax.plot(-1*trace+pad_info['depth'], color='k', lw=1)
        #     el_pads_rects[-1].set_facecolor(col)
        #     left_ax.text(-16/2, depth, f"e{el_i:05d}", 
        #                     va='center', ha='right', fontsize=6)
        
    # # set up x-axis for right plot      
    # right_ax.set_xlim(0, data.columns[-1])
    # right_ax.set_xticklabels([f"{xt/1000:.0f}ms" for xt in right_ax.get_xticks()])
    # # set up y-axis for right plot
    # corr_ax.yaxis.set_ticks_position('right')
    # corr_ax.yaxis.set_label_position('right')
    # corr_ax.set_xlabel('Depth Neighbour Correlation')
    # # make the canvas for the correlation plot transparent
    # corr_ax.set_facecolor('none')
    
    # # draw yscale amplitude bar
    # right_ax.plot([data.columns[-1], data.columns[-1]],  
    #               [min_depth+50, min_depth+50+scaler], color='black', lw=4)
    # right_ax.text(data.columns[-1], min_depth+50+scaler/2, f"1mV",
    #                 va='center', ha='left', fontsize=9)
    
    # set up x, y-axis for left plot
    # left_ax.set_xlim(-13/2, 10)
    # left_ax.xaxis.set_visible(False)
    # left_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    left_ax.set_ylim(max_depth, min_depth)    
    left_ax.set_ylabel('Shank depth (um)')
    left_ax.set_xlim(-4,4)    
    left_ax.set_xticks([])
    # set ytick labels fontsize
    left_ax.tick_params(axis='y', labelsize=8)
    
    right_ax.set_xlabel('Sample ID (20,000 samples/s)')
    corr_ax.xaxis.set_visible(False)
    
    # print(left_ax.get_ylim())
    # print out current ylim


    # remove spines and add electrode pads
    [left_ax.add_patch(rect) for rect in el_pads_rects]
    [spine.set_visible(False) for spine in right_ax.spines.values()]
    [spine.set_visible(False) for spine in left_ax.spines.values()]
    [spine.set_visible(False) for spine in corr_ax.spines.values()]

    plt.show()
    

def draw_mea1k(bg='black', el_color='#111111'):
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
    C.MEA1K_EL_2D_TABLE_PIXEL
    for y in np.arange(0+17.5/4, 2100, 17.5):
        for x in np.arange(0+17.5/4, 3850, 17.5):
            recs.append(plt.Rectangle((x, y), 9, 
                                      C.MEA1K_EL_HEIGHT_MICROMETER, 
                                      facecolor=colors[i], 
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
 
def viz_mea1k_config(pad_alignment, stim_mea1k_el=None):
     # darw mewa1k
    
    (fig, ax), els = draw_mea1k()
    
    pad_circles = []
    for el_i, el_rec in enumerate(els):
        if el_i not in pad_alignment.mea1k_el.values:
            continue # not measured during connectivity analysis
        el_entry = pad_alignment[pad_alignment['mea1k_el'] == el_i].iloc[0]
        el_rec.set_alpha(min(1,el_entry.mea1k_connectivity/3))
        if pd.isna(el_entry.pad_id):
            continue
        
        col = pad_alignment[pad_alignment['mea1k_el'] == el_i][['r', 'g', 'b']].values[0]/255
        # el_rec.set_facecolor(col)

        whiteness = np.clip(el_entry.mea1k_connectivity*1, 0, 1)
        el_rec.set_facecolor((whiteness, whiteness, whiteness))
        
        if el_i == stim_mea1k_el:
            el_rec.set_edgecolor("yellow")
        
        if (el_entry.connectivity_order == 1) and (el_entry.mea1k_connectivity > .8) and el_entry.notna()["shank_id"]:
            el_rec.set_alpha(1)
            el_rec.set_facecolor(col)
            pad_circles.append(plt.Circle((el_entry.x_aligned, el_entry.y_aligned), 
                                          el_entry.pad_diameter_um/2 *1.7, color=col, 
                                          fill=False, linewidth=.8,
                                          alpha=.5))
    [ax.add_patch(pc) for pc in pad_circles]
    plt.show()