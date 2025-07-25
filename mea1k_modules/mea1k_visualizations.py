import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ephys_constants as C

import colorsys
import matplotlib.colorbar as mcolorbar

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

def vis_shank_traces(data, implant_mapping, scaler=80, stim_mea1k_el=None, uVrange=6438, stimulated=None):
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
    trace_endpoints = {}
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
        
        min_potential = depth -scaler*uVrange/2
        max_potential = depth +scaler*uVrange/2
        # right_ax.text(data.shape[1], max_potential, f"{-uVrange/1000/2:.1f} mV", va='center', ha='left', fontsize=7)
        # right_ax.text(data.shape[1], min_potential, f"{uVrange/1000/2:.1f} mV", va='center', ha='left', fontsize=7)
        bg_rects.append(plt.Rectangle((0, min_potential), data.shape[1], scaler*uVrange,
                                      facecolor='black', alpha=.03 if not stim_pad else .15))
        
        for iloc in traces_iloc:
            if implant_mapping.iloc[iloc].mea1k_el == stim_mea1k_el:
                kwargs = {'lw':2, "linestyle":':'}
            else:
                kwargs = {"lw":.5, 'linestyle':'solid'}
            col = pad_colors[i] if not stim_pad else 'black'
            implant_mapping.at[implant_mapping.index[iloc], 'color'] = col
            # print(min(data[iloc]), max(data[iloc]))
            transformed_trace = -1*data[iloc]*scaler + depth
            right_ax.plot(transformed_trace, color=col, **kwargs)
            trace_endpoints[implant_mapping.iloc[iloc].mea1k_el] = transformed_trace[-1]
        
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
    
    print(trace_endpoints)
    print()
    print()
    print()
    if stimulated is not None:
        # print(implant_mapping)
        stimulated = stimulated.set_index("electrode").reindex(implant_mapping.mea1k_el)
        # print(stimulated)
        for i, (mea1k_el, imp_info) in enumerate(stimulated.iterrows()):
            # print(mea1k_el, imp_info)
            # if pd.isna(imp_info.stim):
            #     continue
            stim_unit = imp_info.stim_unit
            if pd.isna(stim_unit):
                corr_ax.text(-.3, trace_endpoints[mea1k_el], 
                              f"{imp_info.imp_stim_ratio:.1f}", va='center', ha='left', fontsize=7,)
            else:
                corr_ax.text(-.3, trace_endpoints[mea1k_el], 
                              f"{imp_info.imp_kOhm:.1f} kOhm, StimUnit:{int(stim_unit)}", va='center', ha='left',fontsize=12,)
                fig.suptitle(f"Pad ID: {imp_info.pad_id}")
            
            
            
        # corr_ax.xaxis.set_visible(False)
        # fullfname = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/devices/implant_devices/250205_MEA1K03_H1278pad4shankB5/recordings/2025-03-25_15.32.21_invivo_imp_mode='small_current'_stimpulse='sine'_amplitude=10/processed/extracted_imp_voltages.csv"
        # impdata = pd.read_csv(fullfname)
        # impdata = impdata[impdata.stim_unit.notna()]
        # impdata.sort_values(by='pad_id', inplace=True)
        
        # corr_ax.barh(width=impdata.imp_kOhm.values, y=impdata.depth.values, height=25)
        # set x aaxis to log scale
        # corr_ax.set_xscale('log')
    
    
    
    # exit()
    
    # print(left_ax.get_ylim())
    # print out current ylim


    # remove spines and add electrode pads
    [left_ax.add_patch(rect) for rect in el_pads_rects]
    [spine.set_visible(False) for spine in right_ax.spines.values()]
    [spine.set_visible(False) for spine in left_ax.spines.values()]
    [spine.set_visible(False) for spine in corr_ax.spines.values()]

    

def draw_mea1k(bg='black', el_color='#111111', mapping=None, cmap_scaler=1):
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
    elif isinstance(el_color, list):
        colors = el_color
    else:
        colors = [el_color]*26400

    i = 0
    recs = []
    # mea1k_yx = []
    for y in np.arange(0+17.5/4, 2100, 17.5):
        for x in np.arange(0+17.5/4, 3850, 17.5):
            recs.append(plt.Rectangle((x, y), 9, 9, facecolor=colors[i], 
                                      edgecolor='none', alpha=.7))
            if mapping is not None:
                # change color to connectivity
                whiteness = np.clip(mapping.loc[i].mea1k_connectivity*cmap_scaler, 0, 1)
                recs[-1].set_facecolor((whiteness, whiteness, whiteness))
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
        cbar_fig.subplots_adjust(right=0.25, top=0.98, bottom=0.02)
    elif orientation == 'horizontal':
        cbar_fig, cbar_ax = plt.subplots(figsize=(2100/300, 1.4))
        # cbar_fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
        cbar_fig.subplots_adjust(top=.95, bottom=.5)
    else:
        raise ValueError("Orientation must be either 'vertical' or 'horizontal'")
    
    cbar = mcolorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation=orientation)
    cbar.set_label(lbl)
    return cbar_fig, cbar

def draw_interconnect_pads(mapping, edgecolor='pad_id', pad_alpha=.6, add_pad_label=True,
                           pad_scalar=1.5, draw_on_ax=None, skip_not_connected=True):
    pad_circles, texts = [], []
    for pad_id in sorted(mapping.pad_id.dropna().unique()):
        pad_entry = mapping[mapping['pad_id'] == pad_id].sort_values('connectivity_order').iloc[0]
        
        if edgecolor == 'pad_id':
            col = pad_entry[['r', 'g', 'b']].values/255
        elif edgecolor=="metal":
            col = 'purple' if pad_entry.metal == 2 else 'green'
        elif isinstance(edgecolor, dict):
            col = edgecolor.get(int(pad_id), [.3,.3,.3])
        else:
            col = edgecolor
            
        if skip_not_connected and pad_entry.mea1k_connectivity < .8:
            continue
        
        pad_circles.append(plt.Circle((pad_entry.x_aligned, pad_entry.y_aligned), 
                                       pad_entry.pad_diameter_um/2 *pad_scalar, 
                                       color=col,
                                       fill=True, linewidth=2,
                                       alpha=pad_alpha))
        if add_pad_label:
            draw_on_ax.text(pad_entry.x_aligned, pad_entry.y_aligned, f"{int(pad_id)}",
                                    fontsize=7, ha='center', va='center', color='white')
    if draw_on_ax is not None:
        [draw_on_ax.add_patch(pc) for pc in pad_circles]
    return pad_circles

def viz_mea1k_config(pad_alignment, stim_mea1k_el=None, col_scaler=1):
     # darw mewa1k
    
    (fig, ax), els = draw_mea1k()
    
    pad_circles = []
    for el_i, el_rec in enumerate(els):
        if el_i not in pad_alignment.mea1k_el.values:
            continue # not measured during connectivity analysis
        el_entry = pad_alignment[pad_alignment['mea1k_el'] == el_i].iloc[0]
        el_rec.set_alpha(1)
        if pd.isna(el_entry.pad_id):
            continue
        
        col = pad_alignment[pad_alignment['mea1k_el'] == el_i][['r', 'g', 'b']].values[0]/255
        # el_rec.set_facecolor(col)

        whiteness = np.clip(el_entry.mea1k_connectivity*col_scaler, 0, 1)
        el_rec.set_facecolor((whiteness, whiteness, whiteness))
        
        if el_i == stim_mea1k_el:
            el_rec.set_edgecolor("yellow")
        
        if (el_entry.connectivity_order == 1) and (el_entry.mea1k_connectivity > .8) and el_entry.notna()["shank_id"]:
            # el_rec.set_alpha(1)
            el_rec.set_facecolor(col)
            pad_circles.append(plt.Circle((el_entry.x_aligned, el_entry.y_aligned), 
                                          el_entry.pad_diameter_um/2 *1.7, color=col, 
                                          fill=False, linewidth=.8,
                                          alpha=.5))
            ax.text(el_entry.x_aligned, el_entry.y_aligned, f"{int(el_entry.pad_id)}",
                                fontsize=7, ha='center', va='center', color='white')
            
    [ax.add_patch(pc) for pc in pad_circles]