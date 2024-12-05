import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
# import cv2

def viz_pad_pattern(connectivity, pad_mapping, path):
    (fig, ax), el_pads = draw_mea1k(bg='white', el_color='#dddddd')
    print(pad_mapping)
    connectivity = connectivity.sort_index()
    print(connectivity)
    scaler = ((2**16)-1)
    # for i, rec in enumerate(el_pads):
    #     if i in pad_mapping.index:
    #         # rec.set_edgecolor('gray')
    #         pass

    i = 0
    final_pads = {}
    for pad_i in pad_mapping['index'].unique():
        # print(pad_i)
        color = pad_mapping[pad_mapping['index']==pad_i].values[0, 1:]
        # print(color)
        els = pad_mapping[pad_mapping['index']==pad_i].index.values
        l1 = (len(els))
        # print(els)
        els = [e for e in els if e in connectivity.index]
        l2 = (len(els))
        if l1 != l2:
            print("Error: ", pad_i, l1, l2)
        if len(els) == 0:
            print("No elements in pad: ", pad_i)
            continue
        # print(connectivity.index[115:120])
        pad_els = connectivity.loc[els, 'ampl'].sort_values()
        # pad_els = pad_els[pad_els<30]
        val, target_el = pad_els.iloc[-1], pad_els.index[-1]
        # if val > 30:
        #     val, target_el = pad_els.iloc[-2], pad_els.index[-2]
            
        good_pads = []
        j = 0
        for el, val in pad_els.items():
            if val >20:
                # if j == 0:
                #     if val >20:
                el_pads[el].set_facecolor(color/scaler)
                el_pads[el].set_edgecolor(color/scaler)
                el_pads[el].set_linewidth(2)
                final_pads[target_el] = [pad_i, val]
                # print(val)
                i += 1
                        # pad_center = (el_pads[el].get_x() + 4.5, el_pads[el].get_y() + 4.5)
                        # make a circle patch
                        # good_pads.append(plt.Circle(pad_center, 20, linewidth=5, color=color/scaler, fill=False, alpha=.7))
                # else:
                #     el_pads[el].set_edgecolor(color/scaler)
                #     el_pads[el].set_linewidth(1)
                # j += 1
                # change rectange size
                # el_pads[el].set_width(20)
                # el_pads[el].set_height(20)
            else:
                el_pads[el].set_edgecolor("#aaaaaa")
                
        # [ax.add_patch(circ) for circ in good_pads]
        # if val >20:
        #     # if val>30:
        #     el_pads[target_el].set_facecolor(color/255)
        #     el_pads[target_el].set_edgecolor(color/255)
        #     i += 1
        # else:
        #     el_pads[target_el].set_edgecolor("gray")
        #     pass
        #     rec.set_edgecolor('gray')
            
    final_pads = pd.DataFrame(final_pads).T
    final_pads.columns = ['pad', 'ampl']
    print(final_pads)
    # save
    final_pads.to_pickle(f"{path}/final_pads.pkl")
    print(len(final_pads)/1278)
    # print(i)
    print(1278-len(final_pads))
    plt.show()
    
    print(f"{path}/final_pads.png")
    fig.savefig(f"{path}/final_pads.png", dpi=300, 
                bbox_inches='tight', pad_inches=0)
    
        
        
    
    
def viz_stim_pattern(data):
    mea1k = np.zeros(26400, dtype=float)
    
    config = 0
    # tile = 1
    stimset = 1
    print(data.index.unique("stim_set"))
    
    # print(data.loc[config, stimset])
    all_stimulated = []
    for tile in data.loc[config, stimset].index.get_level_values('tile').unique():
        d = data.loc[config, stimset, :, tile]
        stimulated = d[d.index.get_level_values('stimulated')==True]
        print(list(stimulated.index))
        all_stimulated.extend(list(stimulated.index.get_level_values('el'))) # should be 1
        # print(stimulated)
        rec = d[d.index.get_level_values('stimulated')==False]
        # mea1k[stimulated.index.get_level_values('el')] = stimulated.values
        mea1k[rec.index.get_level_values('el')] = rec.values
    fig, ax = plt.subplots()
    ax.set_title("Stimulated Pattern")
    fig.subplots_adjust(top=.99, bottom=.01, right=.8)
    cmap = plt.get_cmap('viridis')
    cmap.set_bad('black')
    mea1k[mea1k==0] = np.nan
    im = plt.imshow(mea1k.reshape(120,220), cmap=cmap)
    
    # coo = els2xy(all_stimulated)
    # print(coo)
    # stimulated = data[data.index.get_level_values('stimulated')==True]
    # draw an x at the stimulated electrodes
    plt.scatter(np.array(all_stimulated)% 220, np.array(all_stimulated)// 220, marker='2', color='white', s=100, linewidths=1)
    
        # Create an axis for the colorbar that matches the height of the imshow plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('1KHz Power')    
    
    plt.show()

def connectivity_hist(data, path):
    which = 'ampl'
    connectivity = data[which]

    t1_data = connectivity.values.copy()
    t1_data[t1_data>25] = 25
    t1_data  /= 25
    fig, ax = plt.subplots(figsize=(4,2))
    fig.subplots_adjust(left=.3)
    # col = '#7f9b70'
    # lbl = "Day 1"
    col = '#709b83'
    lbl = "Day 5"
    # col = '#70979b'
    # lbl = 'Day 14'
    plt.hist(t1_data, bins=20, label=lbl, edgecolor='gray', facecolor=col)
    plt.xlabel("Electrode connectivity", labelpad=-10)
    plt.ylabel("n electrodes")
    [ax.spines[spine].set_visible(False) for spine in ['top', 'right']]
    plt.xticks((0,1))
    plt.legend(loc='upper right')
    
    fig.savefig(f"{path}/connectivity_hist_{which}_3.svg", dpi=300)
    plt.show()
    
def mean_power_connectivity(data, path):
    if 'stimulated' in data.index.names:
        connectivity = data[data.index.get_level_values('stimulated')==False].groupby("el").mean()
        lbl = '1KhZ Power'
        which = 'power'
    else:
        # connectivity = data.droplevel('config')
        which = 'ampl'
        # which = 'power'
        # connectivity = data[which]/ (2/1000) # 2 uA to mA
        connectivity = data[which]
        lbl = '1KhZ Power' if which == 'power' else 'Amplitude [mV]'
        pass
    print(connectivity)
    
    mea1k = np.zeros(26400, dtype=float)
    mea1k[connectivity.index] = connectivity.values
    mea1k = mea1k.reshape(120,220)
    mea1k[mea1k==0] = np.nan
    
    cmap = plt.get_cmap('gray_r')
    # cmap = plt.get_cmap('afmhot_r')
    vmin, vmax = connectivity.values.min(), connectivity.values.max() /4
    # vmax = 25
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    print((connectivity.values>20).sum()/26400)
    plt.hist(connectivity.values, bins=20)
    plt.show()
    
    print(mea1k)
    # mask = cv2.imread(f"{path}/connectivity_ampl_aligned.png")
    # print(mask)
    # plt.imshow(mask)
    # plt.show()
    
    # (fig, ax), el_pads = draw_mea1k()
    (fig, ax), el_pads = draw_mea1k(bg='white', el_color='#eeeeee')
    fig_cb, _ = draw_mea1K_colorbar(cmap, norm, lbl)

    # x, y = np.meshgrid(np.arange(39,39+15), np.arange(9,9+15))
    # subset = np.arange(26400).reshape(120,220)[x,y].flatten()
    
    # subset = connectivity.index[:2023]
    
    
    
    # set the color of the pads to the mean power
    for i, rec in enumerate(el_pads):
        if i in connectivity.index: #and i in subset:
            rec.set_facecolor(cmap(norm(mea1k.flatten()[i])))
            # rec.set_edgecolor(cmap(norm(mea1k.flatten()[i])))
            # rec.set_linewidth(1)
            
            # if i in subset:
            #     rec.set_edgecolor('red')
    print("Saving the figure")
    print(f"{path}/connectivity_{which}.png")
    fig.savefig(f"{path}/connectivity_{which}.png", dpi=300, transparent=True, 
                bbox_inches='tight', pad_inches=0)
    fig_cb.savefig(f"{path}/colorbar_{which}.png", dpi=300, transparent=True,
                bbox_inches='tight', pad_inches=0)
    plt.show()
    
def vis_recording_pattern(path):
    # x, y = np.meshgrid(np.arange(39,39+15), np.arange(9,9+15))
    # subset = np.arange(26400).reshape(120,220)[x,y].flatten()
    
    subset = np.random.choice(np.arange(26400), size=1024)
    
    (fig, ax), el_pads = draw_mea1k()
    for i, rec in enumerate(el_pads):
        if i in subset:
            rec.set_edgecolor('red')
        
    fig.savefig(f"{path}/recording_pattern_2.svg", dpi=300, transparent=True, 
                bbox_inches='tight', pad_inches=0)
        
    plt.show()
            
    
def random_low_value_mea1k(path):
    mea1k = np.random.rand(26400) / 2.5
    mea1k = mea1k.reshape(120,220)
    
    cmap = plt.get_cmap('afmhot_r')
    # cmap = plt.get_cmap('gray_r')
    
    # draw the mea1k
    (fig, ax), el_pads = draw_mea1k(bg='white', el_color='#eeeeee')
    (fig_cb, _) = draw_mea1K_colorbar(cmap, mcolors.Normalize(vmin=0, vmax=1), 'electrode connectivity', 'horizontal')
    
    # for i, rec in enumerate(el_pads):
    #     rec.set_facecolor(cmap(mea1k.flatten()[i]))
        # rec.set_edgecolor(cmap(mea1k.flatten()[i]))
        # rec.set_linewidth(1)
        
    fig.savefig(f"{path}/empty_map.svg", dpi=300, 
                bbox_inches='tight', pad_inches=0)
    # fig_cb.savefig(f"{path}/shortcut_map_colorbar.png", dpi=300,
    #             bbox_inches='tight', pad_inches=0)
    
    
        
    plt.show()
    
    
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
            # recs.append(plt.Rectangle((x, y), 9, 9, facecolor=colors[i], 
            #                           edgecolor='none', alpha=.7))
            recs.append(plt.Rectangle((x, y), 5, 9, facecolor=colors[i], 
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

def viz_impedance(data, data3, path):
    # print(data)
    # print(data3)
    connected_els = data3[data3.ampl > 20].index
    connected_els = data.index.intersection(connected_els)  
    # print(connected_els)
    
    voltage_connected_channels = data.loc[connected_els, 'ampl']
    ohms_connected_channels = voltage_connected_channels/ (2/1e3)

    # print(data.index.drop(connected_els))
    voltage_disconnected_channels = data.loc[data.index.drop(connected_els), 'ampl']
    ohms_disconnected_channels = voltage_disconnected_channels/ (2/1e3)
    
    
    cmap = plt.get_cmap('afmhot_r')
    vmin, vmax = 0, 400
    # vmax = 25
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    (fig, ax), el_pads = draw_mea1k(bg='white', el_color='#eeeeee')
    fig_cb, _ = draw_mea1K_colorbar(cmap, norm, "Impedance [kΩ]", 'horizontal')

    # set the color of the pads to the mean power
    for i, rec in enumerate(el_pads):
        if i in ohms_connected_channels.index: #and i in subset:
            rec.set_facecolor(cmap(norm(ohms_connected_channels[i]/1000)))
            
            
    plt.show()
            
            
    (fig,ax), el_pads = draw_mea1k(bg='white', el_color='#eeeeee')
    colormaps = plt.get_cmap("afmhot_r")
    for i, rec in enumerate(el_pads):
        if i in connected_els:
            rec.set_facecolor('red')
            rec.set_edgecolor('red')
            rec.set_linewidth(1)
        # else:
        #     rec.set_edgecolor('gray')
        #     rec.set_linewidth(1)
    

    
    # violin plot
    fig, ax = plt.subplots()
    ax.violinplot([ohms_disconnected_channels,ohms_connected_channels], vert=True)
    ax.set_ylabel("Impedance [Ohm]")
    ax.set_xticks([1,2])
    ax.set_xticklabels(['Disconnected electroddes', 'Connected electrodes'])
    plt.show()
    
    # merged_data = pd.concat([ohms_connected_channels, ohms_disconnected_channels], axis=1)
    data.sort_index(inplace=True)
    data3 = data3.sort_index().rename(columns={'ampl':'ampl_extVoltage'})
    merged_data = pd.concat([data, data3], axis=1).dropna()
    print(merged_data)
    
    plt.scatter(merged_data.ampl_extVoltage, merged_data.ampl, c='black', s=2, alpha=.1)
    plt.xlabel("Response to 1KHz 25mV - in [mV]")
    plt.ylabel("Response to 1KHz 2uA - in [mV]")
    plt.show()
    # print(ohms)
    
    # plt.scatter([1]*len(ohms), ohms, c='black', s=2, alpha=.1)
    # plt.scatter([1], ohms.mean(), c='red', s=10)
    # plt.show()


def imp_violin(amp_input_connectivity, vol_input_connectivity, labels, colors, path="."):
    connected_els = vol_input_connectivity[vol_input_connectivity > 20].index
    connected_els = amp_input_connectivity.index.intersection(connected_els)
    ohms_connected_channels = amp_input_connectivity.loc[connected_els]/ (2/1e3)
    
    fig, ax = plt.subplots(figsize=(1.75, 1.15))
    fig.subplots_adjust(left=.16, right=1, top=1, bottom=.4)
    ax.yaxis.grid(True, linestyle='--', which='major', alpha=0.5)
    
    # means = all_data.mean(axis=0)
    # print(means)
    # all_data[all_data < 0.5] = np.nan
    # print(all_data)
    
    # Create violin plot
    # for i, col in enumerate(all_data.columns):
    d = ohms_connected_channels.values
    
    print(d.mean())
    print(d.std())
    print(d.mean())
    part = ax.violinplot(d/1000, showmeans=False, positions=[2], showmedians=False, showextrema=False, widths=1)

    part['bodies'][0].set_facecolor('gray')
    part['bodies'][0].set_linewidth(0.5)
    part['bodies'][0].set_edgecolor('gray')
    part['bodies'][0].set_alpha(1)
    part['bodies'][0].set_zorder(10)
        
    # ax.set_xlabel("Days post-bonding", fontsize=8)
    # ax.set_yticks(50)
    # ax.set_yticklabels((0.5,None,None,None,1),fontsize=8)
    ax.set_ylabel("Impedance [kΩ]", fontsize=8)
    ax.set_yticks((0,50,200,250,300,350))
    ax.set_yticklabels((None,'50kΩ',None,None,'300kΩ',None), fontsize=8, rotation=90, va='center', ha='right')
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(["100Hz",'1KHz','10KHz'], fontsize=8)
    ax.set_xlim(0.5,3.5)
    # ax.set_xticklabels(labels, fontsize=8)
    [ax.spines[spine].set_visible(False) for spine in ax.spines]
    
    plt.savefig(f"{path}/impednace_freq.svg", dpi=300)
    print(f"{path}/impednace_freq.svg")
    plt.show()
    

def main():
    basepath = "/mnt/SpatialSequenceLearning/Simon/impedance/"
    basepath = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance/"
    device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_neighbours3'
    # device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded_extCurrent1024_rec2'
    device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_ext1KHz_rec3'
    # device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_dry_ext1KHz_rec4'
    # device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded2_D1_1KHz_1024_rec1'
    # device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded2_D2_1KHz_1024_rec3'
    # device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded4_D0_1KHz_1024_rec1'
    
    PRV_PATH = basepath + device_name
    
    # device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_meshstim_rec1'
    # device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded_extCurrent_singleElAll'
    device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_ext1KHz_Current_singelEl_rec1'
    device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded4_D0_1KHz_1024_rec2'
    # device_name = 'device_headmount_new3EpoxyWalls/connectivity_bonded4_D5_brain_1024_rec1'
    # device_name = 'device_headmount_new3EpoxyWalls/connectivity_bonded4_D8_brain_1024_rec2'

    PATH = basepath + device_name
    # PATH = "/Users/loaloa/local_data/impedance_bonded_extCurrent_singleAll"
    
    if not os.path.exists(PATH):
        print("Path does not exist: ", PATH)
        return
    
    # random_low_value_mea1k(PATH)
    
    # fname = 'extracted_signal_all.pkl'
    fname = 'extracted_signal.pkl'
    data = pd.read_pickle(f"{PATH}/{fname}")
    
    # fname = 'extracted_signal.csv'
    # data = pd.read_csv(f"{PATH}/{fname}", index_col=[0])
    # save as pkl
    # data.to_pickle(f"{PATH}/extracted_signal2.pkl")
    
    print(data)
    
    
    # data3 = pd.read_pickle(f"{PRV_PATH}/extracted_signal.pkl")
    
    # connectivity_hist(data, PATH)
    
    mean_power_connectivity(data, PATH)
    
    # vis_recording_pattern(PATH)/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance/device_headmount_new3EpoxyWalls/impedance_bonded4_D0_1KHz_1024_rec1/connectivity_ampl.png
    # viz_stim_pattern(data)
    
    # fname = 'pad_assignment.pkl'
    # data2 = pd.read_pickle(f"{PATH}/{fname}")
    # viz_pad_pattern(data, data2, PATH)
    # draw_mea1k(bg='white')
    
    # viz_impedance(data, data3, PATH)
    
    imp_violin(data['ampl'], data3['ampl'], ['Day 1', 'Day 5'], ['#709b83', '#70979b'], path=PATH)

if __name__ == "__main__":
    main()