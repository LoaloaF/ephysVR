import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import cv2

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

def mean_power_connectivity(data, path):
    if 'stimulated' in data.index.names:
        connectivity = data[data.index.get_level_values('stimulated')==False].groupby("el").mean()
        lbl = '1KhZ Power'
        which = 'power'
    else:
        # connectivity = data.droplevel('config')
        which = 'ampl'
        # connectivity = data[which]/ (2/1000) # 2 uA to mA
        connectivity = data[which]
        lbl = '1KhZ Power' if which == 'power' else 'Amplitude [mV]'
        pass
    print(connectivity)
    
    mea1k = np.zeros(26400, dtype=float)
    mea1k[connectivity.index] = connectivity.values
    mea1k = mea1k.reshape(120,220)
    mea1k[mea1k==0] = np.nan
    
    # plt.hist(mea1k.flatten(), bins=100)
    # plt.show()
    
    # cmap = plt.get_cmap('gray_r')
    cmap = plt.get_cmap('viridis')
    vmin, vmax = connectivity.values.min(), connectivity.values.max()/4
    # vmax = 25
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    

    # mask = cv2.imread(f"{path}/connectivity_ampl_aligned.png")
    # print(mask)
    # plt.imshow(mask)
    # plt.show()
    
    (fig, ax), el_pads = draw_mea1k()
    # (fig, ax), el_pads = draw_mea1k(bg='white', el_color='#dddddd')
    fig_cb, _ = draw_mea1K_colorbar(cmap, norm, lbl)

    # x, y = np.meshgrid(np.arange(39,39+15), np.arange(9,9+15))
    # subset = np.arange(26400).reshape(120,220)[x,y].flatten()
    
    # subset = connectivity.index[:2023]
    
    
    
    # set the color of the pads to the mean power
    for i, rec in enumerate(el_pads):
        if i in connectivity.index: #and i in subset:
            rec.set_facecolor(cmap(norm(mea1k.flatten()[i])))
            
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
            
    
    
    
    
def draw_mea1K_colorbar(cmap, norm, lbl):
    cbar_fig, cbar_ax = plt.subplots(figsize=(1.4, 2100/300))
    cbar_fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    cbar_fig.subplots_adjust(right=0.25)
    cbar = mcolorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
    cbar.set_label(lbl)
    return cbar_fig, cbar

def draw_mea1k(bg='black', el_color='#222222'):
    fig, ax = plt.subplots(figsize=(3850/300, 2100/300), facecolor='none')
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    # fig.patch.set_facecolor('black')
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
    
def main():
    basepath = "/mnt/SpatialSequenceLearning/Simon/impedance/"
    basepath = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance/"
    device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_neighbours3'
    device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded_extCurrent1024_rec2'
    device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_ext1KHz_rec3'
    device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_dry_ext1KHz_rec4'
    
    device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_meshstim_rec1'

    # device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded_extCurrent_singleElAll'
    PATH = basepath + device_name
    # PATH = "/Users/loaloa/local_data/impedance_bonded_extCurrent_singleAll"
    
    if not os.path.exists(PATH):
        print("Path does not exist: ", PATH)
        return
    
    # fname = 'ext_current_powers.pkl'
    fname = 'extracted_signal.pkl'
    data = pd.read_pickle(f"{PATH}/{fname}")
    mean_power_connectivity(data, PATH)
    
    
    # vis_recording_pattern(PATH)
    
    # viz_stim_pattern(data)
    # draw_mea1k(bg='white')


if __name__ == "__main__":
    main()