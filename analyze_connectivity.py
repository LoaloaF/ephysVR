import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    cmap = plt.cm.get_cmap('viridis')
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

def mean_power_connectivity(data):
    connectivity = data[data.index.get_level_values('stimulated')==False].groupby("el").mean()
    
    mea1k = np.zeros(26400, dtype=float)
    mea1k[connectivity.index] = connectivity.values
    mea1k = mea1k.reshape(120,220)
    
    cmap = plt.cm.get_cmap('viridis')
    # set 0 to black
    cmap.set_bad('black')
    mea1k[mea1k==0] = np.nan
    plt.imshow(mea1k, cmap=cmap, )
    plt.show()
    
def main():
    basepath = "/mnt/SpatialSequenceLearning/Simon/impedance/"
    basepath = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance/"
    device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_neighbours3'
    PATH = basepath + device_name
    if not os.path.exists(PATH):
        print("Path does not exist: ", PATH)
        return
    
    fname = 'stim_powers.pkl'
    data = pd.read_pickle(f"{PATH}/{fname}")
    mean_power_connectivity(data)
    # viz_stim_pattern(data)
if __name__ == "__main__":
    main()