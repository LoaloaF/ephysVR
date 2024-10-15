import os
from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import scipy.signal as signal
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert

import multiprocessing as mp
from functools import partial

import pandas as pd
import cv2

    
    

def align_pads(path):
    # Load the image
    pads = cv2.imread(f"{path}/padlayout_aligned.png")
    print(pads.shape)
    plt.imshow(pads)
    plt.figure()
    mea1k = cv2.imread(f"{path}/connectivity_ampl.png")
    mea1k = mea1k[:,:,0] # black and white
    print(mea1k.shape)
    plt.imshow(mea1k, cmap='gray_r')
    plt.xlim()
    
    which_layout = "46pads"
    padcolor = pd.read_csv(f"./assets/padcolors_{which_layout}.csv")
    print(padcolor)
    
    mea1k_xy = []
    assignments = []
    i = 0
    for y in np.arange(17.5/4, 2100, 17.5):
        for x in np.arange(17.5/4, 3850, 17.5):
            if not (x < 300 and y<300):
                continue
            mea1k_xy.append((int(x+4.5),int(y+4.5)))
            
            col = (pads[int(y+4.5),int(x+4.5)])
            pad = padcolor[(padcolor['r'] == col[0]) & (padcolor['g'] == col[1]) & (padcolor['b'] == col[2])]
            # print all infos
            print(i, x, y, end=' ')
            if not pad.empty:
                print(col, pad)
                pad = pad.copy().reset_index().rename({"index":"pad"})
                if pad.shape[0] > 1:
                    pad = pad.iloc[0:1]
                pad.loc[:,'el'] = i
                assignments.append(pad)
                
                rect = plt.Rectangle((x, y), 9, 9, facecolor=col/255, edgecolor='black')
                plt.gca().add_patch(rect)
            
            print()
            i += 1
            
            # if len(assignments) >20:
            #     break
    
    
    assignments = pd.concat(assignments)
    print(assignments)            
            
    # plt.scatter(*zip(*mea1k_xy), c='red', s=1)
            
    plt.show()
    
    
def main():
    basepath = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance/"
    device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded_extCurrent_singleElAll'
    device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded_extCurrent1024_rec2'
    PATH = basepath + device_name
    print(PATH)
    if not os.path.exists(PATH):
        print("Path does not exist: ", PATH)
        exit()

    align_pads(PATH)
    
if __name__ == '__main__':
    main()