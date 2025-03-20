import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import glob
from matplotlib import pyplot as plt

from CustomLogger import CustomLogger as Logger

from mea1k_modules.mea1k_raw_preproc import get_raw_implant_mapping
from mea1k_modules.mea1k_visualizations import draw_interconnect_pads
from mea1k_modules.mea1k_visualizations import draw_mea1k


implant_name = "250205_MEA1K03_H1278pad4shankB5"    
implant_name = "241211_MEA1K06_H1278pad4shankB5"    

path = f"/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/devices/implant_devices/{implant_name}/bonding/imp_rec_configs"

def checkconfigs(path):
    pad_ids = [-1]
    colors = {}
    for f in sorted(glob.glob(os.path.join(path, "*.cfg"))):
        pad_id = int(f[-8:-4])
        if pad_id - pad_ids[-1] != 1:
            print(f"Missing pad: {pad_ids[-1]+1}")
        pad_ids.append(pad_id)
        colors[pad_id] = (1, 0, 0)
        
    (fig, ax), el_rects = draw_mea1k()
    mapping = get_raw_implant_mapping(implant_name=implant_name)
    draw_interconnect_pads(mapping, colors, draw_on_ax=ax)
    plt.show()
    
if __name__ == "__main__":
    checkconfigs(path)