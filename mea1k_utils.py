import os
import sys
from glob import glob
import shutil
import time
import datetime
import threading

import pandas as pd
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

# import parent dir with general modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# copy over base dir, but not single files in it except the procssed subidr, what in there all gets copied over
def cp_rec_dir_to_implant_dir(rec_dir, implant_name, ):
    nas_dir = device_paths()[0]
    implant_dir = os.path.join(nas_dir, f"devices/implant_devices/{implant_name}/recordings")
    
    # create the implant dir if it doesn't exist
    if not os.path.exists(implant_dir):
        raise FileNotFoundError(f"Implant directory {implant_dir} does not exist. Please create it first.")
    
    print(f"Copying recording directory:\n{rec_dir} to implant directory {implant_dir}...")
    # recreate rec_dir in implant_dir
    implant_dir_new = os.path.join(implant_dir, os.path.basename(rec_dir))
    if not os.path.exists(implant_dir_new):
        os.makedirs(implant_dir_new)
    
    # # copy over the rec_dir to implant dir, but prossed subdir only, not the single files in rec_dir
    s = os.path.join(rec_dir, 'processed')
    d = os.path.join(implant_dir_new, 'processed')
    shutil.copytree(s, d, dirs_exist_ok=False)
    print("Now in: ", implant_dir_new)

def load_impedance_data(rec_dir, nas_dir, subdir, implant_name=''):
    """Load and merge impedance data with implant mapping."""
    fullfname = os.path.join(nas_dir, "devices", "implant_devices", implant_name, "bonding", 
                            f"bonding_mapping_{implant_name}.csv")
    if os.path.exists(fullfname):
        print(f"Loading implant mapping from {fullfname}...")
        implant_mapping = pd.read_csv(fullfname)
    else:
        print(f"Implant mapping file not found: {fullfname}")
        implant_mapping = None

    full_recdir = os.path.join(nas_dir, subdir, rec_dir)
    if not os.path.exists(full_recdir):
        raise FileNotFoundError(f"Recording directory {full_recdir} does not exist.")
    aggr_df = pd.read_csv(os.path.join(full_recdir, "processed", "all_impedance.csv"))
    aggr_df.rename(columns={"electrode": "mea1k_el"}, inplace=True)
    
    if implant_mapping is not None:
        # Merge once with implant mapping
        aggr_df = pd.merge(aggr_df, implant_mapping[['mea1k_el', 'shank_id', 'mea1k_connectivity', 'pad_metal']], 
                           on='mea1k_el', how='left')
    else:
        aggr_df['shank_id'] = None
        aggr_df['mea1k_connectivity'] = None
        aggr_df['pad_metal'] = None
        
    aggr_df.rename(columns={'mea1k_el': 'electrode'}, inplace=True)
    return aggr_df, implant_mapping, full_recdir