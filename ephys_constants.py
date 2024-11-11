import platform
import sys
import os
import pandas as pd
import numpy as np

SAMPLING_RATE = 20_000
MAX_AMPL_mV = 3300.
ADC_RESOLUTION = 2**10
DEVICE_NAME = '241016_headstage03_46pad4shank'
SEED = 43

def _delfault_paths():
    which_os = platform.system()
    user = os.getlogin()
    print(f"OS: {which_os}, User: {user}")
    
    if which_os == 'Linux' and user == 'houmanjava':
        nas_dir = "/mnt/SpatialSequenceLearning/"
        local_data_dir = "/home/houmanjava/local_data/"
        code_dir = "/home/houmanjava/meatesting/ephysVR"
    
    elif which_os == 'Linux' and user == 'vrmaster':
        nas_dir = "/mnt/SpatialSequenceLearning/"
        local_data_dir = "/home/vrmaster/local_data/"
        code_dir = "/home/vrmaster/Projects/VirtualReality/ephysVR"
    
    elif which_os == "Darwin" and user == "root":
        nas_dir = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/"
        local_data_dir = "/Users/loaloa/local_data/"
        code_dir = "/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/ephysVR"
    
    else:
        nas_dir, local_data_dir, code_dir = None, None, None
        raise ValueError("Unknown OS or user")
    
    if not os.path.exists(nas_dir):
        msg = f"NAS directory not found: {nas_dir} - VPN connected?"
        raise FileNotFoundError(msg)
    return nas_dir, local_data_dir, code_dir

NAS_DIR, LOCAL_DATA_DIR, CODE_DIR = _delfault_paths()

def _mea1k_el_center_table_micrometer():
    el_i = 0
    all_els = {}
    for y in np.arange(17.5/4, 2100, 17.5):
        for x in np.arange(17.5/4, 3850, 17.5):
            all_els[el_i] = (y, x)
            el_i += 1
    mea1k = pd.DataFrame(all_els).T
    mea1k.columns = ['y', 'x']
    mea1k.index.name = 'el'
    return mea1k

MEA1K_EL_CENTER_TABLE_MICROMETER = _mea1k_el_center_table_micrometer()
MEA1K_EL_CENTER_TABLE_PIXEL = MEA1K_EL_CENTER_TABLE_MICROMETER.copy().astype(np.uint16)
MEA1K_EL_TABLE_PIXEL_YX_IDX = MEA1K_EL_CENTER_TABLE_PIXEL.reset_index().set_index(['y', 'x'])




SHANK_BASE_COLORs = {1.0: (1.0, 0.61, 0.101),
                     3.0: (0.988, 1.0, 0.101),
                     2.0: (0.305, 1.0, 0.101),
                     4.0: (0.529, 0.102, 1),}
METALLIZATION_COLOR_OFFSET = .5













# future should use the whole mea1k el pixels
MEA1K_EL_WIDTH_MICROMETER = 5
MEA1K_EL_HEIGHT_MICROMETER = 9

def _mea1k_el_pixel_table():
    cached_fullfname = os.path.join(CODE_DIR, 'assets', "mea1k_el_pixel_table.pkl")
    if os.path.exists(cached_fullfname):
        return pd.read_pickle(cached_fullfname)
    
    all_el_pixels = []
    for el_i, (y, x) in MEA1K_EL_CENTER_TABLE_MICROMETER.iterrows():
        all_y = np.arange(y - MEA1K_EL_HEIGHT_MICROMETER/2, 
                          y + MEA1K_EL_HEIGHT_MICROMETER/2, 1)
        all_x = np.arange(x - MEA1K_EL_WIDTH_MICROMETER/2, 
                          x + MEA1K_EL_WIDTH_MICROMETER/2, 1)
        # stack the x and y coordinates to get a 2D grid, then collapse 2D to 1D
        el_i_yx = np.stack(np.meshgrid(all_y, all_x)).reshape(2, -1).round().astype(np.uint16)
        multiindex = pd.MultiIndex.from_arrays(el_i_yx, names=['y', 'x'])
        all_el_pixels.append(pd.Series([el_i]*len(el_i_yx.T), index=multiindex, name='el'))
    pd.to_pickle(pd.concat(all_el_pixels), cached_fullfname)
    return pd.concat(all_el_pixels)
    
MEA1K_EL_2D_TABLE_PIXEL = _mea1k_el_pixel_table()