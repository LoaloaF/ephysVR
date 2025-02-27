import platform
import sys
import os
import pandas as pd
import numpy as np

EPHYS_DECOMPRESS_WITH_MULTIPROCESSING = True
EPHYS_DECOMPRESS_CHUNKSIZE_S = 60 *5 # 5 minutes

SAMPLING_RATE = 20_000
MAX_AMPL_mV = 3300.
ADC_RESOLUTION = 2**10
# DEVICE_NAME = '241016_headstage03_46pad4shank'
# DEVICE_NAME_RAT006 = '241016_headstage03_46pad4shank'
DEVICE_NAME_RAT006 = '241016_MEA1K03_H1278pad4shankB5'
DEVICE_NAME_RAT011 = '241211_MEA1K06_H1278pad4shankB5'
SEED = 43

MEA_OVERRIDE_GAIN = 7

def device_paths():
    which_os = platform.system()
    user = os.getlogin()
    # print(f"OS: {which_os}, User: {user}")

    nas_dir, local_data_dir, project_dir = None, None, None
    if which_os == 'Linux' and user == 'houmanjava':
        nas_dir = "/mnt/SpatialSequenceLearning/"
        local_data_dir = "/home/houmanjava/local_data/"
        project_dir = "/home/houmanjava/VirtualReality"
    
    elif which_os == 'Linux' and user == 'vrmaster':
        nas_dir = "/mnt/SpatialSequenceLearning/"
        local_data_dir = "/home/vrmaster/local_data/"
        project_dir = "/home/vrmaster/Projects/VirtualReality/"
    
    elif which_os == "Darwin" and user == "root":
        nas_dir = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/"
        folders = [f for f in os.listdir("/Users") if os.path.isdir(os.path.join("/Users", f))]

        if "loaloa" in folders:
            local_data_dir = "/Users/loaloa/local_data/analysisVR_cache"
            project_dir = "/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/"
        elif "yaohaotian" in folders:
            local_data_dir = "/Users/yaohaotian/Downloads/Study/BME/Research/MasterThesis/code/data/analysisVR_cache"
            project_dir = "/Users/yaohaotian/Downloads/Study/BME/Research/MasterThesis/code/"
        else:
            raise ValueError("Unknown MacOS user")
    
    else:
        raise ValueError("Unknown OS or user")
    
    if not os.path.exists(nas_dir) or os.listdir(nas_dir) == []:
        msg = f"NAS directory not found: {nas_dir} - VPN connected?"
        print(msg)
        # raise FileNotFoundError(msg)
    return nas_dir, local_data_dir, project_dir

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



#045180
#006033
#6f0074
#9e3203
SHANK_BASE_COLORS = {1.0: np.array((4, 81, 128))/255,
                     3.0: np.array((0, 96, 51))/255,
                     2.0: np.array((111, 0, 116))/255,
                     4.0: np.array((158, 50, 3))/255,
}
METALLIZATION_COLOR_OFFSET = .5
















# future should use the whole mea1k el pixels
MEA1K_EL_WIDTH_MICROMETER = 5
MEA1K_EL_HEIGHT_MICROMETER = 9

def _mea1k_el_pixel_table():
    code_dir = device_paths()[2]
    if code_dir is None:
        return None
    cached_fullfname = os.path.join(code_dir, 'ephysVR', 'assets', "mea1k_el_pixel_table.pkl")
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