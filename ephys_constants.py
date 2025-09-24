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

MEA_LOGGER_DEFAULT_GAIN = 512
MEA_OVERRIDE_GAIN = 7
# MEA_OVERRIDE_GAIN = None

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