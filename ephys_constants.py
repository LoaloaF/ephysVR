import platform
import sys
import os

SAMPLING_RATE = 20_000
MAX_AMPL_mV = 3300.
ADC_RESOLUTION = 2**10
DEVICE_NAME = '241016_headstage03_46pad4shank'
SEED = 43

def DEFAULT_PATHS():
    which_os = platform.system()
    user = os.getlogin()
    
    if which_os == 'Linux' and user == 'houmanjava':
        nas_dir = "/mnt/SpatialSequenceLearning/"
        local_data_dir = "/home/houmanjava/local_data/"
        code_dir = "/home/houmanjava/meatesting/ephysVR"
    
    elif which_os == 'Linux' and user == 'vrmaster':
        nas_dir = "/mnt/SpatialSequenceLearning/"
        local_data_dir = "/home/vrmaster/local_data/"
        code_dir = "/home/vrmaster/Projects/VirtualReality/ephysVR"
    
    elif which_os == "Darwin" and user == "loaloa":
        nas_dir = "/Volumes/SpatialSequenceLearning/"
        local_data_dir = "/Users/loaloa/local_data/"
        code_dir = "/Users/loaloa/homedataAir/ratvr/VirtualReality/ephysVR"
    
    else:
        nas_dir, local_data_dir, code_dir = None, None, None
    return nas_dir, local_data_dir, code_dir

NAS_DIR, LOCAL_DATA_DIR, CODE_DIR = DEFAULT_PATHS()