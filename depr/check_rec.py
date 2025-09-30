import os
import sys

# to import from parent dir
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from CustomLogger import CustomLogger as Logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import ephys_constants as C
from mea1k_modules.mea1k_raw_preproc import read_raw_data
from mea1k_modules.mea1k_raw_preproc import read_stim_DAC
from mea1k_modules.mea1k_raw_preproc import get_recording_implant_mapping

from mea1k_modules.mea1k_visualizations import vis_shank_traces, viz_mea1k_config, draw_interconnect_pads
from mea1k_modules.mea1k_visualizations import draw_mea1k
from mea1k_modules.mea1k_visualizations import draw_mea1K_colorbar


def check_recording(path, fname, implant_name, time_slice=slice(None)):
    L = Logger()
    data = read_raw_data(path, fname, convert2uV=True,
                        subtract_dc_offset=True, 
                        col_slice=time_slice,
    )
        
    mapping = get_recording_implant_mapping(path, fname, implant_name=implant_name,
                                            drop_non_bonded=False,)
    mask = (mapping.shank_id == 3).values #& (mapping.depth<9200).values & (mapping.depth>8800).values
    data = data[mask]
    mapping = mapping[mask]
    print(data.shape)
    print(mapping)
    # viz_mea1k_config(mapping, stim_mea1k_el=stimulated.electrode[stimulated.stim].item())
    vis_shank_traces(data, mapping, scaler=1/2)#, uVrange=470_000)
    
    plt.subplot(2, 1, 1)
    plt.plot(data.T)
    plt.subplot(2, 1, 2, sharex=plt.gca())              
    plt.plot(read_stim_DAC(path, fname))
    plt.show()
    
def main():
    L = Logger()
    L.init_logger(None, None, "DEBUG")
    L.logger.debug("Starting in vivo impedance analysis")
    
    nas_dir = device_paths()[0]
    animal_name = "rYL010"
    implant_name = "250205_MEA1K03_H1278pad4shankB5"
    
    # path = f"{nas_dir}/devices/implant_devices/{implant_name}/recordings"
    
    session_name = "2025-03-25_15-28_rYL010_P1000_MotorLearningStop_3min"
    path = f"{nas_dir}/RUN_{animal_name}/{animal_name}_P1000/{session_name}/"
    print(path)
    
    check_recording(path, 'ephys_output.raw.h5', implant_name, time_slice=slice(100_000, 120_000))
    
if __name__ == "__main__":
    main()
