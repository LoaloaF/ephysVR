
# %%
%load_ext autoreload
%autoreload 2

# %%
import os
import sys
from glob import glob
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join('..',))
sys.path.append(os.path.join('..', '..'))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

import mea1k_modules.mea1k_visualizations as vis
from mea1k_connectivity_scripts.analyze_extSine_connectivity import extract_connectivity, vis_connectivity, create_implant_dir, get_connectivity_data
from mea1k_connectivity_scripts.analyze_shortcut_stim import plot_amplitude_ridges, connected_islands

from mea1k_utils import load_impedance_data as load_impedance_data

nas_dir = device_paths()[0]

# %%
L = Logger()
L.init_logger(None, None, "WARNING")











# %%
headstage_name = "MEA1K27"

subdir = f"devices/headstage_devices/{headstage_name}/recordings"
nas_dir = device_paths()[0]


rec_dir = "2026-10-12_13.00.58_2x8x56umP_PT4_TileShorts"
rec_dir = "2026-10-12_17.06.30_2x8x56umP_PT4_TileShorts"
rec_dir = "2026-10-13_13.53.25_2x8x56umP_PT4_TileShorts"

full_recdir = os.path.join(nas_dir, subdir, rec_dir)
connected_islands(full_recdir, output_dir=os.path.join(full_recdir, "processed"), 
                  conn_thr=.5, output_fname=f"connected_islands.png",)
                #   draw_mea_args={"mapping":conn, "el_color":"#595959", 'cmap_scaler':2})

plot_amplitude_ridges(full_recdir, bins=60, n_groups=8, shorted_thresh=0.8,
                      min_stim_uV=None, dpi=100, rebuilt_graph=False)

# %%


headstage_name = "MEA1K27"

subdir = f"devices/headstage_devices/{headstage_name}/recordings"
nas_dir = device_paths()[0]

rec_name = f'2026-10-13_16.16_test10_100k_36umP_PT1_20mVFPGAExt'
rec_name = f'2026-10-15_13.05_test10_100k_36umP_PT1_20mVFPGAExt'
rec_name = f'2026-10-15_13.13_test10_100k_36umP_PT1_reSoldered_20mVFPGAExt'
rec_name = f'2026-10-15_13.21_test10_100k_36umP_PT2_reSoldered_20mVFPGAExt'
rec_name = f'2026-10-15_13.27_test10_100k_36umP_PT3_reSoldered_20mVFPGAExt'
rec_name = f'2026-10-15_14.21_test10_100k_36umP_PT4_reSoldered_20mVFPGAExt'

rec_name = f'2026-10-15_14.21_test10_100k_36umP_PT4_reSoldered_20mVFPGAExt'

# rec_name = f'2026-10-16_12.01_test10_100k_36umP_PT4_reSoldered_10mVFPGA' # soldering joint broke -.-

# rec_name = f'2026-10-21_14.45_test11_100k_36umP_PT1_10mVFPGA' 
# rec_name = f'2026-10-22_13.56_test11_100k_36umP_PT2_10mVFPGA' 
# rec_name = f'2026-10-22_13.58_test11_100k_36umP_PT3_10mVFPGA' 
# rec_name = f'2026-10-22_15.21_test11_100k_36umP_PT4_10mVFPGA' 

# rec_name = f'2026-10-25_16.47_test11_100k_36umP_PT5_10mVFPGA' # wrong side tightened...

# rec_name = f'2026-10-25_16.58_test11_100k_36umP_PT6_10mVFPGA' # untightened an retightened, new balance

# rec_name = f'2026-10-25_17.03_test11_100k_36umP_PT7_10mVFPGA' # 

# rec_name = f'2026-10-28_14.52_test11_100k_36umP_PT7_10mVFPGA' # 

rec_name = f'2026-11-01_20.09_test12_100k_36umP_56umPE_PT1_10mVFPGA' # resoldered wire, wasn never attached... but bottom borke off? almost no signal
rec_name = f'2026-11-01_20.09_test12_100k_36umP_56umPE_PT1_10mVFPGA' # 
rec_name = f'2026-11-01_20.17_test12_100k_36umP_56umPE_PT2_10mVFPGA' # bottom device only 4-5 electrodes? til check again

full_path = os.path.join(nas_dir, subdir, rec_name)

# extract_connectivity(full_path, input_ampl_mV=10, n_samples=8000, gain=7, debug=False, )
# vis_connectivity(full_path, input_ampl_mV=10, cmap_scaler=1)

conn = get_connectivity_data(full_path)
vis.draw_mea1k(mapping=conn, fast=True,  )
plt.show()
plt.hist(conn['mea1k_connectivity'], bins=50)
plt.yscale('log')
plt.show()

print(rec_name)
# %%



headstage_name = "MEA1K27"

subdir = f"devices/headstage_devices/{headstage_name}/recordings"
nas_dir = device_paths()[0]

rec_dirs = [
    # "2026-10-15_15.28_test10_100k_36umP_PT4_resoldered_Imp",
    # "2026-10-15_15.40_test10_100k_36umP_PT4_resoldered_Imp",
    # "2026-10-15_15.55_test10_100k_36umP_PT4_resoldered_Imp", # super weird looked good for one run -.-
    
    # "2026-10-22_14.01_test11_100k_36umP_PT3_Imp", # puhh.. bad first i see bimodel good+bad mixed imp -.- 35 minutes wasn't good?
    # "2026-10-22_15.44_test11_100k_36umP_PT4_Imp",
    # "2026-10-22_19.16_test11_100k_36umP_PT4_Imp",
    
    # "2026-10-22_22.44_test11_100k_36umP_PT4_Imp",
    # "2026-10-22_23.55_test11_100k_36umP_PT4_Imp",
    
    # "2026-10-24_15.44_test11_100k_36umP_PT4_Imp",
    
    # "2026-10-25_17.14_test11_100k_36umP_PT7_Imp",
    
    # "2026-10-26_13.45_test11_100k_36umP_PT7_Imp",

    # "2026-10-26_13.57_test11_100k_36umP_PT7_Imp",
    
    # "2026-10-28_14.57_test11_100k_36umP_PT7_Imp",
    
    # "2026-10-31_17.04_test12_100k_36umP_56umPE_PT1_Imp_oneCorner",   # without GND cable connected, but not connected to ring GND apparnetly, all max Imp
    # "2026-11-01_08.50_test12_100k_36umP_56umPE_PT1_Imp_oneCorner",    # with cable still not connected
    # "2026-11-02_07.11_test12_100k_36umP_56umPE_PT2_Imp",
    "2026-11-02_07.22_test12_100k_36umP_56umPE_PT2_Imp",
    # "2026-11-02_08.18_test12_100k_36umP_56umPE_PT2_Imp_bottomCorner"
]

labels = [
    # "PT4",
    # "PT4.1",
    # "PT3",
    # "PT4", 
    # "PT4+2h",
    # "PT4+6h",
    # "PT4+2d",
    # "PT7",
    # "PT7+1d",
    # "PT7+4d",
    
    # 'test12 PT1 one corner',
    # 'test12 PT1 one corner',
    # 'test12 PT2',
    'test12 PT2',
    # 'test12 PT2 one corner',
]

data = []
for rec_dir in rec_dirs:
    aggr_df, implant_mapping, full_recdir = load_impedance_data(rec_dir, nas_dir, subdir, )
    # aggr_df = aggr_df[aggr_df['stim_unit'] == 0]
    vis.plot_impedance_analysis(rec_dir, aggr_df, log_scale=True,
                                ylims=(1_000, 1_800_000), skip_bottom=False,
                                color_by_metal=False, use_imp_connectivity=True)
    plt.show()
    data.append(aggr_df)

vis.compare_impedance(data, labels, show_scatter=True, log_scale=True)
# %%


headstage_name = "MEA1K27"

subdir = f"devices/headstage_devices/{headstage_name}/recordings"
nas_dir = device_paths()[0]

rec_dir = f'2026-10-30_18.24.30_test12_100k_36umP_56umPE_PT1_TileShorts'
rec_dir = f'2026-10-31_21.47.17_test12_100k_36umP_56umPE_PT1_TileShorts' # using ssequential stim 
rec_dir = f'2026-11-01_20.22.38_test12_100k_36umP_56umPE_PT2_TileShorts' # using ssequential stim 


conn = get_connectivity_data(os.path.join(nas_dir, subdir, f'2026-11-01_20.17_test12_100k_36umP_56umPE_PT2_10mVFPGA' ))
full_recdir = os.path.join(nas_dir, subdir, rec_dir)
connected_islands(full_recdir, output_dir=os.path.join(full_recdir, "processed"), 
                  conn_thr=.5, output_fname=f"connected_islands.png",
                  draw_mea_args={"mapping":conn, "el_color":"#595959", 'cmap_scaler':1})
# )
# plot_amplitude_ridges(full_recdir, bins=60, n_groups=8, shorted_thresh=0.5,
#                       min_stim_uV=None, dpi=100, rebuilt_graph=False)


# %%
