# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import sys
from glob import glob
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join('.', ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

import mea1k_modules.mea1k_visualizations as vis
from mea1k_connectivity_scripts.analyze_extSine_connectivity import extract_connectivity, vis_connectivity, create_implant_dir, get_connectivity_data
from mea1k_connectivity_scripts.analyze_shortcut_stim import plot_amplitude_ridges, connected_islands, plot_3x3_short_results

from mea1k_utils import cp_rec_dir_to_implant_dir
from mea1k_utils import load_impedance_data as load_impedance_data

nas_dir = device_paths()[0]

# %%
L = Logger()
L.init_logger(None, None, "WARNING")

# %% [markdown]
# ### Assembled on Wednesday: 15.7, 14 shanks, called _ReEtched14ShankReGoldPlated
# * Used Chip24, regoldplated on the day before by Alexei, looked good visually
# * there was a notable assymmetry though from top and bottom half, top was scraped a long time ago, bottom one day before goldplating, not sure which one was higher
# * Used Baran's reetched wafer for the first time, called it Batch7 (although a bit unsure if its the correct number)
# * Initially wanted to use MEAE23, promising before, but redecided to use Chip24
# * PEG & Silk proper pickup, put a bit to much spring force on them, improve next time. Main issue: dibers didn't bundle soo well.
# * Checked PT0, but no connection. But with low pressure signal started showing up. Which metal?
#
#
# #### Side notes:
# * Generally current mode stim is very stable now, have not had issues with it in the last week, for Chip23 too
# * intersting observation on PT impmedance: it looks like amplitude did increase in quite a few cases, ie some saturation, then high imp, on order of 100ms. What could this be?
#
#
#

# %%
headstage_name = "MEA1K24"
bonding_date = '260715'
HEADSTAGE_DEVICE_NAME = 'MEA1K24'
ELECTRODE_DEVICE_NAME = 'S1688pad14shank'
batch = 7
IMPLANT_DEVICE_NAME = f"{bonding_date}_{HEADSTAGE_DEVICE_NAME}_{ELECTRODE_DEVICE_NAME}B{batch}"
implant_name = IMPLANT_DEVICE_NAME

subdir = f"devices/headstage_devices/{headstage_name}/recordings"
impl_subdir = f"devices/implant_devices/{IMPLANT_DEVICE_NAME}/recordings"
nas_dir = device_paths()[0]

# %%
rec_names = [
    f'2026-07-15_11.21_ReEtched14ShankReGoldPlated_VRef15_PT0',
    f'2026-07-15_11.28_ReEtched14ShankReGoldPlated_VRef15_PT0',
    f'2026-07-15_11.32_ReEtched14ShankReGoldPlated_VRef15_PT0',
    f'2026-07-15_11.38_ReEtched14ShankReGoldPlated_VRef15_PT1',
    f'2026-07-15_12.31_ReEtched14ShankReGoldPlated_VRef15_PT2',
    f'2026-07-15_13.59_ReEtched14ShankReGoldPlated_VRef15_PT3',
]

data = []
for rec_name in rec_names:
    full_path = os.path.join(nas_dir, impl_subdir, rec_name)
    # extract_connectivity(full_path, input_ampl_mV=10, n_samples=8000, debug=False, )
    # vis_connectivity(full_path, input_ampl_mV=10, cmap_scaler=1)
    
    conn = get_connectivity_data(full_path)
    conn = conn[conn['pad_metal'] == 2]
    vis.draw_mea1k(mapping=conn, fast=True,  )
    data.append(conn)
    print(rec_name)
    plt.show()
    
    # create_implant_dir(full_path, nas_dir, HEADSTAGE_DEVICE_NAME, IMPLANT_DEVICE_NAME)


# %%
# compare over runs
vis.compare_many_connectivities(data, ['PT0_0', 'PT0_1', 'PT0_2', 'PT1', 'PT2', 'PT3'], color_metal=True, )



# %%
rec_names = [
    "2026-07-15_12.04_ReEtched14ShankReGoldPlated_Imp1_PT1_LC",
    "2026-07-15_12.35_ReEtched14ShankReGoldPlated_Imp2_PT2_LC",
    "2026-07-15_14.04_ReEtched14ShankReGoldPlated_Imp3_PT3_LC",
]

data = []
for rec_name in rec_names:
    aggr_df, implant_mapping, full_recdir = load_impedance_data(rec_name, nas_dir, impl_subdir, implant_name)
    # aggr_df = aggr_df[aggr_df['pad_metal'] == 2]
    vis.plot_impedance_analysis(rec_name, aggr_df, log_scale=True, ylims=(1_000, 2_000_000), 
                                skip_bottom=True, color_by_metal=True )
    data.append(aggr_df)
    # vis.plot_ext_current_comparison(aggr_df, diff_max=2_000_000, title="")
    plt.show()
    

# %%
vis.compare_impedance(data, ['PT1', 'PT2', 'PT3'], color_metal=True, )

# %%
plt.figure(figsize=(20, 20), dpi=200)
for data in [aggr_df_PT1, aggr_df_PT2, aggr_df_PT3]:
    # Spatial CMOS distribution (never log)
    heatmap_x = (data.electrode % 220) * 17.5 + 17.5/4
    heatmap_y = (data.electrode // 220) * 17.5 + 17.5/4
    plt.scatter(heatmap_x, heatmap_y, alpha=0.5, s=10)
# invert y axis to match the physical layout of the electrodes
plt.gca().invert_yaxis()
# fix aspect ratio to 1:1
plt.gca().set_aspect('equal', adjustable='box')

# %%
# def phase_impedance_comparison(aggr_df1, aggr_df2, aggr_df3=None, title1=None, title2=None, title3=None, color_by='phase'):

# vis.phase_impedance_comparison(aggr_df_PT1, aggr_df_PT2, )
# vis.compare_impedance_simple(aggr_df_PT2, aggr_df_PT3, )
# how many overlaps
display(aggr_df_PT1[['electrode', 'connectivity']].sort_values('connectivity'))
display(aggr_df_PT2[['electrode', 'connectivity']].sort_values('connectivity'))
display(aggr_df_PT3[['electrode', 'connectivity']].sort_values('connectivity'))
# print("Number of overlapping electrodes between PT1 and PT2:")
# print(aggr_df_PT1.electrode.isin(aggr_df_PT2.electrode).sum())
# print("Number of overlapping electrodes between PT2 and PT3:")
# print(aggr_df_PT2.electrode.isin(aggr_df_PT3.electrode).sum())

# plt.figure(figsize=(8*3, 6*3), dpi=500)
# plt.plot(np.ones(aggr_df_PT1.shape[0]), aggr_df_PT1.electrode, 'o', label='PT1')
# plt.hlines(aggr_df_PT1.electrode, 0, 3,  label='PT1', alpha=0.5, linewidth=0.5)
# plt.plot(np.ones(aggr_df_PT2.shape[0])*2, aggr_df_PT2.electrode, 'o', label='PT2')
# plt.hlines(aggr_df_PT2.electrode, 0, 3,  label='PT2', color='orange', alpha=0.5, linewidth=0.5)
# plt.plot(np.ones(aggr_df_PT3.shape[0])*3, aggr_df_PT3.electrode, 'o', label='PT3')
# plt.hlines(aggr_df_PT3.electrode, 0, 3,  label='PT3', color='green', alpha=0.5, linewidth=0.5)





# check connectivity overlap between PT1, PT2, PT3
conn1 = pd.read_csv(os.path.join(nas_dir, subdir, "2026-07-15_11.38_ReEtched14ShankReGoldPlated_VRef15_PT1", 'processed', 'extr_connectivity.csv'))
conn2 = pd.read_csv(os.path.join(nas_dir, subdir, "2026-07-15_12.31_ReEtched14ShankReGoldPlated_VRef15_PT2", 'processed', 'extr_connectivity.csv'))
conn3 = pd.read_csv(os.path.join(nas_dir, subdir, "2026-07-15_13.59_ReEtched14ShankReGoldPlated_VRef15_PT3", 'processed', 'extr_connectivity.csv'))

# join the 3 tables based on el, and then plot three point lines for each electrode, with x-axis as PT1, PT2, PT3 and y-axis as connectivity
conn_merged = pd.merge(conn1, conn2, on='el', suffixes=('_PT1', '_PT2'))
conn_merged = pd.merge(conn_merged, conn3, on='el')
conn_merged.rename(columns={'connectivity': 'connectivity_PT3'}, inplace=True)

# how many electrodes overlap that were high connectivity than 0.3
print("Number of overlapping electrodes with connectivity > 0.3 between PT1 and PT2, total number in PT1")
print(((conn_merged.connectivity_PT1 > 0.3) & (conn_merged.connectivity_PT2 > 0.3)).sum(), (conn_merged.connectivity_PT1>.3).sum())
print("Number of overlapping electrodes with connectivity > 0.3 between PT2 and PT3, total number in PT2")
print(((conn_merged.connectivity_PT2 > 0.3) & (conn_merged.connectivity_PT3 > 0.3)).sum(), (conn_merged.connectivity_PT2>.3).sum())
print("Number of overlapping electrodes with connectivity > 0.3 between PT1 and PT3, total number in PT3")
print(((conn_merged.connectivity_PT1 > 0.3) & (conn_merged.connectivity_PT3 > 0.3)).sum(), (conn_merged.connectivity_PT3>.3).sum())

# %%
rec_dir = "2026-07-15_20.03.21_ReEtched14ShankReGoldPlated_SingleShorts_PT3"
# rec_dir = "2026-07-15_18.26.30_ReEtched14ShankReGoldPlated_SingleShorts_PT3"
full_recdir = os.path.join(nas_dir, impl_subdir, rec_dir)
connected_islands(full_recdir, output_dir=os.path.join(full_recdir, "processed"), 
                  conn_thr=.9, output_fname=f"connected_islands.png",
                  draw_mea_args={"mapping":conn, "el_color":"#595959", 'cmap_scaler':2})

plot_amplitude_ridges(full_recdir, shorted_thresh=.9, n_groups=1, min_stim_uV=1000, rebuilt_graph=True)


# %%

# %% [markdown]
#
#
# ### MEA24 history:
#
# * Early May: I think this was the **Bond1**, 260501_MEA1K24_S1688pad14shankB5. This was a wafer from Shubham, hist first or second? Put silk and everything, Had hope. Did a lot of shortcut stim on tiles before, recorded 50Hz voltage patterns, kept dry for long. Pressure high, but not very high if i remember correctly. Imp not good. Especially metal2 not good, later found reason for it. And deterioarted over time. Measured this like crazy, so much, tried to understand it. Single shank 13 and 14. Found suspicious shorts that now i would say are connected to GND ringnode. Deterioaration was linked to to then silver electrode, and zeroDACcode drifting/ being off. Wrote DAC drift code. Or cases where LC mode + amplitude 30, super high currents. In the end (22nd-31st of May) got current sense circuit to work with ALexei. FOund that tiny DAC shifts influence measurements a lot. Imp can have constant value offset (good case), fully match (perfect) or be completely off (no match)  ![alt text](image-2.png) ![alt text](image-3.png) 
#
# * 10.6. - 16.6 Extremly unsure what this is, called **Bond3** in recordings dir, looks like i realigned this a bunch of times. Looks very wet in voltage map. No idea what electrodes these are, must be wafer 5 again?. Did i try to recover those? In ultrasonic cleaner? Didn't work , remember that ![alt text](image-4.png).
#
# * 17.6 New bond, now called **Bond4**, last one till now. **scraped gold off of top part** of chip to check if lower gold can help (it didn't). Connected the wrong cable with this one -.- one day lost, next day, found capctitive connection for the first time, after doing it, but for the half with pillars. Single MEA electrodes connected. No silk on those electrodes. Tried to do proper depth prifle. Used interconnects with some error, Shubham said this wafer (6) is not that good, but Baran put PEDOT. With high pressure, found some electrodes also connceted on scraped side. But, unfortuatnely, over all pressures, imp stayed high... Used 00-10 for the first time, thought needs lower pressure, but not really i think. Clean surface is the key to an early bond. Probably not electrodes, rigth? SHould have checked better.. Also struggled wi    th StimUnit calibration, had to manuallt shift by 20 DAC units (SC) that aligned external and internal current somewhat, but still results looked quite bad. These amplifiers are sooo off? Only Unit14 had bimodal reasonable things, suggesting there is a good connection?  ![alt text](image-5.png)![alt text](image-6.png) 
# ![alt text](image-8.png)![alt text](image-9.png)
# ![alt text](image-7.png)
#
# * Chip is sitting there - could check under mic?
#
#
# ![alt text](image-25.png)
#

# %% [markdown]
#
