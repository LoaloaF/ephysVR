import os
import sys
from glob import glob
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

from mea1k_modules.mea1k_config_utils import start_saving, stop_saving
from mea1k_modules.mea1k_config_utils import attampt_connect_el2stim_unit, create_stim_sine_sequence
from mea1k_modules.mea1k_config_utils import turn_on_stimulation_units, turn_off_stimulation_units
from mea1k_modules.mea1k_config_utils import shift_DAC, reset_MEA1K
from mea1k_modules.mea1k_config_utils import get_maxlab_saving, get_maxlab_array

from mea1k_modules.mea1k_raw_preproc import read_raw_data, read_stim_DAC
from mea1k_modules.mea1k_post_processing import get_raw_implant_mapping
from mea1k_connectivity_scripts.signal_helpers import estimate_frequency_power
from mea1k_modules.mea1k_visualizations import draw_mea1k

def _process_single_el_config(config_fullfname, path, rec_time, dac_sine_amplitude,
                              s, dac_id, seq):
    config_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    array = get_maxlab_array()
    array.load_config(config_fullfname)
    
    fname = os.path.basename(config_fullfname).replace(".cfg", "")
    start_saving(s, dir_name=path, fname=fname, legacy=True)
    
    el = int(config_map.electrode.item())
    stim_unit = int(config_map.stim_unit.item())
    attampt_connect_el2stim_unit(el, array, used_up_stim_units=[],
                                 with_download=True)

    print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ with StimUnit{stim_unit} ")
    seq.send()
    time.sleep(rec_time)
    array.disconnect_electrode_from_stimulation(el)

    stop_saving(s)
    array.close()

def _extract_sine_amplitude(dir_name, fname, debug=True):
    amplifier = int(fname.split("Ampl")[-1].split("_")[0][:-7])
    data = read_raw_data(dir_name, fname, convert2uV=True,
                            subtract_dc_offset=False,)
    dac = read_stim_DAC(dir_name, fname)
    mean_ampl, phase_shift = estimate_frequency_power(data[amplifier].astype(float), 
                                                          sampling_rate=20_000, 
                                                          debug=debug, 
                                                          min_band=960, max_band=1040,
                                                          dac=dac.astype(float))
    return mean_ampl, phase_shift
        
def scatter_vis_impedance(aggr_df, output_dir=None):
    aggr_df = aggr_df[aggr_df.impedance_Ohm > 0]
    print(aggr_df)
    fig, ax = plt.subplots(1, 1)
    ax.axhline(100, color='k', linestyle='--')
    ax.axhline(50, color='k', linestyle='--')
    # Use actual min/max for color scaling
    imp_kohm = aggr_df.impedance_Ohm / 1000
    stimunit = aggr_df.stim_unit
    vmin = imp_kohm.min()
    vmax = 1100
    # sc = ax.scatter(aggr_df.connectivity, imp_kohm + 1, alpha=.3, s=20)
    # sc = ax.scatter(aggr_df.connectivity, imp_kohm + 1, c=imp_kohm, cmap='viridis', vmin=vmin, vmax=vmax, s=20)
    # plt.colorbar(sc, label='Impedance (kOhm)')
    sc = ax.scatter(aggr_df.connectivity, imp_kohm + 1, c=stimunit, cmap='tab20', s=20)
    plt.colorbar(sc, label='StimUnit Impedance (kOhm)')
    
    ax.set_yscale('log')
    ax.set_xlabel('Connectivity (external Sine signal)')
    ax.set_ylabel('Impedance (kOhm)')
    plt.title('Impedance Measurement Results')
    plt.savefig(f"./live_figures/all_imp_vs_connectivity_scatter.png")
    # if output_dir is not None:
    #     plt.savefig(os.path.join(output_dir, "all_imp_vs_connectivity_scatter.png"))
    # plt.show()
    
def stim_unit_wise_vis_impedance(aggr_df, output_dir=None):
    aggr_df = aggr_df[aggr_df.impedance_Ohm > 0].copy()
    aggr_df['imp_kohm'] = aggr_df.impedance_Ohm / 1000
    
    # Calculate mean impedance per stim unit and sort
    mean_imp = aggr_df.groupby('stim_unit')['imp_kohm'].mean().sort_values()
    ordered_stim_units = mean_imp.index.tolist()
    ordered_stim_units = np.arange(32)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot each stim unit's data with x-jitter
    for x_pos, su in enumerate(ordered_stim_units):
        subset = aggr_df[aggr_df.stim_unit == su]
        jitter = np.random.normal(0, 0.15, size=len(subset))
        ax.scatter(x_pos + jitter, subset.imp_kohm, alpha=0.6, s=15, c=subset.stim_unit, cmap='tab20', vmin=0, vmax=31)
        
    ax.set_xticks(range(len(ordered_stim_units)))
    ax.set_xticklabels(ordered_stim_units)
    ax.set_yscale('log')
    ax.set_xlabel('Stim Unit (Sorted by Mean Impedance)')
    ax.set_ylabel('Impedance (kOhm)')
    plt.title('Stim Unit-wise Impedance')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig(f"./live_figures/stim_unit_wise_impedance.png")

def hist_vis_impedance(aggr_df, output_dir=None):
    print(aggr_df)
    aggr_df = aggr_df[aggr_df.impedance_Ohm > 0]
    print(aggr_df)
    fig, ax = plt.subplots(1, 1)
    # ax.axvline(100, color='k', linestyle='--')
    # ax.axvline(50, color='k', linestyle='--')
    # Use actual min/max for color scaling
    imp_kohm = aggr_df.impedance_Ohm / 1000
    stimunit = aggr_df.stim_unit
    vmin = imp_kohm.min()
    # vmax = 1100
    # imp_kohm = np.clip(imp_kohm, vmin, vmax)
    
    #  hist
    ax.hist(imp_kohm, bins=100, alpha=0.5, color='gray', )#range=(vmin, vmax))
    # ax.set_xscale('log')
    ax.set_xlabel('Impedance (kOhm)')
    ax.set_ylabel('Count')
    plt.title('Impedance Measurement Histogram')
    plt.savefig(f"./live_figures/all_impedance_histogram.png")
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "all_impedance_histogram.png"))
    plt.close()
    
    
    # # sc = ax.scatter(aggr_df.connectivity, imp_kohm + 1, alpha=.3, s=20)
    # sc = ax.scatter(aggr_df.connectivity, imp_kohm + 1, c=imp_kohm, cmap='viridis', vmin=vmin, vmax=vmax, s=20)
    # plt.colorbar(sc, label='Impedance (kOhm)')
    # # sc = ax.scatter(aggr_df.connectivity, imp_kohm + 1, c=stimunit, cmap='tab20', s=20)
    # # plt.colorbar(sc, label='StimUnit Impedance (kOhm)')
    
    # ax.set_yscale('log')
    # ax.set_xlabel('Connectivity (external Sine signal)')
    # ax.set_ylabel('Impedance (kOhm)')
    # plt.title('Impedance Measurement Results')
    # plt.savefig(f"./live_figures/all_imp_vs_connectivity_scatter.png")
    # if output_dir is not None:
    #     plt.savefig(os.path.join(output_dir, "all_imp_vs_connectivity_scatter.png"))
    # # plt.show()
        
def mea1k_vis_impedance(data, output_dir=None, cmap_scaler=1):
    data.set_index('electrode', inplace=True)
    
    # create a colormap from 0 to 1000, matplotlib colormap 
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=1100)
    print(data)
    data = data[data.index.duplicated(keep=False)==False]
    print(data)
    (fig,ax), el_recs = draw_mea1k()
    conn_data, imp_data = [], []
    for el_i, el_recs in enumerate(el_recs):
        if el_i not in data.index:
            # print("missing", el_i, end=' ')
            # el_recs.set_edgecolor((.8,0,0))
            continue
        if data.loc[el_i:el_i].connectivity.isna().any():
            # el_recs.set_edgecolor((.8,0,0))
            print("NaN", el_i, end=' ')
            continue
        # needed for new local configs that were used for a hort time
        if data.loc[el_i].shape[0] == 2:
            print("duplicate", el_i, end=' ')
            continue
        
        if data.loc[el_i].connectivity < .3:
            continue
        
        xy = el_recs.get_xy()
        # if not (((xy[0] > 1600) and (xy[0] < 2700)) and (xy[1] < 800)):
        # # if not ((xy[0] > 2900) and (xy[1] > 1200)):
        #     continue
        # conn_data.append(data.loc[el_i].connectivity)
        # imp_data.append(data.loc[el_i].impedance_Ohm)
        
        # color by impedance
        imp = data.loc[el_i].impedance_Ohm
        color = cmap(norm(imp/600))
        el_recs.set_edgecolor(color)
        
        # annotate with stimulation unit
        stim_unit = int(data.loc[el_i].stim_unit)
        ax.scatter([xy[0]], [xy[1]], alpha=0.6, s=15, c=[stim_unit], cmap='tab20', 
                   vmin=0, vmax=31, edgecolors='none')
        # add annotation
        # ax.annotate(f"SU{stim_unit}", (xy[0]+20, xy[1]+20), fontsize=6, color='white', 
        #             ha='center', va='center')
        
        # print(data.loc[el_i].connectivity)
        whiteness = np.clip(data.loc[el_i].connectivity*cmap_scaler, 0, 1)
        el_recs.set_facecolor((whiteness, whiteness, whiteness))
    fig.savefig("./live_figures/all_imp_vs_connectivity_CMOS.png", dpi=300, transparent=False,
                bbox_inches='tight', pad_inches=0)
    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, "all_imp_vs_connectivity_CMOS.png"), 
                    dpi=300, transparent=False,
                    bbox_inches='tight', pad_inches=0)
    # plt.show()

def measure_impedance(full_recdir, rec_time, nas_dir, configs_basepath, stim_settings,
                      implant_mapping, gain, dac_sine_amplitude, dac_id):
    
    reset_MEA1K(gain=gain, enable_stimulation_power=True)
    s = get_maxlab_saving()
    
    aggr = []
    for stim_unit in list(range(32)):
        fnames = glob(os.path.join(nas_dir, configs_basepath, f"StimUnit{stim_unit:02d}", "*.cfg"))
        print(f"Found {len(fnames)} configs for StimUnit{stim_unit}")
        
        # set the DAC to zero current, create the sine sequence around it
        dac_code = stim_settings.loc[stim_unit, "zero_current_DAC"]
        if isinstance(dac_code, pd.Series):
            dac_code = (stim_settings.loc[stim_unit, "zero_current_DAC"].iloc[0])
        dac_code = int(dac_code)
        
        shift_DAC(dac_code)
        seq = create_stim_sine_sequence(dac_id=dac_id, amplitude=dac_sine_amplitude, 
                                    f=1000, ncycles=400, 
                                    center_around=dac_code)
        turn_on_stimulation_units([stim_unit], dac_id=dac_id, mode='small_current')
        
        # # files look like el_config_El16752_StimUnit08_Ampl0114.cfg
        # all_mea1k_els = [int(fname[fname.find("El")+2:fname.find("El")+7]) for fname in fnames]
        # test_el_entries = implant_mapping[(implant_mapping.mea1k_el.isin(all_mea1k_els) & 
        #                                   (implant_mapping.pad_id.notna() | implant_mapping.mea1k_connectivity > .01))].sort_values(["connectivity_order", 'mea1k_connectivity'], ascending=[True, False])
        #                                 #   (implant_mapping.mea1k_connectivity > .7))].sort_values(["connectivity_order", 'mea1k_connectivity'], ascending=[True, False])
        # print(test_el_entries)

        all_mea1k_els = [int(fname[fname.find("El")+2:fname.find("El")+7]) for fname in fnames]
        test_el_entries = implant_mapping[(implant_mapping.mea1k_el.isin(all_mea1k_els))].sort_values(['mea1k_connectivity'], 
                                                                                                      ascending=[True])
        # top 100 and bottom 100 connectivity electrodes
        test_el_entries = pd.concat([test_el_entries[test_el_entries.mea1k_connectivity>.4], 
                                     test_el_entries.iloc[-5:]])
        
        print(test_el_entries)
        # shuffle dataframe rows
        reorder = np.arange(len(test_el_entries))
        np.random.shuffle(reorder)
        print(reorder)
        test_el_entries = test_el_entries.iloc[reorder]
        print(test_el_entries)

        # exit()
        # cmos_arr = np.arange(26400).reshape(120,220)
        # el_subset = cmos_arr[5:50, 60:110].flatten()
        # test_el_entries = implant_mapping[implant_mapping.mea1k_el.isin(el_subset)]
        # print(test_el_entries)
        
        for _, el_row_i in test_el_entries.iterrows():
            
            # if not (el_row_i.mea1k_connectivity > .5 or np.random.random()<.01):
            #     print(f"Skipping electrode {el_row_i.mea1k_el} with connectivity {el_row_i.mea1k_connectivity}")
            #     continue
            # print(el_row_i.to_frame())
            
            config_fullfname = [fname for fname in fnames if f"El{int(el_row_i.mea1k_el):05d}_" in fname]
            if len(config_fullfname) == 0:
                print(f"Missing config for stimulating electrode {el_row_i.mea1k_el} with StimUnit {stim_unit}. Skipping.")
                continue
            config_fullfname = config_fullfname[0]
            _process_single_el_config(config_fullfname, full_recdir, rec_time, dac_sine_amplitude,
                            s, dac_id, seq)
            ampl, phase_shift = _extract_sine_amplitude(full_recdir, 
                                                       os.path.basename(config_fullfname).replace(".cfg", ".raw.h5"),
                                                       debug=False)
            
            lsb = stim_settings.loc[stim_unit, "LSB_small_current_nA"].mean()
            print("LSB (small current) nA:", lsb)
            aggr.append(pd.Series({
                "stim_unit": stim_unit,
                "LSB_small_current_nA": lsb,
                # "x": el_row_i.x,
                # "y": el_row_i.y,
                # "pad_id": el_row_i.pad_id,
                # "metal": el_row_i.metal,
                "electrode": el_row_i.mea1k_el,
                "dac_code": dac_code,
                "connectivity": el_row_i.mea1k_connectivity,
                "amplitude_uV": ampl,
                "impedance_Ohm": ampl / (lsb*dac_sine_amplitude/1000)  # in Ohm
            }))
        turn_off_stimulation_units([stim_unit])  # reset all stim units

        aggr_df = pd.DataFrame(aggr)
        if not os.path.exists(os.path.join(full_recdir, "processed")):
            os.makedirs(os.path.join(full_recdir, "processed"))
        aggr_df.to_csv(os.path.join(full_recdir, "processed", "all_impedance.csv"), index=False)
        # live redraw
        scatter_vis_impedance(aggr_df)
        mea1k_vis_impedance(aggr_df)

def main():
    L = Logger()
    L.init_logger(None, None, "DEBUG")
    
    # ======== PARAMETERS ========
    nas_dir = device_paths()[0]
    # implant_name = "250917_MEA1K12_H1628pad1shankB5"
    # implant_name = "250926_MEA1K12_H1278pad4shankB5"
    implant_name = "260413_MEA1K22_S1688pad14shankB5"
    headstage_name = "MEA1K22"
    # subdir = f"devices/implant_devices/{implant_name}/recordings"
    subdir = f"devices/headstage_devices/{headstage_name}/recordings"
    configs_basepath = f"mea1k_configs/single_el2stimunit_configs2"
    stimulater_settings_path = f"devices/headstage_devices/{headstage_name}/smallcurrent_lsb_characterization.csv"
    stim_settings = pd.read_csv(os.path.join(nas_dir, stimulater_settings_path)).set_index("stimunit_id", drop=True)

    rec_dir = "Bond2_ShubhamW3_16Shank_impedance"
    post_download_wait_time = .6
    rec_time = .5
    gain = 7
    dac_sine_amplitude = 10
    dac_id = 0
    # ======== PARAMETERS ========
    # setup dir
    full_recdir = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(full_recdir)} - ", full_recdir)
    
    
    # proper way
    fullfname = os.path.join(nas_dir, "devices", "implant_devices", implant_name, "bonding", f"bonding_mapping_{implant_name}.csv")
    implant_mapping = pd.read_csv(fullfname)
    # if not mapped yet
    # implant_mapping = pd.read_csv(os.path.join(nas_dir, subdir, 
    #                                            "Bond2_r4BothHalfs_ShubhamW3_16Shank_Vref15", 
    #                                            "processed", "extr_connectivity.csv"))
    # implant_mapping = implant_mapping.rename(columns={"el":"mea1k_el",
    #                                                 "connectivity":"mea1k_connectivity"})
    print(implant_mapping)
    
    
    
    # measure_impedance(full_recdir, rec_time, 
    #                   nas_dir, configs_basepath, stim_settings, implant_mapping,
    #                   gain=gain, dac_sine_amplitude=dac_sine_amplitude, 
    #                   dac_id=dac_id)
    
    aggr_df = pd.read_csv(os.path.join(full_recdir, "processed", "all_impedance.csv"))
    # mea1k_vis_impedance(aggr_df, output_dir=os.path.join(full_recdir, "processed"))
    # stim_unit_wise_vis_impedance(aggr_df, output_dir=os.path.join(full_recdir, "processed"))
    
    # scatter_vis_impedance(aggr_df, output_dir=os.path.join(full_recdir, "processed"))
    # hist_vis_impedance(aggr_df, output_dir=os.path.join(full_recdir, "processed"))
    # area_wise_imp(aggr_df, output_dir=os.path.join(full_recdir, "processed"))
    
    imp = aggr_df.loc[:, ['electrode', 'impedance_Ohm',]].astype(int)
    implant_mapping = pd.merge(implant_mapping, imp, left_on='mea1k_el', 
                               right_on='electrode', how='left')
    print(implant_mapping.columns)
    print(implant_mapping)
    # save the updated implant mapping with impedance values
    # implant_mapping.to_csv(fullfname, index=False)
    
    
    IMP_UPPER_THR = 200_000
    n_pads = implant_mapping['pad_id'].nunique()
    n_connected_pads = implant_mapping[implant_mapping['impedance_Ohm'] < IMP_UPPER_THR]['pad_id'].nunique()
    print(f"Connected pads: {n_connected_pads}/{n_pads} ({n_connected_pads/n_pads:.1%})")

    
    
    
if __name__ == "__main__":
    main()