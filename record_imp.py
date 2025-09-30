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
        
def scatter_vis_impedance(aggr_df):
    aggr_df = aggr_df[aggr_df.impedance_Ohm > 0]
    fig, ax = plt.subplots(1, 1)
    ax.axhline(70, color='k', linestyle='--')
    ax.axhline(30, color='k', linestyle='--')
    # Use actual min/max for color scaling
    imp_kohm = aggr_df.impedance_Ohm / 1000
    vmin = imp_kohm.min()
    vmax = 90
    sc = ax.scatter(aggr_df.connectivity, imp_kohm + 1, alpha=.3, s=20)
    # sc = ax.scatter(aggr_df.connectivity, imp_kohm + 1, c=imp_kohm, cmap='viridis', vmin=vmin, vmax=vmax, s=20)
    # plt.colorbar(sc, label='Impedance (kOhm)')
    ax.set_yscale('log')
    ax.set_xlabel('Connectivity (external Sine signal)')
    ax.set_ylabel('Impedance (kOhm)')
    plt.title('Impedance Measurement Results')
    plt.savefig(f"./live_figures/all_imp_vs_connectivity_scatter.png")
    # plt.show()
        
def mea1k_vis_impedance(data, cmap_scaler=1):
    data.set_index('electrode', inplace=True)
    
    # create a colormap from 0 to 1000, matplotlib colormap 
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=110)

    (fig,ax), el_recs = draw_mea1k()
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
        
        if data.loc[el_i].connectivity < .5:
            continue
        
        # color by impedance
        imp = data.loc[el_i].impedance_Ohm
        color = cmap(norm(imp/1000))
        el_recs.set_edgecolor(color)
        
        xy = el_recs.get_xy()
        # annotate with stimulation unit
        stim_unit = int(data.loc[el_i].stim_unit)
        # ax.text(xy[0]+3, xy[1]+3, f"{stim_unit:02d}", fontsize=4, color='gray')
        
        print(data.loc[el_i].connectivity)
        whiteness = np.clip(data.loc[el_i].connectivity*cmap_scaler, 0, 1)
        el_recs.set_facecolor((whiteness, whiteness, whiteness))
    fig.savefig("./live_figures/all_imp_vs_connectivity_CMOS.png", dpi=300, transparent=False,
                bbox_inches='tight', pad_inches=0)
    # plt.show()

def measure_impedance(full_recdir, rec_time, nas_dir, configs_basepath, stim_settings,
                      implant_mapping, gain, dac_sine_amplitude, dac_id):
    
    reset_MEA1K(gain=gain, enable_stimulation_power=True)
    s = get_maxlab_saving()
    
    aggr = []
    for stim_unit in list(range(8)) + list(range(16,32)):
        fnames = glob(os.path.join(nas_dir, configs_basepath, f"StimUnit{stim_unit:02d}", "*.cfg"))
        print(f"Found {len(fnames)} configs for StimUnit{stim_unit}")
        
        # set the DAC to zero current, create the sine sequence around it
        print(stim_settings.loc[stim_unit, "zero_current_DAC"])
        dac_code = int(stim_settings.loc[stim_unit, "zero_current_DAC"].iloc[0])
        shift_DAC(dac_code)
        seq = create_stim_sine_sequence(dac_id=dac_id, amplitude=dac_sine_amplitude, 
                                    f=1000, ncycles=400, 
                                    center_around=dac_code)
        turn_on_stimulation_units([stim_unit], dac_id=dac_id, mode='small_current')
        
        # files look like el_config_El16752_StimUnit08_Ampl0114.cfg
        all_mea1k_els = [int(fname[fname.find("El")+2:fname.find("El")+7]) for fname in fnames]
        test_el_entries = implant_mapping[(implant_mapping.mea1k_el.isin(all_mea1k_els) & 
                                          (implant_mapping.pad_id.notna() | implant_mapping.mea1k_connectivity > .7))].sort_values(["connectivity_order", 'mea1k_connectivity'], ascending=[True, False])
        print(test_el_entries)

        # cmos_arr = np.arange(26400).reshape(120,220)
        # el_subset = cmos_arr[5:50, 60:110].flatten()
        # test_el_entries = implant_mapping[implant_mapping.mea1k_el.isin(el_subset)]
        # print(test_el_entries)
        
        for _, el_row_i in test_el_entries.iterrows():
            if not (el_row_i.mea1k_connectivity > .5 or np.random.random()<.01):
                print(f"Skipping electrode {el_row_i.mea1k_el} with connectivity {el_row_i.mea1k_connectivity}")
                continue
            print(el_row_i.to_frame())
            
            config_fullfname = [fname for fname in fnames if f"El{int(el_row_i.mea1k_el):05d}_" in fname]
            if len(config_fullfname) == 0:
                print(f"Missing config for stimulating electrode {el_row_i.mea1k_el} with StimUnit {stim_unit}. Skipping.")
                continue
            config_fullfname = config_fullfname[0]
            _process_single_el_config(config_fullfname, full_recdir, rec_time, dac_sine_amplitude,
                            s, dac_id, seq)
            ampl, phase_shift = _extract_sine_amplitude(full_recdir, 
                                                       os.path.basename(config_fullfname).replace(".cfg", ".raw.h5"),
                                                       debug=True)
            
            lsb = stim_settings.loc[stim_unit, "LSB_small_current_nA"].mean()
            if lsb <5:
                lsb = 5  # avoid too low LSB values
            print("LSB (small current) nA:", lsb)
            aggr.append(pd.Series({
                "stim_unit": stim_unit,
                "LSB_small_current_nA": lsb,
                "x": el_row_i.x,
                "y": el_row_i.y,
                "pad_id": el_row_i.pad_id,
                "metal": el_row_i.metal,
                "electrode": el_row_i.mea1k_el,
                "dac_code": dac_code,
                "connectivity": el_row_i.mea1k_connectivity,
                "amplitude_uV": ampl,
                "impedance_Ohm": ampl / (lsb*dac_sine_amplitude/1000)  # in Ohm
            }))
        turn_off_stimulation_units([stim_unit])  # reset all stim units

        aggr_df = pd.DataFrame(aggr)
        aggr_df.to_csv(os.path.join(full_recdir, "all_impedance.csv"), index=False)
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
    implant_name = "250929_MEA1K12_H1628pad1shankB5"
    headstage_name = "MEA1K12"
    subdir = f"devices/implant_devices/{implant_name}/recordings"
    configs_basepath = f"mea1k_configs/single_el2stimunit_configs2"
    stimulater_settings_path = f"devices/headstage_devices/{headstage_name}/smallcurrent_lsb_characterization.csv"
    stim_settings = pd.read_csv(os.path.join(nas_dir, stimulater_settings_path)).set_index("stimunit_id", drop=True)

    rec_dir = "5thBond_imp_measurement"
    post_download_wait_time = .6
    rec_time = .5
    gain = 7
    dac_sine_amplitude = 10
    dac_id = 0
    # ======== PARAMETERS ========
    
    implant_mapping = get_raw_implant_mapping(implant_name)
    full_recdir = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(full_recdir)} - ", full_recdir)
    
    # measure_impedance(full_recdir, rec_time, 
    #                   nas_dir, configs_basepath, stim_settings, implant_mapping,
    #                   gain=gain, dac_sine_amplitude=dac_sine_amplitude, 
    #                   dac_id=dac_id)
    
    aggr_df = pd.read_csv(os.path.join(full_recdir, "all_impedance.csv"))
    mea1k_vis_impedance(aggr_df)
    scatter_vis_impedance(aggr_df)

if __name__ == "__main__":
    main()