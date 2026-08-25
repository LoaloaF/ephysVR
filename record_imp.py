import os
import sys
from glob import glob
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from datetime import datetime

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
from mea1k_connectivity_scripts.signal_helpers import estimate_frequency_power, extract_amplitude
from mea1k_modules.mea1k_visualizations import draw_mea1k
from mea1k_utils import cp_rec_dir_to_implant_dir
import mea1k_modules.mea1k_visualizations as vis

from PicoRecorder import PicoRecorder

def _stimlate_electrode(config_fullfname, path, s, dac_id, seq, 
                        stim_mode, stim_settings, stim_sequence_DAC_amplitude,
                              stim_sequence_frequency_Hz,
                              stim_sequence_ncycles, pico_rec=None):
    config_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    array = get_maxlab_array()
    array.load_config(config_fullfname)
    
    fname = os.path.basename(config_fullfname).replace(".cfg", "")
    start_saving(s, dir_name=path, fname=fname, legacy=True)
    pico_fullfname = os.path.join(path, f"{fname}.npz")
    if pico_rec:
        pico_rec.start(pico_fullfname)
    
    el = int(config_map.electrode.item())
    stim_unit = int(config_map.stim_unit.item())
    time.sleep(.1)
    attampt_connect_el2stim_unit(el, array, used_up_stim_units=[],
                                 with_download=True)

    seq_duration = (stim_sequence_ncycles / stim_sequence_frequency_Hz)
    print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ with StimUnit{stim_unit} for {seq_duration:.1f}s")
    seq.send()
    time.sleep(seq_duration)
    time.sleep(.1)  # wait a bit before disconnecting the electrode from stimulation
    array.disconnect_electrode_from_stimulation(el)
    array.download()
    time.sleep(.3)  # wait a bit before disconnecting the electrode from stimulation

    stop_saving(s)
    if pico_rec:
        pico_rec.stop()  # make sure to stop the PicoRecorder to save the file properly
    array.close()
    
    
    # save the config settings
    s = pd.Series({
                "stim_unit": stim_unit,
                "stim_mode": stim_mode,
                "LSB_nA": stim_settings.loc[stim_unit, f"{stim_mode}_median_LSB_nA"].item(),
                "electrode": el,
                "dac_id": dac_id,
                "stim_sequence_DAC_amplitude": stim_sequence_DAC_amplitude,
                "stim_sequence_frequency_Hz": stim_sequence_frequency_Hz,
                "stim_sequence_ncycles": stim_sequence_ncycles,
    }).to_frame().T
    s.to_csv(os.path.join(path, f"{fname}.csv"), index=True)
            
def comma_formatter(x):
    # Check if the value is a number (and not a boolean)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        # If it's an integer or a float representing a whole number (like 1000.0)
        if isinstance(x, int) or x.is_integer():
            return f"{int(x):,}"
        else:
            return f"{x:,.2f}"  # Change .2f to adjust decimal places
    return str(x)   

def _extract_sine_amplitude(dir_name, fname,  min_band, max_band, debug=True,):
    amplifier = int(fname.split("Ampl")[-1].split("_")[0][:-7])
    data = read_raw_data(dir_name, fname, convert2uV=True,
                            subtract_dc_offset=False,)
    dac = read_stim_DAC(dir_name, fname)
    # analysis window comes from the DAC onset/offset; notch + demean on by default
    mean_ampl, phase_shift, _, _ = extract_amplitude(
        data[amplifier].astype(float),
        sampling_rate=20_000,
        highpass=100,
        notch=True,
        min_band=min_band, max_band=max_band,
        zoom_from=-150, zoom_to=150,
        dac=dac.astype(float),
        use_dac_interval=True,
        name=fname.replace('.raw.h5', ''),
        plot_fname='debug_amplifier',
        debug=debug,
    )
    return mean_ampl, phase_shift

def measure_impedance(full_recdir, nas_dir, configs_basepath, stim_mode,
                      stim_settings, implant_mapping, gain, dac_sine_amplitude,
                      stim_sequence_frequency_Hz, stim_sequence_ncycles,
                      dac_id, stim_units=list(range(32)), debug=False,
                      shank_subset=None, dac_offset=0, negative_ctrl=True,
                      every_pad=None, every_pi_el=False, connectivity_threshold=.6,
                      skip_stimulation=False, skip_post_proc=False, rec_current_externally=False):
    if not skip_stimulation:
        reset_MEA1K(gain=gain, enable_stimulation_power=True)
        s = get_maxlab_saving()
        if rec_current_externally:
            pico_rec = PicoRecorder(voltage_range="5V", resolution="16BIT", sample_interval_us=50)
        else:
            pico_rec = None
    else:
        pico_rec = None
    
    aggr = []
    for stim_unit in stim_units:
        fnames = glob(os.path.join(nas_dir, configs_basepath, f"StimUnit{stim_unit:02d}", "*.cfg"))
        print(f"\n\nFound {len(fnames)} configs for StimUnit{stim_unit}")
        
        dac_code = int(stim_settings.loc[stim_unit, f"{stim_mode}_zero_current_DAC"])
        if not skip_stimulation:
            # set the DAC to zero current, create the sine sequence around it
            shift_DAC(dac_code+dac_offset)
            seq = create_stim_sine_sequence(dac_id=dac_id, amplitude=dac_sine_amplitude, 
                                            f=stim_sequence_frequency_Hz,
                                            ncycles=stim_sequence_ncycles,
                                            center_around=dac_code+dac_offset)
            turn_on_stimulation_units([stim_unit], dac_id=dac_id, mode=stim_mode)
        
        # stim unit electrodes. files look like el_config_El16752_StimUnit08_Ampl0114.cfg
        all_mea1k_els = [int(fname[fname.find("El")+2:fname.find("El")+7]) for fname in fnames]
        print(f"StimUnit{stim_unit} has {len(all_mea1k_els):,} electrodes that can be stimulated")
        test_el_entries = implant_mapping[(implant_mapping.mea1k_el.isin(all_mea1k_els))]
        conn_mask = test_el_entries.mea1k_connectivity > connectivity_threshold
        neg_ctrl_entries = test_el_entries.nsmallest(1, 'mea1k_connectivity')
        test_el_entries = test_el_entries[conn_mask]
        # only do subset of device
        if shank_subset is not None:
            test_el_entries = test_el_entries[test_el_entries.shank_id.isin(shank_subset)]
            
        print(test_el_entries)
        x,y = test_el_entries.mea1k_el %220, test_el_entries.mea1k_el //220
        # test_el_entries = test_el_entries[y>=40]

        print(f"For StimUnit{stim_unit}, {len(test_el_entries):,} electrodes have connectivity > {connectivity_threshold}")
        
        if every_pad is not None:
            assert 'pad_connectivity_rank' in implant_mapping.columns, "pad_connectivity_rank column not found in data"
            # test_el_entries = test_el_entries[test_el_entries.pad_connectivity_rank == 1]
            test_el_entries = test_el_entries[test_el_entries.pad_connectivity_rank.isin(every_pad)]
            print(f"After filtering for every pad, {len(test_el_entries):,} electrodes remain for StimUnit{stim_unit}")

        elif every_pi_el:
            assert 'el_connectivity_rank' in implant_mapping.columns, "el_connectivity_rank column not found in data"
            test_el_entries = test_el_entries[test_el_entries.el_connectivity_rank == 1]
            print(f"After filtering for every polyimide el, {len(test_el_entries):,} electrodes remain for StimUnit{stim_unit}")
        
        if negative_ctrl:
            test_el_entries = pd.concat([test_el_entries, neg_ctrl_entries]).drop_duplicates()
            
        for i, (_, el_row_i) in enumerate(test_el_entries.iterrows()):
            
            print(f"StimUnit{stim_unit}: electrode {i}/{len(test_el_entries)} with connectivity {el_row_i.mea1k_connectivity:.3f}...", end="\n")
            config_fullfname = [fname for fname in fnames if f"El{int(el_row_i.mea1k_el):05d}_" in fname]
            config_fullfname = config_fullfname[0]
            if not skip_stimulation:
                _stimlate_electrode(config_fullfname, full_recdir,
                                    s, dac_id, seq, stim_mode,
                                    stim_settings=stim_settings,
                                    stim_sequence_DAC_amplitude=dac_sine_amplitude,
                                    stim_sequence_frequency_Hz=stim_sequence_frequency_Hz,
                                    stim_sequence_ncycles=stim_sequence_ncycles,
                                    pico_rec=pico_rec,)
            if skip_post_proc:
                continue
            
            fname = os.path.basename(config_fullfname).replace(".cfg", ".raw.h5")
            # when skipping stimuation, files may not exist. used for post postprocessing when stimuated in previous run
            if not os.path.exists(os.path.join(full_recdir, fname)):
                continue
            print(os.path.join(full_recdir, fname))
            rng = max(1, stim_sequence_frequency_Hz//100) # allow 1% frequency deviation
            
            ampl, phase_shift = _extract_sine_amplitude(full_recdir, fname,                             
                                                        min_band=stim_sequence_frequency_Hz-rng, 
                                                        max_band=stim_sequence_frequency_Hz+rng,
                                                        debug=debug)
            # picoscope
            pico_fullfname = os.path.join(full_recdir, f"{fname.replace('.raw.h5', '.npz')}")
            if os.path.exists(pico_fullfname):
                pico_data = np.load(pico_fullfname)
                pico_time_s = pico_data['time_s']
                pico_voltage_mV = pico_data['mv']
                # convert to current in uA
                # 10K sense resistor, 100 amplfication, pico voltage in mV -> convert back to uA
                pico_current_uA = pico_voltage_mV / 100 * 1000 / 10_000

                if len(pico_time_s) < 2:
                    print("Picoscope data too short to estimate sampling rate.")
                else:
                    # estimate amplitude at stimulation frequency
                    pico_ampl, _, baseline_dc, sine_dc = extract_amplitude(
                        pico_current_uA,
                        # dac = read_stim_DAC(full_recdir, fname),
                        sampling_rate=20_000,
                        min_band=stim_sequence_frequency_Hz-rng,
                        max_band=stim_sequence_frequency_Hz+rng,
                        estimate_amplitude_from=3600,
                        estimate_amplitude_to=11000,
                        lowpass=3000.0,
                        name=f'StimUnit={stim_unit}, DAC={dac_code}+{dac_offset}, el={int(el_row_i.mea1k_el):05d}',
                        ylim=(-0.28, 0.28),
                        debug=debug,
                        plot_fname = 'debug_picoscope_signal',
                        save_to_dir = os.path.join(full_recdir, "processed")
                    )
                    time.sleep(2)
                    # 10K sense resisitor, 1000 amplfication, pico voltage in mV -> convert back to uA
                    ext_current_uA = pico_ampl
                    ext_current_imp = ampl / ext_current_uA
                    
            else:
                ext_current_uA = np.nan
                ext_current_imp = np.nan
                baseline_dc = np.nan
                sine_dc = np.nan
                # exit()
            
            # get the settings
            stim_info = pd.read_csv(os.path.join(full_recdir, f"{fname.replace('.raw.h5', '.csv')}")).iloc[0]
            amplitude_uA = stim_info.loc["stim_sequence_DAC_amplitude"].item() * stim_info.loc["LSB_nA"].item() / 1000  # in uA
            stim_res = pd.Series({
                "connectivity": el_row_i.mea1k_connectivity,
                "amplitude_uV": ampl,
                "current_uA": amplitude_uA,
                "external_current_uA": ext_current_uA,
                "external_current_pre_DC_offset": baseline_dc,
                "external_current_DC_offset": sine_dc,
                # micro volt / (LSB in nA * sine amplitude in DAC units / 1000 to convert to uA)) = Ohm
                "impedance_Ohm": ampl / amplitude_uA,  # in Ohm
                "external_impedance_Ohm": ext_current_imp,
                "phase_shift_deg": phase_shift,
            })
            print(pd.concat([stim_info, stim_res]).map(comma_formatter))
            aggr.append(pd.concat([stim_info, stim_res]))
            
        turn_off_stimulation_units([stim_unit])  # reset all stim units

        aggr_df = pd.DataFrame(aggr)
        if not os.path.exists(os.path.join(full_recdir, "processed")):
            os.makedirs(os.path.join(full_recdir, "processed"))
            
        
        # update the table
        if aggr_df.shape[0] > 0: 
            stim_unit_aggr =  aggr_df[aggr_df.stim_unit==stim_unit]
            if stim_unit_aggr.shape[0] > 0:
                stim_unit_aggr.to_csv(os.path.join(full_recdir, "processed", f"stim_unit_{stim_unit:02d}_impedance.csv"), index=False)
                
            aggr_df.to_csv(os.path.join(full_recdir, "processed", "all_impedance.csv"), index=False)
            data_plotting = aggr_df.copy()
            data_plotting['pad_metal'] = 1
            fig = vis.plot_impedance_analysis(os.path.join(full_recdir, "processed"), 
                                              data_plotting, skip_bottom=False, 
                                              color_by_metal=False, use_imp_connectivity=True)
            # save plot
            fig.savefig(os.path.join(full_recdir, "processed",
                                     "all_impedance_analysis.png"), dpi=300)

    # close the picoscope once, after all stim units are done
    if pico_rec is not None:
        pico_rec.close()


def get_all_stimunit_impedance_data(rec_dir):
    processed_dir = os.path.join(rec_dir, "processed")
    all_files = glob(os.path.join(processed_dir, "stim_unit_*_impedance.csv"))
    dfs = []
    for file in sorted(all_files):
        df = pd.read_csv(file)
        dfs.append(df)
    if len(dfs) > 0:
        aggr_df = pd.concat(dfs, ignore_index=True)
        return aggr_df
    else:
        return pd.DataFrame()

def main():
    L = Logger()
    L.init_logger(None, None, "WARNING")
    
    # ======== PARAMETERS ========
    nas_dir = device_paths()[0]
    # headstage_name = "MEA1K22"
    # implant_name = "260611_MEA1K24_S1688pad14shankB5"
    # headstage_name = "MEA1K24"
    # implant_name = "260602_MEA1K23_S844pad8shankB5"
    headstage_name = "MEA1K23"
    implant_name = "260625_MEA1K23_S1688pad14shankB6"
    # implant_name = "260715_MEA1K23_J1688pad2shankB1"
    # implant_name = "260703_MEA1K23_S844pad8shankB1"
    
    subdir = f"devices/implant_devices/{implant_name}/recordings"
    # subdir = f"devices/headstage_devices/{headstage_name}/recordings"
    configs_basepath = f"mea1k_configs/single_el2stimunit_configs"
    stimulater_settings_path = f"devices/headstage_devices/{headstage_name}/{headstage_name}_stimunit_characterization.csv"
    stim_settings = pd.read_csv(os.path.join(nas_dir, stimulater_settings_path)).set_index("stimunit_id", drop=True)
    
    gain = 7
    dac_id = 0
    debug = False
    # ======== PARAMETERS ========
    # stim_mode = 'small_current'
    stim_mode = 'large_current'
    stim_sequence_frequency_Hz = 1000
    stim_sequence_ncycles = 400
    dac_sine_amplitude = 3 if stim_mode == 'large_current' else 30

    
    t = datetime.now().strftime("%Y-%m-%d_%H.%M")
    # rec_dir = 'test'
    # rec_dir = f"{t}_Imp_Bond_extCur10KSensePico_PT4_{stim_mode}_f{stim_sequence_frequency_Hz:04d}Hz"
    # rec_dir = f"{t}_ +0DAC_PT4_{stim_mode}_f{stim_sequence_frequency_Hz:04d}Hz"

    rec_name = f'2026-06-02_12.24_8Sh4SilverPaint_VRefFPGA_ampl15_PT1_2.6mm'
    rec_name = f'2026-06-02_12.40_8Sh4SilverPaint_VRefFPGA_ampl15_PT1_2.2mm'
    rec_name = f'2026-06-02_12.55_8Sh4SilverPaint_VRefFPGA_ampl15_PT1_1.9mm'
    rec_name = f'2026-06-02_15.00_8Sh4SilverPaint_VRefFPGA_ampl15_PT1_1.7mm'
    rec_name = f'2026-06-25_13.59_Bond5_VrefFPGAStim_ampl15_PT4_1.7mm'
    rec_name = '2026-07-13_15.50_JDesignFirstSoldered2_VRef15_PT2'
    rec_name = '2026-07-15_07.55_JDesignFirstSoldered2_VRef15_PT3'
    rec_name = '2026-07-16_16.45_JDesignFirstSoldered2_VRef15_PT4' # tightened to check the bottom half of the device, does imp improve or worsen with pressure?  
    rec_name = '2026-07-21_14.47_JDesignFirstSoldered2_VRef15_PT5' 
    # rec_name = '2026-07-15_11.38_ReEtched14ShankReGoldPlated_VRef15_PT1'
    # rec_name = '2026-07-15_12.31_ReEtched14ShankReGoldPlated_VRef15_PT2'
    # rec_name = f'2026-07-15_13.59_ReEtched14ShankReGoldPlated_VRef15_PT3'
    
    # rec_name = f'2026-06-18_09.48_Bond4_VrefFPGAStim_ampl15_14shanksSilver_PT0'
    # rec_name = f'2026-06-18_09.48_Bond4_VrefFPGAStim_ampl15_14shanksSilver_PT0'
    
    # rec_name = f'2026-06-23_09.53_Bond4_VrefFPGAStim_ampl15_14shanksSilver_PT0'
    # rec_name = f'2026-06-23_11.28_Bond4_VrefFPGAStim_ampl15_PT1_2.2mm'
    # rec_name = f'2026-06-23_13.15_Bond4_VrefFPGAStim_ampl15_PT2_2.0mm'
    # rec_name = f'2026-06-24_20.24_Bond4_VrefFPGAStim_ampl15_PT3_1.8mm'
    # rec_name = f'2026-06-24_21.41_Bond4_VrefFPGAStim_ampl15_PT4_1.6mm'
    
    # rec_name = f'2026-07-10_14.03_ACF_VRefAmpl15_Try9_PT1'
    # rec_name = f'2026-07-16_14.58_ACF_VRefAmpl15_Try9_PTT3'
    # rec_name = f'2026-07-20_15.09_ACF_PIEl_VRefAmpl15_Try10_PT4'
    # rec_name = "2026-07-20_16.17_ACF_PIEl_VRefAmpl15_Try10_PT5_2mm"
    
    # if not mapped yet, pseudo implant_mapping
    implant_mapping = pd.read_csv(os.path.join(nas_dir, subdir,
                                               rec_name, 
                                               "processed", "extr_connectivity.csv")).rename(
                                                columns={"el":"mea1k_el",
                                                         "connectivity":"mea1k_connectivity"})
    implant_mapping['pad_connectivity_rank'] = None
    
    # fullfname = os.path.join(nas_dir, "devices", "implant_devices", implant_name, 
    #                          "bonding", f"bonding_mapping_{implant_name}.csv")
    # implant_mapping = pd.read_csv(fullfname)
    
    
    
    
    
    # rec_dir = f"{rec_name}_Imp_{stim_mode}_f{stim_sequence_frequency_Hz:04d}Hz"
    # rec_dir = f"2026-06-02_12.55_8Sh4SilverPaint_VRefFPGA_ampl15_PT1_1.9mm_Imp_large_current_f1000Hz_old"
    # rec_dir = f"test"
    
    rec_dir = f'{t}_ACF_PIEl_Try10_PT5_2mm_Imp2'
    # rec_dir = f'{t}_ReEtched14ShankReGoldPlated_Imp2_PT2_LC'
    # rec_dir = f"{t}_JDesignFirstSoldered2_Imp11_Bottom_shiftAdjusted_PT5_LC"
    
    measure_impedance(os.path.join(nas_dir, subdir, rec_dir), 
                    nas_dir, configs_basepath, 
                    stim_mode, stim_settings, implant_mapping,
                    gain=gain, dac_sine_amplitude=dac_sine_amplitude, 
                    dac_id=dac_id, 
                    stim_sequence_frequency_Hz=stim_sequence_frequency_Hz,
                    stim_sequence_ncycles=stim_sequence_ncycles,
                    stim_units = list(range(32)),
                    #   every_pi_el=True, 
                    dac_offset=0,
                    negative_ctrl=True,
                    # every_pad=range(10),
                    # shank_subset=[4,5,6,7], 
                    connectivity_threshold=.5,
                    skip_stimulation=False,
                    skip_post_proc=False,
                    rec_current_externally=False,
                    debug=debug,
                    )

    # cp_rec_dir_to_implant_dir(full_recdir, implant_name)
    
if __name__ == "__main__":
    main()