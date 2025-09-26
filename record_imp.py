import os
from glob import glob
import time
import maxlab

import pandas as pd
import numpy as np

from ephys_constants import device_paths
# import mea1k_modules.mea1k_config_utils as mea1k

import matplotlib.pyplot as plt
import ephys_constants as C
from mea1k_modules.mea1k_config_utils import start_saving, stop_saving, try_routing
from mea1k_modules.mea1k_config_utils import attampt_connect_el2stim_unit, create_stim_sine_sequence
from mea1k_modules.mea1k_config_utils import reset_MEA1K, turn_on_stimulation_units, array_config2df, turn_off_stimulation_units
from mea1k_modules.mea1k_config_utils import shift_DAC

from mea1k_modules.mea1k_config_utils import create_stim_pulse_sequence
from mea1k_modules.mea1k_config_utils import create_stim_onoff_sequence

from mea1k_modules.mea1k_raw_preproc import read_raw_data, read_stim_DAC, get_raw_implant_mapping
from mea1k_connectivity_scripts.signal_helpers import estimate_frequency_power



def process_config(config_fullfname, path, rec_time, dac_sine_amplitude,
                   s, dac_id, seq):
    config_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    array = maxlab.chip.Array()
    array.load_config(config_fullfname)
    
    fname = os.path.basename(config_fullfname).replace(".cfg", "")
    start_saving(s, dir_name=path, fname=fname, legacy=True)
    time.sleep(.2)
    
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

def vis_one_recording(dir_name, fname, debug=True):
    amplifier = int(fname.split("Ampl")[-1].split("_")[0][:-7])
    print(f"Processing {fname}, amplifier {amplifier}")
    print()
    print(os.path.join(dir_name, fname))
    print()
    data = read_raw_data(dir_name, fname, convert2uV=True,
                            subtract_dc_offset=False,)
    print(data.shape)
    dac = read_stim_DAC(dir_name, fname)


    mean_ampl, phase_shift = estimate_frequency_power(data[amplifier].astype(float), 
                                                          sampling_rate=20_000, 
                                                          debug=debug, 
                                                          min_band=960, max_band=1040,
                                                          dac=dac.astype(float))
    return mean_ampl, phase_shift
    
    # if debug:
    #     fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    #     ax[1].plot(data[amplifier], linewidth=1, alpha=.7)
    #     # col = ax[1].lines[-1].get_color()
        
    #     # ax[1].hlines(peak, 0, len(dac_transient), alpha=0.4, linestyle='--', color=col)
    #     # ax[1].vlines(peak_at, -200000, 200000, alpha=0.4, linestyle='--', color=col)
    #     # ax[1].scatter([peak_at], [peak], color=col, s=50, edgecolor=col, 
    #     #                 marker='o', zorder=10, label=f'DAC {DAC_val}, Peak {peak/1000:.0f} mV')
        
    #     if dac is not None:
    #         ax[0].plot(dac.astype(float), linewidth=1, color='black', 
    #                 alpha=0.4, label='DAC')
            
    #         ax[0].set_ylim(512-240, 512+240)
    #         ax[0].set_yticks(np.array((512-140, 512, 512+140)))
    #         ax[0].set_ylabel("DAC value")
    
    #     plt.title(f"Raw data from {fname}")
    #     plt.legend(fontsize=6, ncol=2)
    #     plt.savefig('./debug_signal.png')
    
    
    

def main():
    # L = Logger()
    # L.init_logger(None, None, "DEBUG")
    
    # ======== PARAMETERS ========
    nas_dir = device_paths()[0]
    implant_name = "250917_MEA1K12_H1628pad1shankB5"
    subdir = f"devices/implant_devices/{implant_name}/recordings"
    configs_basepath = f"mea1k_configs/single_el2stimunit_configs"
    stimulater_settings_path = "devices/headstage_devices/MEA1K12/smallcurrent_lsb_characterization.csv"

    rec_dir = "full_imp_measurement_outsideSolution"
    post_download_wait_time = .6
    rec_time = .5
    gain = 7
    dac_sine_amplitude = 10
    dac_id = 0
    # ======== PARAMETERS ========
    
    mapping = get_raw_implant_mapping(implant_name)
    print(mapping)
    print(mapping.columns)
    
    
    stim_settings = pd.read_csv(os.path.join(nas_dir, stimulater_settings_path)).set_index("stimunit_id", drop=True)

    full_recdir = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(full_recdir)} - ", full_recdir)
    reset_MEA1K(gain=gain, enable_stimulation_power=True)
    s = maxlab.Saving()
    
    # print(os.path.join(nas_dir, configs_basepath, which_configs))
    # fnames = glob(os.path.join(nas_dir, configs_basepath, which_configs, "*.cfg"))
    
    fig, ax = plt.subplots(1, 1)
    ax.axhline(1250, color='k', linestyle='--')
    ax.axhline(3750, color='k', linestyle='--')
    all_ampl = []
    all_connect = []
    stim_units = []
    aggr = []
    for stim_unit in range(32):
        fnames = glob(os.path.join(nas_dir, configs_basepath, f"StimUnit{stim_unit:02d}", "*.cfg"))
        print(f"Found {len(fnames)} configs for StimUnit{stim_unit}")

        
        dac_code = int(stim_settings.loc[stim_unit, "zero_current_DAC"])
        shift_DAC(dac_code)
        seq = create_stim_sine_sequence(dac_id=dac_id, amplitude=dac_sine_amplitude, 
                                    f=1000, ncycles=400, 
                                    center_around=dac_code)
        
        turn_on_stimulation_units([stim_unit], dac_id=dac_id, mode='small_current')
        for i, config_fullfname in enumerate(sorted(fnames)):
            
            print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}", flush=True)
            process_config(config_fullfname, full_recdir, rec_time, dac_sine_amplitude,
                            s, dac_id, seq)
            
            ampl, phase_shift = vis_one_recording(full_recdir, os.path.basename(config_fullfname).replace(".cfg", ".raw.h5"))
            
            
            config_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
            conn = mapping[mapping.mea1k_el==config_map.electrode.item()]
            all_connect.append(conn.mea1k_connectivity.item())
            all_ampl.append(ampl)
            stim_units.append(stim_unit)
            
            print(conn)
            aggr.append(pd.Series({
                "stim_unit": stim_unit,
                "ampl": ampl,
                "connectivity": conn.mea1k_connectivity.item(),
                "x": conn.x.item(),
                "y": conn.y.item(),
                "pad_id": conn.pad_id.item(),
                "electrode": config_map.electrode.item(),
                "dac_code": dac_code,
            }))
            
            ax.scatter(all_connect, all_ampl, c=stim_units)
            # log scale
            ax.set_yscale('log')
            fig.savefig(f"./all_imp_vs_connectivity.png")
        turn_off_stimulation_units([stim_unit])  # reset all stim units

    aggr = pd.DataFrame(aggr)
    aggr.to_csv(os.path.join(full_recdir, "all_imp_vs_connectivity.csv"), index=False)
    print(aggr)
    print(aggr.groupby("connectivity").ampl.describe())
    print(aggr.groupby("pad_id").ampl.describe())

    # for i, config_fullfname in enumerate(sorted(fnames)):
    #     print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}")
    #     config_info = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    #     # print(config_info)
    #     print()

    #     array = mea1k.get_maxlab_array()
    #     array.load_config(config_fullfname)
    #     print("Downloading presaved config...")
    #     array.download()
        
        # if with_external_sine:
        #     mea1k.begin_fpga_sine_stim()
        # time.sleep(post_download_wait_time)        
        
        # fname = os.path.basename(config_fullfname).replace(".cfg", "")
        # mea1k.start_saving(s, dir_name=path, fname=fname)
        # time.sleep(rec_time)
        
        # if with_external_sine:
        #     mea1k.end_fpga_sine_stim()
        # array.close()
        # mea1k.stop_saving(s)

if __name__ == "__main__":
    main()