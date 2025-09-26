import datetime
import glob
import time
import random

import maxlab
from mea1k_modules.mea1k_config_utils import create_stim_sine_sequence
from mea1k_modules.mea1k_config_utils import reset_MEA1K, turn_on_stimulation_units, turn_off_stimulation_units
from mea1k_modules.mea1k_config_utils import start_saving, stop_saving, shift_DAC
from mea1k_modules.mea1k_config_utils import setup_stim_unit_characterization, find_stim_unit_amplifier
# from glob import glob
from mea1k_connectivity_scripts.signal_helpers import estimate_frequency_power
from mea1k_connectivity_scripts.signal_helpers import lowpass_filter
from mea1k_modules.mea1k_raw_preproc import read_stim_DAC, read_raw_data
import numpy as np
import pandas as pd
import os
# fix state

import matplotlib.pyplot as plt


def _sweep_DAC(dirname, array, stim_unit, ampl_id, set_id, DAC_values, debug):
    s = maxlab.Saving()
        
    # first measurement is always off, skip saving and run it twice
    DAC_values = [DAC_values[0]] + DAC_values
    for i, DAC_val in enumerate(DAC_values):
        fname = f"config_StimUnit{int(stim_unit):02d}_Ampl{ampl_id:04d}_Set{set_id}_DAC{DAC_val}"
        if i != 0:
            start_saving(s, dir_name=dirname, fname=fname, channels=[ampl_id], legacy=debug)
        time.sleep(.02)
        
        # Indicator 1
        # large effect, trace should come back to baseline
        print(f"Turning on stimulation unit {stim_unit}...\n\n")
        turn_on_stimulation_units([stim_unit], mode='small_current')
        time.sleep(.4)
        
        # Indicator 2
        shift_DAC(DAC_val)
        time.sleep(.3)
            
        # Indicator 3
        # has large effect on trace, what we care about
        array.connect_amplifier_to_ringnode(int(ampl_id))
        array.download()
        time.sleep(.3)

        print(f"Turning on stim units...\n\n")
        array.connect_amplifier_to_stimulation(ampl_id)
        array.download()
        # this is the dislocation we care about?
        time.sleep(.3)

        print(f"Turning off stim units...\n\n")
        turn_off_stimulation_units([stim_unit])
        array.disconnect_amplifier_from_stimulation(ampl_id)
        array.disconnect_amplifier_from_ringnode(ampl_id)

        if i != 0:
            stop_saving(s)

def _get_DAC_candidate_values(centered_around, delta):
    # emforce max 32 values
    resolution = max(1, int((2*delta) / 32))
    DAC_candidates = np.arange(centered_around-delta, centered_around+delta+1, resolution)
    # ensure between 0 and 1023
    if DAC_candidates[0] < 0 or DAC_candidates[-1] > 1023:
        print("Warning! DAC values out of range, shifting into range")
        shift_lower_end_by, shift_upper_end_by = 0, 0
        for i in range(len(DAC_candidates)):
            if DAC_candidates[i] < 0:
                DAC_candidates[i] += 0 + shift_lower_end_by
                shift_lower_end_by += 32
            if DAC_candidates[i] > 1023:
                DAC_candidates[i] = 1023 - shift_upper_end_by
                shift_upper_end_by += 32
    return DAC_candidates.astype(int).tolist()
    
def char_stim_units(dirname, n_amplifiers=2, stim_units=list(range(32)), debug=False):
    array = setup_stim_unit_characterization(dirname)
    
    for stim_unit in stim_units:
        # initial DAC sweep parameters
        centered_around = 512
        delta = 16
        for which_amplifier in range(n_amplifiers):
            ampl_id = find_stim_unit_amplifier(array, stim_unit=stim_unit, which_amplifier=which_amplifier)

            # new amplfier starts around previous amplifier best DAC value
            set_id = 0
            while True:
                print(f"\n==========={centered_around}+-{delta}================")
                DAC_values = _get_DAC_candidate_values(centered_around, delta)
                _sweep_DAC(dirname, array, stim_unit, ampl_id, set_id, DAC_values, debug=debug)
                results = extract_DAC_transient_set(dirname, ampl_id, stim_unit, 
                                                    set_id, debug=debug)

                # check if the transient is low enough, if yes, break - the exact number is hard to define
                peak = results.peak_uV.abs().min()
                if peak < 20_000:
                    print(f"Stim unit {stim_unit} amplifier {ampl_id} set {set_id} done, peak_uV: {peak} uV")
                    print(results.sort_values(by='peak_uV', key=abs, ascending=True).head(3))
                    if delta == 16:
                        print("Delta is already 16, and peak is <20mV, stopping here.\n\n\n")
                        break
                    else:
                        print(f"Peak is low enough, but delta={delta} > 16")
                        delta = max(16, delta // 4)
                        centered_around = int(results.iloc[results.peak_uV.abs().argmin()].DAC_val)
                        print(f"Recentering around {centered_around}, new delta={delta}")
                
                set_id += 1
                peak_std = results.peak_uV.std()
                if peak_std <30_000:
                    delta *= 4
                    print(f"Stim unit {stim_unit} amplifier {ampl_id} set {set_id} "
                          f"done, peak std={peak_std}, low variance, increasing DAC range to {delta}")
                
                else:
                    # find the DAC value that gives the lowest peak_uV
                    best_row = results.iloc[results.peak_uV.abs().argmin()]
                    print(f"Stim unit {stim_unit} amplifier {ampl_id} set {set_id} "
                          f"with high variance: {peak_std} - peak_uV={best_row.peak_uV:.1f} uV at DAC"
                          f" {best_row.DAC_val}, centering around it and reducing delta to 1/4")
                    centered_around = int(best_row.DAC_val)
                    delta = max(16, delta//4) # should get smaaller while closing in on the best value
                
                if set_id > 6:
                    print("Too many sets, stopping here.\n\n\n")
                    break

def extract_fname_info(fname):
    # config_StimUnit00_Ampl0828_Set0_DAC480.raw.h5
    stimunit_id = int(fname.split("_")[1].replace("StimUnit", ""))
    ampl_id = int(fname.split("_")[2].replace("Ampl", ""))
    set_id = int(fname.split("_")[3].replace("Set", ""))
    DAC_code = int(fname.split("_")[4].replace("DAC", "").replace(".raw.h5", ""))
    return stimunit_id, ampl_id, set_id, DAC_code

def extract_DAC_transient_set(dirname, ampl_id, stim_unit, set_id, debug=False):
    if debug:
        fig, ax = plt.subplots(figsize=(20, 8), nrows=2, sharex=True)
    
    # go through all files in the directory, skip those that do not match the set_id, ampl_id, stim_unit
    dac_transients_res = []
    for fname in sorted(os.listdir(dirname)):
        if not fname.endswith(".raw.h5"):
            continue
        if (f"Set{set_id}_" not in fname) or ("DAC" not in fname):
            continue
        if f"Ampl{ampl_id:04d}_" not in fname:
            continue
        if f"StimUnit{stim_unit:02d}_" not in fname:
            continue
        
        # extract dac_val
        _, _, _, DAC_val = extract_fname_info(fname)
        
        print(f"Processing {fname} ", ampl_id, stim_unit)
        data = read_raw_data(dirname, fname, convert2uV=True,
                             subtract_dc_offset=False,)
        print(data.shape)
        dac = read_stim_DAC(dirname, fname)
        
        # 12_500, 20_000 interval contains the transient
        dac_transient = data[0, 12_500:20_000].astype(float)
        if dac is not None:
            dac = dac[12_500:20_000]
        
        baseline = np.median(dac_transient[:2_000])
        dac_transient -= baseline

        # calculate derivative
        diff_dac_transient = np.diff(dac_transient)
        
        # find peak closely after the fast rise
        fast_rise_at = np.argmax(np.abs(diff_dac_transient)) 
        peak_at = fast_rise_at + np.argmax(np.abs(dac_transient[fast_rise_at: fast_rise_at + 8]))
        peak = dac_transient[peak_at]
        print(f"Baseline: {baseline:.1f}, Peak: {peak:.1f}, DAC_val: {DAC_val}, at {peak_at}")
        
        if peak_at <2000 or peak_at >3000:
            print("\n\nWARNING! Peak at unexpected location, no peak?")
            # continue
        
        dac_transients_res.append(pd.Series({
            "stimunit_id": stim_unit,
            "ampl_id": ampl_id,
            "set_id": set_id,
            "DAC_val": DAC_val,
            "peak_uV": peak,
            "peak_at": peak_at,
            "baseline": baseline,
        }))
        
        if debug:
            ax[1].plot(dac_transient, linewidth=1, alpha=.7)
            col = ax[1].lines[-1].get_color()
            
            ax[1].hlines(peak, 0, len(dac_transient), alpha=0.4, linestyle='--', color=col)
            ax[1].vlines(peak_at, -200000, 200000, alpha=0.4, linestyle='--', color=col)
            ax[1].scatter([peak_at], [peak], color=col, s=50, edgecolor=col, 
                          marker='o', zorder=10, label=f'DAC {DAC_val}, Peak {peak/1000:.0f} mV')
            
            if dac is not None:
                ax[0].plot(dac.astype(float), linewidth=1, color='black', 
                        alpha=0.4, label='DAC')
                
                ax[0].set_ylim(512-340, 512+340)
                ax[0].set_yticks(np.array((512-140, 512, 512+140)))
                ax[0].set_ylabel("DAC value")
        
            plt.title(f"Raw data from {fname}")
            plt.legend(fontsize=6, ncol=2)
    
    dac_transients_res = pd.DataFrame(dac_transients_res)
    fullfname = os.path.join(dirname, f"results_StimUnit{stim_unit:02d}_Ampl{ampl_id:04d}_Set{set_id}.csv")
    dac_transients_res.to_csv(fullfname, index=False)        
    print(dac_transients_res)
    if debug:
        plt.show()
    return dac_transients_res

def eval_char_stim_units(dirname,  stim_units, debug=False):
    all_results = []
    for stim_unit in stim_units:
        res_fullfnames = glob.glob(os.path.join(dirname, f"results_StimUnit{stim_unit:02d}_*.csv"))
        print(f"\nFound {len(res_fullfnames)} result files for stim unit {stim_unit}")
        
        stimunit_res = pd.DataFrame()
        for fullfname in res_fullfnames:
            print(f"Reading {fullfname}")
            stimunit_res = pd.concat([stimunit_res, pd.read_csv(fullfname)], axis=0)
        
        dac_median = stimunit_res[['DAC_val', 'peak_uV']].groupby('DAC_val').apply(lambda x: x.abs().mean())
        dac_median = dac_median.rolling(window=2, center=True, min_periods=1).mean()
        min_dac_value = dac_median.peak_uV.abs().idxmin()
        peak_uV = dac_median.loc[min_dac_value].peak_uV
        print(f"Stim unit {stim_unit} minimum |peak_uV| at DAC value {min_dac_value}: {peak_uV:.1f} uV")
        all_results.append(pd.Series({
            "stimunit_id": stim_unit,
            "zero_current_DAC_code": min_dac_value,
            "median_peak_uV": peak_uV,
            "n_measurements": len(stimunit_res),
            "peak_std_uV": stimunit_res.loc[stimunit_res.DAC_val==min_dac_value].peak_uV.std(),
        }))
        
        plt.vlines(min_dac_value, 0, 200_000, color='green', linestyle='--', label='Min |peak_uV|')
        plt.axhline(peak_uV, color='green', linestyle='--')
        plt.text(min_dac_value+2, 180_000, f"{peak_uV/1000:.1f} mV at DAC {min_dac_value}", 
                color='green', ha='left', fontsize=8)
        
        plt.scatter(stimunit_res.DAC_val, abs(stimunit_res.peak_uV), alpha=.4, s=20)
        plt.plot(dac_median.DAC_val, dac_median.peak_uV, color='blue', linewidth=1, 
                alpha=.8, label='Median |peak_uV|')
        plt.title(f"Stim unit {stim_unit}")
        plt.xlabel("DAC value")
        plt.ylabel("Peak uV")
        plt.legend()
        if debug:
            plt.show()
        if not os.path.exists(os.path.join(dirname, "processed")):
            os.makedirs(os.path.join(dirname, "processed"), exist_ok=True)
        plt.savefig(os.path.join(dirname, "processed", f"StimUnit{stim_unit:02d}_characterization.png"))
        plt.close()
        
    all_results = pd.DataFrame(all_results)
    all_results.to_csv(os.path.join(dirname, "processed", "StimUnits_characterization.csv"), index=False)    
    print("\n\nAll results:\n", all_results)

def char_current_lsb(dirname, R, sine_ampl_DAC_units, stim_units, n_amplifiers, debug=False):
    array = setup_stim_unit_characterization(dirname)
    s = maxlab.Saving()
    dac_settings = pd.read_csv(os.path.join(dirname, "processed", "StimUnits_characterization.csv"))
    
    for stim_unit in stim_units:
        for which_amplifier in range(n_amplifiers):
            ampl_id = find_stim_unit_amplifier(array, stim_unit=stim_unit, which_amplifier=which_amplifier)

            print(dac_settings.loc[dac_settings.stimunit_id==stim_unit])
            dac_code = int(dac_settings.loc[dac_settings.stimunit_id==stim_unit].zero_current_DAC_code)
            print(f"\n\nStim unit {stim_unit} amplifier {ampl_id}, zero current DAC code: {dac_code}")

            sine_seq = create_stim_sine_sequence(dac_id=0, amplitude=sine_ampl_DAC_units, 
                                                 f=1000, ncycles=400,
                                                 center_around=dac_code)
            
            fname = f"config_StimUnit{int(stim_unit):02d}_Ampl{ampl_id:04d}_CurrentLSB"
            start_saving(s, dir_name=dirname, fname=fname, legacy=True)
            
            turn_on_stimulation_units([stim_unit], mode='small_current')
            shift_DAC(dac_code)
            array.connect_amplifier_to_ringnode(int(ampl_id))
            array.connect_amplifier_to_stimulation(ampl_id)
            array.download()
            
            sine_seq.send()
            time.sleep(0.5)

            turn_off_stimulation_units([stim_unit])
            array.disconnect_amplifier_from_stimulation(ampl_id)
            array.disconnect_amplifier_from_ringnode(ampl_id)
            stop_saving(s)

def eval_current_lsb(dirname, R, sine_ampl_DAC_units, stim_units, debug=False):
    aggr = []
    for fname in sorted(os.listdir(dirname)):
        if not fname.endswith("_CurrentLSB.raw.h5"):
            continue
        stimunit_id = int(fname.split("_")[1].replace("StimUnit", ""))
        ampl_id = int(fname.split("_")[2].replace("Ampl", ""))

        print(f"\nProcessing {fname} ", ampl_id, stimunit_id)
        data = read_raw_data(dirname, fname, convert2uV=True, )#row_slice=[ampl_id, ],)
        dac = read_stim_DAC(dirname, fname)
        
        interv = 1200, 9400
        zero_current_DAC_code = dac[interv[0]]
        mean_ampl, phase_shift = estimate_frequency_power(data[ampl_id, interv[0]:interv[1]].astype(float), 
                                                          sampling_rate=20_000, 
                                                          debug=debug, 
                                                          min_band=960, max_band=1040,
                                                          dac=dac[interv[0]:interv[1]].astype(float) if dac is not None else None)
        aggr.append(pd.Series({
            "stimunit_id": stimunit_id,
            "mean_amplitude_uV": mean_ampl,
            "zero_current_DAC": zero_current_DAC_code,
            "resistance": R,
            "sine_amplitude_DAC_units": sine_ampl_DAC_units,
            "LSB_small_current_nA": (mean_ampl / R) / (sine_ampl_DAC_units / 1_000),  # in nA
            "phase_shift_deg": phase_shift,
        }))
        
    fig, ax = plt.subplots(figsize=(14, 8), ncols=2, nrows=2, sharex=True)
    ax = ax.flatten()
    
    ax[0].set_title("Mean Amplitude uV")
    ax[0].set_xlabel("Stim Unit ID")
    ax[0].set_ylabel("Mean Amplitude [uV]")
    ax[0].set_xticks(stim_units)
    ax[0].grid(True)
    ax[0].set_ylim(0, 60_000)
    ax[0].scatter([a.stimunit_id for a in aggr], [a.mean_amplitude_uV for a in aggr], s=50)
    
    # Circular plot for phase shift
    ax[1].set_title("Phase Shift (Circular)")
    ax[1].set_xlabel("Stim Unit ID")
    ax[1].set_ylabel("Phase Shift [deg]")
    ax[1].set_xticks(stim_units)
    # Convert degrees to radians for circular plot
    phase_radians = np.deg2rad([a.phase_shift_deg for a in aggr])
    ax[1].scatter([a.stimunit_id for a in aggr], np.mod(phase_radians, 2 * np.pi), s=50)
    ax[1].set_yticks(np.deg2rad([0, 90, 180, 270, 360]))
    ax[1].set_yticklabels(['0°', '90°', '180°', '270°', '360°'])
    ax[1].set_ylim(0, 2 * np.pi)
    ax[1].tick_params(axis='x', which='both', rotation=90)
    
    ax[2].set_title("LSB, small_current mode in [nA]")
    ax[2].set_xlabel("Stim Unit ID")
    ax[2].set_ylabel("LSB [nA]")
    ax[2].set_xticks(stim_units)
    ax[2].scatter([a.stimunit_id for a in aggr], [a.LSB_small_current_nA for a in aggr], s=50)
    ax[2].tick_params(axis='x', which='both', rotation=90)

    result_fullfname = os.path.join(dirname, "processed", "smallcurrent_lsb_characterization.csv")
    aggr = pd.DataFrame(aggr)
    aggr.to_csv(result_fullfname, index=False)
    # copy to devices/headstage_devices basedir
    aggr.to_csv(os.path.join(dirname, "..", "..", "smallcurrent_lsb_characterization.csv"), index=False)
    plt.savefig(result_fullfname.replace(".csv", ".png"))
    if debug:
        plt.show()
    plt.close()
            
            
            
            
            
            
            
            
            
            
            
            

def main():
    random.seed(42)
    debug = False
    
    nas_dir = "/mnt/SpatialSequenceLearning/"
    # nas_dir = "/home/houmanjava/nas_imitation"
    device_dir = "devices/headstage_devices/MEA1K12/recordings"
    # device_dir = "devices/well_devices/4983/recordings"
    R = 1_000_000  # 1 MOhm
    sine_ampl_DAC_units = 10 # in DAC units
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
    rec_dir = f"{t}_{R=}_CharStimUnits"
    # rec_dir = "2025-09-24_16.20_R=1000000_CharStimUnits"
    rec_dir = "2025-09-23_17.00_R=1000000_CharStimUnits"
    # take the newest directory
    # print(sorted(os.listdir(os.path.join(nas_dir, device_dir))))
    # rec_dir = sorted(os.listdir(os.path.join(nas_dir, device_dir)))[-1]
    full_path = os.path.join(nas_dir, device_dir, rec_dir)

    n_amplifiers = 1
    # stim_units = list(range(32))
    stim_units = [0]
    
    # char_stim_units(full_path, n_amplifiers=n_amplifiers, stim_units=stim_units, debug=debug)
    # eval_char_stim_units(full_path, stim_units=stim_units, debug=debug)
    # print(full_path)
    # char_current_lsb(full_path, R=R, sine_ampl_DAC_units=sine_ampl_DAC_units,
    #                  stim_units=stim_units, n_amplifiers=n_amplifiers, debug=debug)
    eval_current_lsb(full_path, R=R, sine_ampl_DAC_units=sine_ampl_DAC_units, 
                     stim_units=stim_units, debug=debug)
    
    
    
if __name__ == "__main__":
    main()