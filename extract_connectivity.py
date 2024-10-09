import os
from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import scipy.signal as signal


import pandas as pd
    
MAX_AMPL_mV = 2900.
BITRATE = 2**10
GAIN = 7
SR = 20_000
CYCLE_LENGTH = 20
NCYCLES = 100

def estimate_frequency_power(signal, sampling_rate, min_band, max_band, debug=False):
    # Compute the FFT of the signal
    fft_result = np.fft.fft(signal)
    # Compute the power spectrum
    power_spectrum = np.abs(fft_result) ** 2
    # Compute the corresponding frequencies
    freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    # Only keep the positive frequencies
    positive_freqs = freqs[freqs >= 1]
    positive_power_spectrum = power_spectrum[freqs >= 1]
    if debug:
        plt.plot(positive_freqs, positive_power_spectrum)
        # plt.yscale('log')
        plt.title('Power Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.grid(True)
        plt.show()
    return positive_power_spectrum[(positive_freqs>min_band) & 
                                   (positive_freqs<max_band)].mean()
    
def get_n_configs(path):
    return len(glob(os.path.join(path, "config_*.raw.h5")))

def get_resolution():
    return (MAX_AMPL_mV/BITRATE) /GAIN

def convert_to_vol(data):
    # scale data to mv
    res = get_resolution()
    data = (data-BITRATE/2) *res
    # floating/ offset
    data += MAX_AMPL_mV/2 
    return data

def get_h5_mapping(path, config_i):
    fname = f'config_{config_i:03d}.raw.h5'
    with h5py.File(os.path.join(path, fname), 'r') as file:
        mapping = np.array([list(m) for m in file['mapping']])
        channels = mapping[:, :2].astype(int)
        channels = pd.Series(channels[:,0], index=channels[:,1]).sort_values()
        el_xy = mapping[:, 2:4]
    return channels, el_xy

def get_stim_mapping(path):
    return pd.read_pickle(f"{path}/stim_mapping.pkl")

def read_data(path, config_i, convert2vol=False, row_slice=slice(None), col_slice=slice(None)):
    fname = f"config_{config_i:03d}.raw.h5"
    with h5py.File(os.path.join(path, fname), 'r') as file:
        print("Reading data...", flush=True, end="")
        data = np.array(file['sig'][row_slice, col_slice])
        print("done.", flush=True)
    if convert2vol:
        data = convert_to_vol(data)
    return data

def get_stim_start_stop_indices(data, debug=True):
    # TTL DAC signal is 512 when not stimulated, else 512+amplitude
    diffdata = np.diff(data)
    diffdata[diffdata>0] = 1
    # smooth over rare 0/512 data during stimulation
    diffdata = medfilt(diffdata, kernel_size=5)
    diffdata = np.diff(diffdata)

    indices = np.where(diffdata>0)[0]
    if debug:
        [plt.axvline(idx, color='r' if i%2==0 else 'g', alpha=0.5) 
         for i, idx in enumerate(indices)]
        plt.plot(data)
        plt.show()
    return indices

def extract_stim_traces(path, config_i, precomputed=None, debug=False):
    stim_mapping = get_stim_mapping(path)
    mapping, _ = get_h5_mapping(path, config_i)
    print(f"Extracting DAC stimulation TTLs for config {config_i:03d}")
    
    dac_data = read_data(path, config_i, row_slice=slice(1024, 1025))[0]
    stim_sequence_indices = get_stim_start_stop_indices(dac_data, debug=debug)

    if stim_mapping[config_i].index.unique('stim_set').size != len(stim_sequence_indices)/2:
        print("Mismatch between number of stim sets and number of stim sequences\n")
        # append last index to stim_sequence_indices
        stim_sequence_indices = np.append(stim_sequence_indices, dac_data.size)
        # return
        
    all_els = stim_mapping[config_i].index.get_level_values('el')
    all_chnls = mapping.loc[all_els]
    reindexer = np.argsort(all_chnls)
    all_els, all_chnls = all_els[reindexer], all_chnls.iloc[reindexer]
    # print(stim_mapping.loc[config_i,:,:,all_els])
    # print()
    
    config_powers = []
    for stim_set_i, (start, stop) in enumerate(zip(stim_sequence_indices[::2], 
                                                   stim_sequence_indices[1::2])):
        if stop-start+1 != NCYCLES*CYCLE_LENGTH:
            print(f"Stim set {stim_set_i}: {start} - {stop}, OFF: {stop-start}"
                  f"/{NCYCLES*CYCLE_LENGTH} points")
        el_in_stim_set_i = stim_mapping.loc[config_i,stim_set_i].values
            
        padding = int(.1 * SR) # seconds
        traces = read_data(path, config_i, convert2vol=True, row_slice=all_chnls,
                           col_slice=slice(max(0, start-padding), 
                                           min(dac_data.size, stop+padding)))
        if debug:
            for el, trace in zip(all_els, traces):
                plt.title(f"Stim set {stim_set_i}: {len(el_in_stim_set_i)}/{len(all_els)}")
                if el in el_in_stim_set_i:
                    lbl = f"E{el}-C{mapping.loc[el].item()}"
                    plt.plot(trace, alpha=.5, label=lbl)
                else:
                    plt.plot(trace, alpha=.05, color='gray')
            plt.legend()
            plt.show()
        
        powers = [estimate_frequency_power(t, SR, 960, 1040, debug=False) for t in traces]
        idx = pd.MultiIndex.from_arrays([[config_i] * len(all_els), 
                                         [stim_set_i] * len(all_els), 
                                         [el in el_in_stim_set_i for el in all_els], 
                                         stim_mapping.loc[config_i,:,:,all_els].index.get_level_values('tile'),
                                         all_els],
            names=['config', 'stim_set', 'stimulated', 'tile', 'el']
        )
        powers = pd.Series(powers, index=idx, name='power')
        config_powers.append(powers)
    config_powers = pd.concat(config_powers)
    return config_powers

def get_n_elsets_for_stimset_chunk_combo(stim_set_dir, chunk_i):
    return len(glob(f"{stim_set_dir}/mxConfig_chunk{chunk_i}_map*.npy"))

def get_resolution():
    return (MAX_AMPL_mV/BITRATE) /GAIN

def convert_to_vol(data):
    # scale data to mv
    res = get_resolution()
    data = (data-BITRATE/2) *res
    # floating/ offset
    data += MAX_AMPL_mV/2 
    return data

def main():
    basepath = "/mnt/SpatialSequenceLearning/Simon/impedance/"
    basepath = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance/"
    device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_neighbours3'
    PATH = basepath + device_name
    if not os.path.exists(PATH):
        print("Path does not exist: ", PATH)
        return
    
    powers = []
    for config_i in range(get_n_configs(PATH)):
        power = extract_stim_traces(PATH, config_i, debug=False)
        powers.append(power)
        pd.concat(powers).to_pickle(f"{PATH}/stim_powers.pkl")
    all_powers = pd.concat(powers)
    
if __name__ == "__main__":
    main()
    