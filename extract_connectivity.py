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

# def high_pass_filter(data, cutoff, fs, order=5):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
#     filtered_data = signal.filtfilt(b, a, data)
#     return filtered_data

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

def get_h5_mapping(path, config_i):
    fname = f'config_{config_i:03d}.raw.h5'
    with h5py.File(os.path.join(path, fname), 'r') as file:
        mapping = np.array([list(m) for m in file['mapping']])
        channels = mapping[:, :2].astype(int)
        channels = pd.Series(channels[:,0], index=channels[:,1]).sort_values()
        el_xy = mapping[:, 2:4]
    return channels, el_xy

# def get_config_mapping(path, compare_config_file_mapping=None):
#     levels = ['config', 'tile', 'el']
#     # Read the CSV file
#     df = pd.read_csv(os.path.join(path, "recording_mapping.csv"))
#     # Split the tuple column into separate columns
#     df[levels] = df['Unnamed: 0'].str.extract(r'\((\d+), (\d+), (\d+)\)')
#     # Convert the new columns to integers
#     df[levels] = df[levels].astype(int)
#     # Set the new columns as the multi-index
#     df.set_index(levels, inplace=True)
#     # Drop the original tuple column
#     channels = df.drop(columns=['Unnamed: 0']).iloc[:,0]
    
#     if compare_config_file_mapping is not None:
#         channels_h5, _ = get_h5_mapping(path, compare_config_file_mapping)
#         print(channels_h5)
#         print(channels)
#         channels.loc[channels_h5.index] = channels_h5.values
#         print(channels)
        
#         exit()
#         mapping_csv = channels.loc[compare_config_file_mapping].droplevel('tile')
            
        
#         if not (channels_h5.index == mapping_csv.index).all() or \
#             not (channels_h5.values == mapping_csv.values).all():
#             print("Mismatch between csv and h5 mapping file")
#             print(channels_h5)
#             print(mapping_csv)
            
            
            
#             exit()
#             # mapping_csv_chnls = mapping_csv.loc[:, (channels_h5[:,1])]
#             # print(remap)
#             # print(remap.index['el'])
#             # print(remap.index['el'])
#             exit()
#             # if remap.values == mapping_csv
#             # channels_h5 = channels_h5[channels_h5[:,0].argsort()]
#             # print(channels_h5.shape)
            
#             # print(config_channels)
#             # # print(el_chnl)
#             # print(config_channels.loc[:, el_chnl[:,1][config_channels.values]])
            
#             # if config_channels.loc[:, el_chnl[:,1][config_channels.values]] != el_xy[:,0]:
#             #     print("Mismatch between mapping and config file")
#             # print(el_chnl)
#             exit()
            
    
    
    return channels


def get_stim_mapping(path):
    return pd.read_pickle(f"{path}/stim_mapping.pkl")

def read_data(path, config_i, row_slice=slice(None), col_slice=slice(None)):
    fname = f"config_{config_i:03d}.raw.h5"
    with h5py.File(os.path.join(path, fname), 'r') as file:
        print("Reading data...", flush=True, end="")
        data = np.array(file['sig'][row_slice, col_slice])
        print("done.", flush=True)
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
        print("Mismatch between number of stim sets and number of stim sequences")
        exit()
        
    all_els = stim_mapping[config_i].index.get_level_values('el')
    all_chnls = mapping.loc[all_els]
    reindexer = np.argsort(all_chnls)
    all_els, all_chnls = all_els[reindexer], all_chnls.iloc[reindexer]
    
    for stim_set_i, (start, stop) in enumerate(zip(stim_sequence_indices[::2], 
                                                   stim_sequence_indices[1::2])):
        if start-stop != NCYCLES*CYCLE_LENGTH:
            print(f"Stim set {stim_set_i}: {start} - {stop}, OFF: {stop-start}"
                  f"/{NCYCLES*CYCLE_LENGTH} points")
        el_in_stim_set_i = stim_mapping.loc[config_i,stim_set_i].values
            
        padding = int(.01 * SR) # seconds
        traces = read_data(path, config_i, row_slice=all_chnls,
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
                                         all_els],
            names=['config', 'stim_set', 'stimulated', 'el']
        )
        powers = pd.Series(powers, index=idx, name='power')
        return powers

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
    basepath = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance/"
    basepath = "/mnt/SpatialSequenceLearning/Simon/impedance/"
    device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_neighbours3'
    PATH = basepath + device_name
    if not os.path.exists(PATH):
        print("Path does not exist: ", PATH)
        return
    
    # all_powers = pd.read_pickle(f"{PATH}/stim_powers.pkl").loc[1:1,]
    # connectivity = all_powers[all_powers.index.get_level_values('stimulated')==False].groupby("el").sum()
    # print(connectivity)
    
    # mea1k = np.zeros(26400, dtype=float)
    # print(np.min(mea1k))
    # print((mea1k==0).sum())
    # mea1k[connectivity.index] = connectivity.values
    # print((mea1k==0).sum())
    # print(len(np.unique(connectivity.index)))
    # print(np.min(mea1k))
    
    # # exit()
    # mea1k = mea1k.reshape(120,220)
    
    # cmap = plt.cm.get_cmap('viridis')
    # # set 0 to black
    # cmap.set_bad('black')
    # mea1k[mea1k==0] = np.nan
    # plt.imshow(mea1k, cmap=cmap)
    # plt.show()
    
    
    
    powers = []
    for config_i in range(get_n_configs(PATH)):
        power = extract_stim_traces(PATH, config_i, debug=True)
        powers.append(power)
        pd.concat(powers).to_pickle(f"{PATH}/stim_powers.pkl")
    all_powers = pd.concat(powers)
    
    # # save
    
    
    print(all_powers)
    # get_n_configs(PATH)
    # o = get_config_mapping(PATH, 0)
    # o = get_stim_mapping(PATH, 0)
    # print(o)
    
    

    # stim_ttls = extract_stim_events(PATH, precomputed='to', debug=False)
    # extract_eletrode_pulses(PATH, stim_ttls, debug=False)

if __name__ == "__main__":
    main()
    