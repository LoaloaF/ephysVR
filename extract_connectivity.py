import os
from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import scipy.signal as signal
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert

import pandas as pd
import multiprocessing as mp
from functools import partial

import pandas as pd
    
MAX_AMPL_mV = 2900.
BITRATE = 2**10
GAIN = 1
SR = 20_000
CYCLE_LENGTH = 20
NCYCLES = 100

def bandpass_filter(signal, sampling_rate, lowcut, highcut, order=4):
    nyquist = 0.5 * sampling_rate  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Create a Butterworth band-pass filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter to the signal using filtfilt for zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def extract_average_amplitude(signal):
    # Compute the analytic signal using the Hilbert transform
    analytic_signal = hilbert(signal)
    # Compute the envelope (magnitude of the analytic signal)
    amplitude_envelope = np.abs(analytic_signal)
    # Compute the average amplitude of the envelope
    average_amplitude = np.mean(amplitude_envelope)
    return average_amplitude, amplitude_envelope

def estimate_frequency_power(signal, sampling_rate, min_band, max_band, debug=False):
    m = signal.mean()
    signal -= m
    
    # Compute the FFT of the signal
    fft_result = np.fft.fft(signal)
    # Compute the power spectrum
    power_spectrum = np.abs(fft_result) ** 2
    # Compute the corresponding frequencies
    freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    # Only keep the positive frequencies
    positive_freqs = freqs[freqs >= 1]
    positive_power_spectrum = power_spectrum[freqs >= 1]
    power_1KHz = positive_power_spectrum[(positive_freqs > min_band) & 
                                         (positive_freqs < max_band)].max()
    
    
    signal_1khz = bandpass_filter(signal, sampling_rate, min_band, max_band)
    mean_ampl, _ = extract_average_amplitude(signal_1khz)
    
    if debug:
        fig, ax = plt.subplots(3, 1, figsize=(12, 6))
        fig.subplots_adjust( hspace=.5)
        fig.suptitle("Voltage with external 2uA 1KHz sine signal")
        
        t = np.arange(len(signal))/sampling_rate *1000
        ax[0].plot(t, signal, color='blue', alpha=.8, label='Signal')
        ax[0].set_xlabel('Time [ms]')
        ax[0].set_yticks((-10,0,10))
        ax[0].set_ylabel(f'Δ Potential\nfrom {m:.0f} mV')
        ax[0].grid(True)
        [ax[0].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
        ax[0].legend()
        
        ax[1].plot(positive_freqs, positive_power_spectrum, color='orange',
                   label='Power Spectrum')
        ax[1].scatter([1000], power_1KHz, edgecolor='red', facecolor='none', 
                      label=f'1KHz Power: {power_1KHz:.1e}', s=100)
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Power')
        ax[1].grid(True)
        [ax[1].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
        ax[1].legend()
        
        ax[2].plot(t, signal_1khz, color='blue', alpha=.5,
                   label='1KHz Bandpass Filtered Signal')
        ax[2].plot([t[0]-20,t[-1]+20], [mean_ampl,mean_ampl], color='k', 
                   linestyle='dashed', label=f'Average Amplitude: {mean_ampl:.1f} mV')
        ax[2].set_xlabel('Time [ms]')
        ax[2].set_ylabel('Amplitude')
        ax[2].set_ylabel(f'Δ Potential\nfrom {m:.0f} mV')
        ax[2].set_yticks((-10,0,10))
        ax[2].grid(True)
        ax[2].sharex(ax[0])
        [ax[2].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
        ax[2].legend()
        plt.show()
    return power_1KHz, mean_ampl

def get_n_configs(path):
    n = len(glob(os.path.join(path, "el_config_*.raw.h5")))
    # n = len(glob(os.path.join(path, "config_*.raw.h5")))
    print(f"Found {n} configurations\n")
    return n

def get_config_names(path):
    configs = glob(os.path.join(path, "config_*.raw.h5"))
    configs = [int(c[c.rfind('config_set')+11:c.rfind('config_set')+16]) for c in configs]
    configs = sorted(configs)
    print(f"Found {len(configs)} configurations\n")
    # print(configs)
    return configs

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
    if fname not in os.listdir(path):
        fname = f'config_set_{config_i:02}_0.raw.h5'
        if fname not in os.listdir(path):
            fname = f'config_set_{config_i:05}.raw.h5'
        
            if fname not in os.listdir(path):
                fname = f'el_config_{config_i:03}.raw.h5'
            
    with h5py.File(os.path.join(path, fname), 'r') as file:
        # print(file.keys())
        # print(file["settings"]['gain'].keys())
        # exit()
        mapping = np.array([list(m) for m in file['mapping']])
        channels = mapping[:, :2].astype(int)
        channels = pd.Series(channels[:,0], index=channels[:,1]).sort_values()
        el_xy = mapping[:, 2:4]
    return channels, el_xy

def get_stim_mapping(path):
    df = pd.read_csv(f"{path}/stim_mapping.csv", index_col=[0,1,2,3]).iloc[:,0]
    print(df)
    return df
    exit()
    # return pd.read_pickle(f"{path}/stim_mapping.pkl")

def read_data(path, config_i, convert2vol=False, row_slice=slice(None), col_slice=slice(None)):
    fname = f"config_{config_i:03d}.raw.h5"
    if fname not in os.listdir(path):
        fname = f'config_set_{config_i:02d}_0.raw.h5'
        if fname not in os.listdir(path):
            fname = f'config_set_{config_i:05}.raw.h5'
        
            if fname not in os.listdir(path):
                fname = f'el_config_{config_i:03}.raw.h5'
            
    with h5py.File(os.path.join(path, fname), 'r') as file:
        print("Reading data...", flush=True)
        data = np.array(file['sig'][row_slice, col_slice])
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
        
        powers = [estimate_frequency_power(t, SR, 960, 1040, debug=False)[0] for t in traces]
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

   
# def get_el_coords_from_mapping(path, ):
#     # all_coos = []
#     # for config_i in range(get_n_configs(path)):
#     #     el_chnl, coo = get_h5_mapping(path, config_i)
#     #     all_coos.append(pd.DataFrame(coo, index=el_chnl.index, columns=['x', 'y']))
#     # all_coos = pd.concat(all_coos).sort_index()
#     # all_coos.to_pickle(f"{path}/el_coords.pkl")
#     # print(all_coos)
#     # for checking
#     all_coos = pd.read_pickle(f"{path}/el_coords.pkl")
    
#     fig, ax = plt.subplots(figsize=(3850/300, 2100/300), facecolor='none')
#     plt.scatter(all_coos['x']+9, all_coos['y']+9, s=10, color='black')
#     fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    
#     # draw 26400 colors from hsv colormap
#     # cmap = plt.cm.get_cmap('hsv', 26400)
#     cmap = plt.get_cmap('hsv', 26400)
#     colors = [list(col) for col in cmap(np.linspace(0, 1, 26400))]
    
#     i = 0
#     for y in np.arange(0, 2100, 17.5):
#         for x in np.arange(0, 3850, 17.5):
#             col = colors[i]
#             ax.add_patch(plt.Rectangle((x+4.5, y+4.5), 9, 9, facecolor=col, alpha=.5, edgecolor='none'))
#             i += 1
#         # plt.axvline(x, color='gray', alpha=.5)
#     ax.set_ylim(2100, 0)
#     ax.set_xlim(0,3850)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_aspect('equal', adjustable='box')
    
#     plt.savefig(f"{path}/el_pads_v1.png", dpi=300, transparent=True, 
#                 bbox_inches='tight', pad_inches=0, )
#     plt.show()

def extract_traces(path, config_i, debug=False):
    print(f"Extracting traces for config {config_i:03d}", flush=True)
    mapping, _ = get_h5_mapping(path, config_i)
    # chnl_i = np.where(mapping.index == config_i)[0][0]
    # chnl_i = 0
    
    mapping = mapping.sort_values()
    traces = read_data(path, config_i, convert2vol=True, row_slice=mapping.values,)
    # print(traces.shape)
    # traces = pd.DataFrame(traces, index=mapping.values, )
    
    # for i,t in enumerate(traces):
    #     s = bandpass_filter(t-t[0], SR, 960, 1040)
    #     # print(sorted(s)[-10:])
    #     # print(s.max(), end='...')
    #     if s.max() > 3:
    #         print(f"Channel {mapping.index[i]}, {i}")
    #     plt.plot(s, alpha=.5)
    # plt.show()
    
    powers_ampl = [estimate_frequency_power(t, SR, 970, 1030,
                                            debug=True) 
                   for i,t in enumerate(traces)]
    powers, amplitudes = zip(*powers_ampl)
    
    data = pd.DataFrame({"power": powers, 'ampl': amplitudes}, index=mapping.index)
    return data.loc[config_i].to_frame().T
    # return data
 
def extract_traces_parallel(PATH, config_i, debug=False):
    try:
        power_ampl = extract_traces(PATH, config_i, debug)
    except Exception as e:
        print(f"Error processing config {config_i}: {e}", flush=True)
        return None
    return power_ampl

def parallel_extract_traces(PATH, debug=False):
    config_names = get_config_names(PATH)
    
    # Create a pool of workers
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use partial to pass the PATH and debug arguments to the function
        func = partial(extract_traces_parallel, PATH, debug=debug)
        # Map the function to the list of configurations
        powers = pool.map(func, config_names)
    
    # Concatenate the results and save to a pickle file
    pd.concat(powers).to_pickle(f"{PATH}/extracted_signal.pkl")

def main():
    basepath = "/mnt/SpatialSequenceLearning/Simon/impedance/"
    basepath = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance/"
    basepath = "/mnt/SpatialSequenceLearning/"
    device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded_extCurrent1024_rec2'
    # device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded_extCurrent_singleAll'
    device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded_extCurrent_singleElAll'
    device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded_extCurrent_singleElAll'
    device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_ext1KHz_Current_singelEl_rec1'
    # device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_dry_ext1KHz_rec4'
    # device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_meshstim_rec1'
    device_name = "headstage_devices/MEA1K05/recordings/25mVext_oneShankbatch2_press"
    PATH = basepath + device_name
    # PATH = "/Users/loaloa/local_data/impedance_bonded_extCurrent_singleAll"
    print(PATH)
    
    
    if not os.path.exists(PATH):
        print("Path does not exist: ", PATH)
        return
    
    powers = []
    # for config_i in range(get_n_configs(PATH)):
    #     if config_i >= 90:
    #         power = extract_stim_traces(PATH, config_i, debug=False)
    #         powers.append(power)
    #         pd.concat(powers).to_pickle(f"{PATH}/extracted_signal2.pkl")
    
    # mea1k = np.arange(26400).reshape(120,220)
    # x, y = np.meshgrid(np.arange(39,39+15), np.arange(9,9+15))
    config_names = get_config_names(PATH)
    # config_names = [cn for cn in config_names if cn in mea1k[x, y]]
    
    # for config_i in config_names:
    for config_i in range(get_n_configs(PATH)):
        print(config_i)
        # if config_i == 106:
        power = extract_traces(PATH, config_i, debug=False)
        powers.append(power)
        pd.concat(powers).sort_values('ampl').to_pickle(f"{PATH}/extracted_signal.pkl")
    
    
    # print(f"Done\n{PATH}/extracted_signal2.pkl")
    # parallel_extract_traces(PATH, debug=False)
    # parallel_extract_traces_multithreading(PATH, debug=False)
    
if __name__ == "__main__":
    main()