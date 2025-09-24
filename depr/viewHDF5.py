import h5py 
import numpy as np
import os
import matplotlib.pyplot as plt

from extract_connectivity import get_config_mapping
from extract_connectivity import get_stim_mapping

def create_stim_sine_sequence(dac=0, amplitude=25, f=1000, ncycles=300, nreps=1):
    # seq = maxlab.Sequence()
    # 50 us * 20kHz = 1000 samples, 1 khz example
    sampling_rate = 20000  # 20 kHz
    # Create a time array
    t = np.linspace(0, 1, int(sampling_rate / f))
    # Create a sine wave with a frequency of 1 kHz
    sine_wave = (amplitude * np.sin(t * 2 * np.pi)).astype(int) + 512
    debug = []
    for i in range(nreps):
        # seq.append(maxlab.system.DelaySamples(40))
        for j in range(ncycles):
            for ampl in sine_wave:
                # seq.append(maxlab.chip.DAC(dac, ampl))
                debug.append(ampl)
    plt.plot(debug)
    np.save("sine_wave.npy", debug)
    return seq

def estimate_frequency_powers(signal, sampling_rate):
    print("in")
    # Compute the FFT of the signal
    fft_result = np.fft.fft(signal)
    
    # Compute the power spectrum
    power_spectrum = np.abs(fft_result) ** 2
    
    # Compute the corresponding frequencies
    freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    
    # Only keep the positive frequencies
    positive_freqs = freqs[freqs >= 1]
    positive_power_spectrum = power_spectrum[freqs >= 1]
    
    return (positive_power_spectrum[(positive_freqs>5600) & (positive_freqs<6400)].mean())
    
    # Plot the power spectrum
    # plt.figure(figsize=(10, 6))
    # plt.plot(positive_freqs, positive_power_spectrum)
    # plt.yscale('log')
    # plt.title('Power Spectrum')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power')
    # plt.grid(True)
    
    # plt.show()
    
    
def check_stim(path, fname):
    mapping = get_config_mapping(path).loc[0]
    stim_mapping = get_stim_mapping(path).loc[0]
    print(mapping)
    print(stim_mapping)
    
    meak1 = np.zeros((120,220))
    for stim_set_i in stim_mapping.index.unique('stim_set'):
        print(stim_set_i)
        els = stim_mapping.loc[stim_set_i].values
        meak1 = meak1.flatten()
        meak1[els] += mapping.loc[:,els].index.get_level_values('tile')+1*5
        meak1 = meak1.reshape(120,220)
        plt.imshow(meak1)
        plt.show()
    
    
    
basepath = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance/"
basepath = "/mnt/SpatialSequenceLearning/Simon/impedance/"
device_name = 'device_headmount_new2EpoxyWalls/impedance_bonded_neighbours3'
PATH = basepath + device_name
print(PATH)
fname = "config_000.raw.h5"

# check_stim(PATH, fname)

with h5py.File(os.path.join(PATH, fname), 'r') as file:
    # print(np.array(file['mapping']))
    dac = np.array(file['sig'][1024, :])
    plt.plot(dac)
    plt.show()
    # data = np.array(file['sig']).astype(int)
    # data = np.array(file['sig'][:1025, 0:6000]).astype(int)
    # data = np.array(file['sig'][:1025, 6000:10000]).astype(int)

# # estimate_frequency_powers(dac, 20000)

# j = 0
# for i, row in enumerate(data):
#     # if i < 800: continue
#     row -= row[0]
#     # if (abs(row[1000:1150]) > 240).any():
#     # if (abs(np.diff(row)) > 200 ).any():
#     if True:
#         ps = estimate_frequency_powers(row, 20000)
#         if ps > 200000:
#             j += 1
#             # plt.scatter(i, ps)
            
#             plt.plot(row, alpha=.4)# + i * 1000)
# print(j)
# plt.show()
    
# create_stim_sine_sequence()
# plt.show()
