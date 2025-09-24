import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# from tqdm import tqdm
import mat73

def load_data(folder):
    # Load the .mat file
    data_dict = mat73.loadmat(folder + '/ephys_755_res.mat')
    
    # Load the filtered spikes file
    spikesFilt_dir = folder + '/ephys_755_filt.jrc'
    
    # Number of clusters
    num_clusters = len(data_dict['clusterNotes'])
    
    # Identify single units
    single_unit = np.zeros(num_clusters)
    for i in range(num_clusters):
        if data_dict['clusterNotes'][i] == ['single']:
            single_unit[i] = 1
    
    # Spike times
    spikeTimes = data_dict['spikeTimes']
    print("spikeTimes ", len(spikeTimes), type(spikeTimes), spikeTimes)
    
    # Spikes by cluster
    spikesByCluster = data_dict['spikesByCluster']
    print("spikesByCluster ", len(spikesByCluster), type(spikesByCluster), type(spikesByCluster[0]), len(spikesByCluster[0]))
    
    # Cluster sites
    clusterSites = data_dict['clusterSites'].astype('int')
    print("clusterSites ", len(clusterSites), type(clusterSites), clusterSites)
    
    # Spike sites
    spikeSites = data_dict['spikeSites']
    print("spikeSites ", len(spikeSites), type(spikeSites), spikeSites)
    
    # Load filtered spikes
    spikesFilt = np.fromfile(spikesFilt_dir, 'int16') #* 0.195
    print("spikesFilt ", len(spikesFilt), type(spikesFilt))
    
    # Reshape filtered spikes
    # Shape: (41 channels, 20 samples per spike, number of spikes)
    spikesFilt_rs = np.reshape(spikesFilt, [41, 20, len(spikeTimes)], 'F')
    print(spikesFilt_rs.shape)
    
    data = calculate_snr(folder, sr=20_000, total_num_channels=755)
    print(data.shape)
    
    return spikesByCluster, spikesFilt_rs, spikeTimes

def plot_spike_waveforms(spikesByCluster, spikesFilt_rs, clusters, color):
    for cluster in clusters:
        cluster_spikes = spikesByCluster[cluster][0].astype(int) - 1
        cluster_spike_wavs = spikesFilt_rs[:, 0, cluster_spikes]
        plt.figure(figsize=[5, 5])
        plt.plot(cluster_spike_wavs[:, 0:500], color, alpha=0.05)
        plt.title(f'Cluster {cluster}')
        plt.show()

def calculate_snr(folder, sr, total_num_channels):
    # Location of the raw data
    raw_data_file = folder + '/ephys_755.dat'
    
    # Arrays containing the root mean square of baseline activity for each session and each recording contact
    rms = np.zeros(total_num_channels)
    
    # Designing the bandpass filter
    b, a = signal.butter(4, [500, 5000], 'bandpass', fs=sr)
    
    print("Reading raw data...")
    data = np.memmap(raw_data_file, dtype='int16', mode='r', order='F')
    print("Done.")
    
    # Calculate the number of time points
    num_timepoints = data.size // total_num_channels
    print(f"Total number of time points: {num_timepoints}")
    
    # Reshape the data to (total_num_channels, num_timepoints)
    data = data.reshape((total_num_channels, num_timepoints), order='F')
    print("Original data shape:", data.shape)
    
    return data

    data = np.fromfile(raw_data_file, 'int16')[:, :1_000_000] #* 0.195
    data_rs = np.reshape(data, [total_num_channels, -1], 'F')
    data_bp = signal.filtfilt(b, a, data_rs, axis=1)
    rms = np.sqrt(np.mean(np.square(data_bp), axis=1))
    
    return rms

def main():
    folder = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Peter/2024-11-04_16-13_rYL006_P1000_MotorLearningStop_22min"
    sr = 20000
    total_num_channels = 755
    
    # Load data
    spikesByCluster, spikesFilt_rs, spikeTimes = load_data(folder)
    
    # Plotting the spike waveforms of sample units from dHPC
    dHPC_clusters = [14, 18, 19]
    # plot_spike_waveforms(spikesByCluster, spikesFilt_rs, dHPC_clusters, 'g')
    
    # Plotting the spike waveforms of sample units from iHPC
    iHPC_clusters = [6, 3, 9]
    # plot_spike_waveforms(spikesByCluster, spikesFilt_rs, iHPC_clusters, 'b')
    
    # Calculate SNR
    # rms = calculate_snr(folder, sr, total_num_channels)
    # print("RMS values: ", rms)

if __name__ == "__main__":
    main()