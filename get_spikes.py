import numpy as np
import mat73
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

fname = "ephys_755"
path = '/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Peter/2024-11-04_16-13_rYL006_P1000_MotorLearningStop_22min'
data_dict = mat73.loadmat(f'{path}/{fname}_res.mat')
spikesFilt_dir = f'{path}/{fname}_filt.jrc'
num_clusters = len(data_dict['clusterNotes'])

single_unit = np.zeros(num_clusters)
for i in range(num_clusters):
    if data_dict['clusterNotes'][i] == ['single']:
        single_unit[i] = 1

spikeTimes = data_dict['spikeTimes']
spikesByCluster = data_dict['spikesByCluster']
clusterSites = data_dict['clusterSites'].astype('int')
spikeSites = data_dict['spikeSites']
spikesFilt = np.fromfile(spikesFilt_dir, 'int16') * 0.195

# Print shapes and lengths of the data
print("Number of clusters:", num_clusters)
print("Shape of single_unit:", single_unit.shape)
print("Length of spikeTimes:", len(spikeTimes))

# Print type and length of each element in spikesByCluster
print("Type of spikesByCluster:", type(spikesByCluster))
print("Length of spikesByCluster:", len(spikesByCluster))
for i, cluster in enumerate(spikesByCluster):
    print(f"Type of spikesByCluster[{i}]:", type(cluster))
    print(f"Length of spikesByCluster[{i}]:", len(cluster))
    if isinstance(cluster, list):
        for j, sublist in enumerate(cluster):
            print(f"  Type of spikesByCluster[{i}][{j}]:", type(sublist))
            print(f"  Length of spikesByCluster[{i}][{j}]:", len(sublist))

print("Shape of clusterSites:", clusterSites.shape)
print("Shape of spikeSites:", spikeSites.shape)
print("Size of spikesFilt:", spikesFilt.size)

channels = 20
spikesFilt_rs = np.reshape(spikesFilt, [41, channels, len(spikeTimes)], 'F')
print("Shape of spikesFilt_rs:", spikesFilt_rs.shape)

spikeClusters = data_dict['spikeClusters']
print("Shape of spikeClusters:", np.shape(spikeClusters))

# Create a DataFrame with spike times as index, cluster IDs, and recording channels
spike_data = pd.DataFrame({'Timestamp': spikeTimes, 'ClusterID': spikeClusters, 'Channel': spikeSites})
spike_data.set_index('Timestamp', inplace=True)

print()
print()
print(spike_data)
spike_data.to_csv('spike_data.csv')

# Make a raster plot of the spike times, each row is a different unit, colored by channel
plt.figure(figsize=[10, 5])
unique_channels = np.unique(spike_data['Channel'])
colors = plt.cm.get_cmap('tab20', len(unique_channels))

for idx, chnl in enumerate(unique_channels):
    channel_spikes = spike_data[spike_data['Channel'] == chnl]
    plt.plot(channel_spikes.index, np.ones_like(channel_spikes.index) * chnl, '|', markersize=2, color=colors(idx))

plt.ylim(np.max(spike_data['Channel']) + 1, np.min(spike_data['Channel']) - 1)
plt.xlabel('Time')
plt.ylabel('Channel')
plt.title('Raster Plot of Spike Times by Channel')
plt.show()

# Plotting the spike waveforms of sample units from dHPC 
cluster = 14
cluster_spikes = spikesByCluster[cluster][0].astype(int) - 1
print("Shape of cluster_spikes:", cluster_spikes.shape)
cluster_spike_wavs = spikesFilt_rs[:, 0, cluster_spikes]

plt.figure(figsize=[5, 5])
plt.plot(cluster_spike_wavs[:, 0:500], 'g', alpha=0.05)
plt.show()