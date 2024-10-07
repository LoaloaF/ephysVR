import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

PATH = "/mnt/SpatialSequenceLearning/Simon/impedance/device_4983/impedance_rec3_externalCurrent2/"
fname = "all_recording.raw.h5"


with h5py.File(os.path.join(PATH,fname), 'r') as file:
    data = np.array(file['sig'][:1025, :])
    
    
for i, row in enumerate(data):
    if i <924: continue
    
    plt.plot(row+i*1000)
    
    # if i > 100:
    #     break
    
plt.show()