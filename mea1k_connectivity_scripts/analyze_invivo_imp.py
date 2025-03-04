import os
import sys

# to import from parent dir
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from CustomLogger import CustomLogger as Logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ephys_constants as C
from mea1k_modules.mea1k_raw_preproc import read_raw_data
from mea1k_modules.mea1k_raw_preproc import read_stim_DAC
from mea1k_modules.mea1k_raw_preproc import get_recording_implant_mapping
from mea1k_modules.mea1k_visualizations import vis_shank_traces
# from signal_helpers import estimate_frequency_power


from mea1k_viz import draw_mea1k

def get_hdf5_fnames_from_dir(subdir):
    fnames, ids = [], []
    for fname in sorted(os.listdir(subdir)):
        if fname.endswith('raw.h5'):
            fnames.append(fname)
            # check 4 digit case...
            pruned_fname = fname.replace('.raw.h5', '')
            id_str = pruned_fname.split('_')[-1]
            ids.append(id_str)
            # try:
            #     ids.append(int(id_str))
            # except ValueError:
            #     print(f"Error: {id_str} is not a valid id. Full name: {fname}")
            #     exit(1)
    return fnames, ids

def save_output(subdir, data, fname):
    fullpath = os.path.join(subdir, "processed")
    if not os.path.exists(fullpath):
        print("creating processed output dir")
        os.makedirs(fullpath)
    data.to_csv(os.path.join(fullpath, fname))

def extract_impedance(subdir, implant_name, debug=False):
    fnames, ids = get_hdf5_fnames_from_dir(subdir)
    all_data = []
    for fname, i in zip(fnames, ids):
        print(f"Config {i} of {len(fnames)}")
        
        stimulated = pd.read_csv(os.path.join(subdir, fname.replace(".raw.h5", ".csv")))
        stim_mea1k_el = stimulated[stimulated.stim].electrode.values
        print(stim_mea1k_el)

        # if fname == "el_config_S1D1650.raw.h5":
        if True:
            dac = read_stim_DAC(subdir, fname)
            data = read_raw_data(subdir, fname, convert2uV=True,
                                subtract_dc_offset=False, col_slice=slice(None, None, None))
            print(data.min(), data.max())
            mapping = get_recording_implant_mapping(subdir, fname, implant_name=implant_name)
            vis_shank_traces(data, mapping, stim_mea1k_el=stim_mea1k_el, scaler=4/1000)
            
            plt.subplot(2, 1, 1)
            plt.plot(data.T)
            plt.subplot(2, 1, 2, sharex=plt.gca())  
            plt.plot(dac)
            plt.show()
        
        continue
        
        

        # print("Filtering...")
        # mean_ampl = []
        # for j,row in enumerate(data.values):
        #     debug = True if j <10 and debug else False
        #     _, m_ampl = estimate_frequency_power(row, sampling_rate=C.SAMPLING_RATE, 
        #                                          debug=debug, min_band=960, max_band=1040)
        #     mean_ampl.append(m_ampl)
        
        # data = pd.DataFrame(mean_ampl, index=data.index, columns=['ampl'])
        # data['connectivity'] = data.ampl.values/input_ampl_mV 
        # data['input_ampl_mV'] = input_ampl_mV
        # data.index = pd.MultiIndex.from_product([[i],data.index], names=['config', 'el'])
        # print(f"Done. n >80%: {(data.connectivity >.8).sum()}\n")

        
def vis_connectivity(subdir, input_ampl_mV, cmap_scaler=2.5):
    fullfname = os.path.join(subdir, "processed", f"extr_connectivity.csv")
    data = pd.read_csv(fullfname)
    data.set_index('el', inplace=True)  
    print(data)    
    plt.hist(data['ampl'], bins=100)
    plt.show()
    (fig,ax), el_recs = draw_mea1k()
    for el_i, el_recs in enumerate(el_recs):
        if el_i not in data.index:
            print("missing", el_i, end=' ')
            continue
        # needed for new local configs that were used for a hort time
        if data.loc[el_i].shape[0] == 2:
            print("duplicate", el_i, end=' ')
            continue
        whiteness = np.clip(data.loc[el_i].connectivity*cmap_scaler, 0, 1)
        el_recs.set_facecolor((whiteness, whiteness, whiteness))
    fig.savefig(fullfname.replace(".csv", ".png"), dpi=300, transparent=False, 
                bbox_inches='tight', pad_inches=0)  
    plt.show()
    
    # save a second version with the colors inverted
    img = plt.imread(fullfname.replace(".csv", ".png"))
    img = img[:,:,:3]
    # invert colors
    img = (img*-1) + 1 
    plt.imshow(img)
    plt.axis('off')
    
    # save inverted
    plt.imsave(fullfname.replace(".csv", "_inverted.png"), img)
    
def main(): 
    L = Logger()
    L.init_logger(None, None, "DEBUG")
    L.logger.debug("Starting in vivo impedance analysis")
    nas_dir = C.device_paths()[0]
    
    implant_name = "241016_MEA1K03_H1278pad4shankB5"
    subdirs = [
        f"devices/implant_devices/{implant_name}/recordings/14:56:47_invivo_localstim_mode='small_current'_stimpulse='sine'_amplitude=3"
    ]
    # el_config_S1D1650.raw.h5
    extract_impedance(os.path.join(nas_dir, subdirs[0]), implant_name=implant_name)
    
    # subdir = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/RUN_rYL006/rYL006_P1100/2025-01-27_13-39_rYL006_P1100_LinearTrackStop_73min"
    # data = read_raw_data(subdir, "ephys_output.raw.h5", convert2mV_float16=True, col_slice=slice(20_000, 30_000), to_df=True, subtract_dc_offset=True)
    # mapping = get_recording_implant_mapping(subdir, "ephys_output.raw.h5", animal_name="rYL006", exclude_shanks=[2,3,4])
    # mapping = mapping[mapping.depth < 2000]
    # mapping = mapping[mapping.shank_side == 'left']
    # data = data.loc[mapping.mea1k_el]
    # vis_shank_traces(data.values, mapping, scaler=180)
    
if __name__ == "__main__":
    main()