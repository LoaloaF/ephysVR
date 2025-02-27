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
from signal_helpers import estimate_frequency_power

from mea1k_viz import draw_mea1k

def get_hdf5_fnames_from_dir(subdir):
    fnames, ids = [], []
    for fname in sorted(os.listdir(subdir)):
        if fname.endswith('raw.h5'):
            fnames.append(fname)
            # check 4 digit case...
            pruned_fname = fname.replace('.raw.h5', '')
            if pruned_fname[-4].isdigit():
                ids.append(int(pruned_fname[-4:]))
            else:
                ids.append(int(pruned_fname[-3:]))
    return fnames, ids

def save_output(subdir, data, fname):
    fullpath = os.path.join(subdir, "processed")
    if not os.path.exists(fullpath):
        print("creating processed output dir")
        os.makedirs(fullpath)
    data.to_csv(os.path.join(fullpath, fname))

def extract_connectivity(subdir, input_ampl_mV, n_samples, debug=False):
    fnames, ids = get_hdf5_fnames_from_dir(subdir)
    all_data = []
    for fname, i in zip(fnames, ids):
        print(f"Config {i} of {len(fnames)}")
        data = read_raw_data(subdir, fname, convert2mV_float16=True, to_df=True,
                             col_slice=slice(0, n_samples))

        print("Filtering...")
        mean_ampl = []
        for j,row in enumerate(data.values):
            debug = True if j <10 and debug else False
            _, m_ampl = estimate_frequency_power(row, sampling_rate=C.SAMPLING_RATE, 
                                                 debug=debug, min_band=960, max_band=1040)
            mean_ampl.append(m_ampl)
        
        data = pd.DataFrame(mean_ampl, index=data.index, columns=['ampl'])
        data['connectivity'] = data.ampl.values/input_ampl_mV 
        data['input_ampl_mV'] = input_ampl_mV
        data.index = pd.MultiIndex.from_product([[i],data.index], names=['config', 'el'])
        print(f"Done. n >80%: {(data.connectivity >.8).sum()}\n")

        all_data.append(data)
    save_output(subdir, pd.concat(all_data), f"extr_connectivity.csv")
        
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
    L.logger.info("Starting connectivity analysis")
    
    input_ampl_mV = 0.8
    n_samples = 20_000
    
    subdirs = [
        # "devices/headstage_devices/MEA1K03/recordings/bonding2_250205_D9_25mVext_2_2Shankbatch5_silk/"
        "devices/well_devices/4983/recordings/all_pad_testrec_VrefFPGAStim_ampl110_rec5",
    ]
    
    nas_dir = C.device_paths()[0]
    
    for subdir in subdirs:
        subdir = os.path.join(nas_dir, subdir)
        if not os.path.exists(os.path.join(subdir)):
            print(f"Error: {os.path.join(subdir)} does not exist.")
            continue
        
        extract_connectivity(subdir, input_ampl_mV, n_samples, debug=True)
        vis_connectivity(subdir, input_ampl_mV, cmap_scaler=1)
    
if __name__ == "__main__":
    main()
