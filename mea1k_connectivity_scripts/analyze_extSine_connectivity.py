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

# from mea1k_viz import draw_mea1k
from mea1k_modules.mea1k_visualizations import draw_mea1k

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
        data = read_raw_data(subdir, fname, convert2uV=True, to_df=True,
                             col_slice=slice(0, n_samples))
        data = data / 1000 # convert to mV
    
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
    print(fullfname)
    data = pd.read_csv(fullfname)
    data.set_index('el', inplace=True)  
    print(data)    
    plt.hist(data['ampl'], bins=100)
    plt.show()
    (fig,ax), el_recs = draw_mea1k()
    for el_i, el_recs in enumerate(el_recs):
        if el_i not in data.index:
            print("missing", el_i, end=' ')
            # el_recs.set_edgecolor((.8,0,0))
            continue
        if data.loc[el_i:el_i].connectivity.isna().any():
            # el_recs.set_edgecolor((.8,0,0))
            print("NaN", el_i, end=' ')
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
    
def create_implant_dir(sine_stim_recdir, nas_dir, HEADSTAGE_DEVICE_NAME, IMPLANT_DEVICE_NAME):
    implant_dir = os.path.join(nas_dir, "devices", "implant_devices", IMPLANT_DEVICE_NAME)
    os.makedirs(os.path.join(implant_dir))
    os.makedirs(os.path.join(implant_dir, "recordings"))
    os.makedirs(os.path.join(implant_dir, "bonding"))
    
    # copy the sine stim recording dir to implant dir
    target = os.path.join(implant_dir, "recordings", os.path.basename(sine_stim_recdir))
    os.system(f"cp -r {sine_stim_recdir} {target}")
    print(f"Copied {sine_stim_recdir} to implant recordings dir")

def main():
    L = Logger()
    L.init_logger(None, None, "DEBUG")
    L.logger.info("Starting connectivity analysis")
    nas_dir = C.device_paths()[0]
    
    # bonding which electrode to which headstage
    bonding_date = '250917'
    HEADSTAGE_DEVICE_NAME = 'MEA1K12'
    ELECTRODE_DEVICE_NAME = 'H1628pad1shank'
    batch = 5
    IMPLANT_DEVICE_NAME = f"{bonding_date}_{HEADSTAGE_DEVICE_NAME}_{ELECTRODE_DEVICE_NAME}B{batch}"
    
    # sine stim recording parameters
    rec_name = f'2ndBondTightened_VrefFPGAStim_ampl16'
    input_ampl_mV = 10
    n_samples = 8_000 # where sine stim is visible

    subdir = f"{nas_dir}/devices/headstage_devices/{HEADSTAGE_DEVICE_NAME}/recordings/{rec_name}"
    if not os.path.exists(subdir):
        print(f"Error: {os.path.join(subdir)} does not exist.")
        exit()
    
    extract_connectivity(subdir, input_ampl_mV, n_samples, debug=False)
    vis_connectivity(subdir, input_ampl_mV, cmap_scaler=1)
    create_implant_dir(subdir, nas_dir, HEADSTAGE_DEVICE_NAME, IMPLANT_DEVICE_NAME)
    
    
if __name__ == "__main__":
    main()