import os
import sys

# to import from parent dir
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from CustomLogger import CustomLogger as Logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

import ephys_constants as C
from mea1k_modules.mea1k_raw_preproc import read_raw_data
from mea1k_modules.mea1k_raw_preproc import read_stim_DAC
from mea1k_modules.mea1k_raw_preproc import get_raw_implant_mapping
from mea1k_modules.mea1k_raw_preproc import get_recording_implant_mapping
from signal_helpers import estimate_frequency_power, calculate_phase_shift

from mea1k_modules.mea1k_visualizations import vis_shank_traces, viz_mea1k_config, draw_interconnect_pads
from mea1k_modules.mea1k_visualizations import draw_mea1k
from mea1k_modules.mea1k_visualizations import draw_mea1K_colorbar

def get_hdf5_fnames_from_dir(subdir):
    fnames, ids = [], []
    for fname in sorted(os.listdir(subdir)):
        if fname.endswith('raw.h5'):
            fnames.append(fname)
            # check 4 digit case...
            pruned_fname = fname.replace('.raw.h5', '')
            id_str = pruned_fname.split('_')[-1]
            ids.append(id_str)
    return fnames, ids

def save_output(subdir, data, fname):
    fullpath = os.path.join(subdir, "processed")
    if not os.path.exists(fullpath):
        print("creating processed output dir: ", fullpath)
        os.makedirs(fullpath)
    print("Saving to: ", os.path.join(fullpath, fname))
    data.to_csv(os.path.join(fullpath, fname))

def analyze_shorts(subdir, implant_name, debug=False, deepdebug=False):
    L = Logger()
    fnames, _ = get_hdf5_fnames_from_dir(subdir)
    aggr_imp_data = []
    for i, fname in enumerate(fnames):
        Logger().logger.info(f"Config {i}/{len(fnames)}")
        
        if debug and i > 3:
            continue
        # get the config information about this configuration
        stimulated = pd.read_csv(os.path.join(subdir, fname.replace(".raw.h5", ".csv")))
        
        # dac = read_stim_DAC(subdir, fname)
        # stim_sample_ids = np.where(dac != 512)[0]
        # shortcut, since we know the stim samples are between 20500 and 29500
        stim_sample_ids = (6000, 14000)
        data = read_raw_data(subdir, fname, convert2uV=True,
                            subtract_dc_offset=True, 
                            # col_slice=slice(None, None),
                            col_slice=slice(stim_sample_ids[0], stim_sample_ids[-1], None)
        )
    
        mean_ampl = []
        for j,row in enumerate(data):
            _, m_ampl = estimate_frequency_power(row.astype(float), 
                                                 sampling_rate=C.SAMPLING_RATE, 
                                                 debug=deepdebug, 
                                                 min_band=960, max_band=1040)
            if stimulated.stim[j]:
                pass
            mean_ampl.append(m_ampl)
        mean_ampl = np.array(mean_ampl)
        
        stimulated.drop("Unnamed: 0", axis=1, inplace=True)
        stimulated['sine_voltage_uV'] = mean_ampl
        # stimulated.drop(columns=['x', 'y'], inplace=True)
        stimulated = stimulated.sort_values(by=['tile', 'stim']).reset_index(drop=True)
        ratios = stimulated.groupby("tile").apply(
            lambda x: (x.sine_voltage_uV /x[x.stim].sine_voltage_uV.item()),
            # include_groups=False
        )
        print(stimulated)
        if len(ratios) != len(stimulated):
            print(ratios)
            print(stimulated)
            print("Warning: not all tiles have a ratio")
            continue
        stimulated['tile_connectivity'] = ratios.values
        stimulated.index = pd.MultiIndex.from_product([[fname.replace(".raw.h5","")],
                                                        stimulated.index], names=['config', 'el'])
        L.logger.debug(stimulated)
        aggr_imp_data.append(stimulated)

    aggr_imp_data = pd.concat(aggr_imp_data)
    save_output(subdir, aggr_imp_data, "extracted_sine_voltages.csv")

def vis_shorts(subdir, output_dir=None, output_fname=None):
    data = pd.read_csv(os.path.join(subdir, "processed", "extracted_sine_voltages.csv"))
    print(data)
    
    centers = data[(data.stim) & data.tile_connectivity.notna()]
    center_shorts = data[(data.tile_connectivity > 0.8) & 
                         np.isin(data.tile, centers.tile) &
                         (data.tile_connectivity != np.inf) &
                         (data.sine_voltage_uV > 100) # valid stimulation should have ampl > 100 uV
                         ]
    print(center_shorts)
    
    (fig, ax), recs  = draw_mea1k(bg='white', el_color='#aaaaaa')
    
    shorts = center_shorts[~center_shorts.stim]
    print(shorts)
    for i, row in shorts.iterrows():
        tile, config_name, el = row.tile, row.config, row.el
        center = center_shorts[(center_shorts.tile == tile) & 
                               (center_shorts.config == config_name) &
                               (center_shorts.stim)]#.iloc[0]
        if center.empty:
            print(f"Warning: no center found for tile {tile}, config {config_name}")
            continue
        center = center.iloc[0]
        # Draw a line between the center and the short
        ax.plot((row.x+8.75, center.x+8.75), (row.y+8.75, center.y+8.75),
                color='red', alpha=0.6, linewidth=1.5)
    
    if output_dir is not None and output_fname is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_fullfname = os.path.join(output_dir, output_fname)
        print("inkscape ", out_fullfname)
        fig.savefig(out_fullfname, dpi=300)
    plt.show()
    
def main(): 
    L = Logger()
    L.init_logger(None, None, "INFO")
    L.logger.debug("Starting shortcut analysis")
    # local_dir = C.device_paths()[1]
    nas_dir = C.device_paths()[0]
    
    # fix seed
    np.random.seed(42)
    # implant_name = "4983"
    implant_name = "MEA1K12"
    
    subdirs = [
        # f"devices/well_devices/{MEA1K08}/recordings/2025-05-09_10.14.37_invivo_imp_mode='voltage'_stimpulse='sine'_amplitude=10",
        # f"devices/headstage_devices/{implant_name}/recordings/2025-07-18_09.30.23_beforeGP_mode='voltage'_stimpulse='sine'_amplitude=10",
        # f"devices/headstage_devices/{implant_name}/recordings/2025-07-18_12.42.31_afterGP_mode='voltage'_stimpulse='sine'_amplitude=10",
        f"devices/headstage_devices/{implant_name}/recordings/2025-09-18_13.59.32_2ndBondBatch5SingelShankTight_mode='voltage'_stimpulse='sine'_amplitude=10",
    ]
    output_dir = os.path.join(nas_dir, subdirs[-1], 'processed')
    output_fname = f"shortcuts_fristimpl_{implant_name}.png"
    

    # el_config_S1D1650.raw.h5
    analyze_shorts(os.path.join(nas_dir, subdirs[0]), implant_name=implant_name, 
                    debug=False, deepdebug=False)
    vis_shorts(os.path.join(nas_dir, subdirs[0]), 
               output_dir=output_dir, 
               output_fname=output_fname)
    plt.show()
    
if __name__ == "__main__":
    main()