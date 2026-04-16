import os
import sys

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

# import parent dir with general modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ephys_constants as EC
from mea1k_modules.mea1k_raw_preproc import read_raw_data
from mea1k_modules.mea1k_raw_preproc import read_stim_DAC
from signal_helpers import estimate_frequency_power

from mea1k_modules.mea1k_visualizations import draw_mea1k

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
        
        # get the config information about this configuration
        stimulated = pd.read_csv(os.path.join(subdir, fname.replace(".raw.h5", ".csv")))
        
        # dac = read_stim_DAC(subdir, fname)
        # stim_sample_ids = np.where(dac != 512)[0]
        # shortcut, since we know the stim samples are between 20500 and 29500

        # stim_sample_ids = (6000, 14000)
        stim_sample_ids = (14000, 22500)
        data = read_raw_data(subdir, fname, convert2uV=True,
                            subtract_dc_offset=True, 
                            # col_slice=slice(None, None),
                            col_slice=slice(stim_sample_ids[0], stim_sample_ids[-1], None)
        )
    
        mean_ampl = []
        for j,row in enumerate(data):
            m_ampl, phase = estimate_frequency_power(row.astype(float), 
                                                 sampling_rate=EC.SAMPLING_RATE, 
                                                 debug=deepdebug, 
                                                 min_band=960, max_band=1040)
            # if stimulated.stim[j]:
            #     pass
            mean_ampl.append(m_ampl)
        mean_ampl = np.array(mean_ampl)
        
        # TODO check if stimulated center el has actually sine 
        stimulated.drop("Unnamed: 0", axis=1, inplace=True)
        stimulated['sine_voltage_uV'] = mean_ampl
        # stimulated.drop(columns=['x', 'y'], inplace=True)
        stimulated = stimulated.sort_values(by=['tile', 'stim']).reset_index(drop=True)
        ratios = stimulated.groupby("tile").apply(
            lambda x: (x.sine_voltage_uV /x[x.stim].sine_voltage_uV.item()),
            # include_groups=False
        )
        
        # single tile case is turned into df instead of two-level series
        if len(ratios.shape) == 2:
            ratios = ratios.iloc[0] 
        
        stimulated['tile_connectivity'] = ratios.values
        stimulated.index = pd.MultiIndex.from_product([[fname.replace(".raw.h5","")],
                                                        stimulated.index], names=['config', 'el'])
        L.logger.debug(stimulated)
        aggr_imp_data.append(stimulated)

    aggr_imp_data = pd.concat(aggr_imp_data)
    save_output(subdir, aggr_imp_data, "extracted_sine_voltages.csv")

def analyze_single_stim(subdir, deepdebug=False):
    L = Logger()
    fnames, _ = get_hdf5_fnames_from_dir(subdir)
    aggr_imp_data = []
    for i, fname in enumerate(fnames):
        L.logger.info(f"Config {i}/{len(fnames)}")
        
        stimulated = pd.read_csv(os.path.join(subdir, fname.replace(".raw.h5", ".csv")))
        
        stim_sample_ids = (14000, 22500)
        data = read_raw_data(subdir, fname, convert2uV=True,
                            subtract_dc_offset=True, 
                            col_slice=slice(stim_sample_ids[0], stim_sample_ids[-1], None))
    
        mean_ampl = []
        for j, row in enumerate(data):
            m_ampl, phase = estimate_frequency_power(row.astype(float), 
                                                 sampling_rate=EC.SAMPLING_RATE, 
                                                 debug=deepdebug, 
                                                 min_band=960, max_band=1040)
            mean_ampl.append(m_ampl)
        mean_ampl = np.array(mean_ampl)
        
        if "Unnamed: 0" in stimulated.columns:
            stimulated.drop("Unnamed: 0", axis=1, inplace=True)
            
        stimulated['sine_voltage_uV'] = mean_ampl
        
        stim_row = stimulated[stimulated.stim]
        if len(stim_row) == 0:
            L.logger.warning(f"No stim electrode found in config {fname}")
            continue
            
        stim_ampl = stim_row.sine_voltage_uV.values[0]
        stimulated['tile_connectivity'] = stimulated['sine_voltage_uV'] / stim_ampl
        stimulated['tile'] = 0 # Dummy tile for compatibility with visualization functions
        
        stimulated.index = pd.MultiIndex.from_product([[fname.replace(".raw.h5","")],
                                                        stimulated.index], names=['config', 'el'])
        aggr_imp_data.append(stimulated)

    aggr_imp_data = pd.concat(aggr_imp_data)
    save_output(subdir, aggr_imp_data, "extracted_sine_voltages.csv")

def connected_islands(subdir, output_dir=None, output_fname=None):
    data = pd.read_csv(os.path.join(subdir, "processed", "extracted_sine_voltages.csv"))
    print(f"Total rows: {len(data)}")
    
    centers = data[(data.stim) & data.tile_connectivity.notna()]
    center_shorts = data[(data.tile_connectivity > 0.8) & 
                         np.isin(data.tile, centers.tile) &
                         (data.tile_connectivity != np.inf) &
                         (data.sine_voltage_uV > 100) # valid stimulation should have ampl > 100 uV
                         ]
    
    # Build graph of connected electrodes
    G = nx.Graph()
    
    # Add nodes with their coordinates
    for _, row in center_shorts.iterrows():
        if row.electrode not in G:
            G.add_node(row.electrode, x=row.x, y=row.y)

    # Add edges between center and its shorts
    grouped = center_shorts.groupby(['config', 'tile'])
    for (config, tile), group in grouped:
        stims = group[group.stim]
        shorts = group[~group.stim]
        
        if len(stims) == 0:
            continue
            
        center_el = stims.iloc[0].electrode
        for _, short_row in shorts.iterrows():
            G.add_edge(center_el, short_row.electrode)

    # Find connected components (islands)
    islands = list(nx.connected_components(G))
    islands = [island for island in islands if len(island) > 1] # Only keep actual shorts
    print(f"Found {len(islands)} shorted islands.")
    
    (fig, ax), recs  = draw_mea1k(bg='white', el_color='#aaaaaa')
    
    # Cycles through a colormap for different islands
    cmap = plt.cm.get_cmap('tab20')
    
    for i, island in enumerate(islands):
        color = cmap(i % 20)
        
        # Get coordinates for all electrodes in this island
        island_x = [G.nodes[el]['x'] + 8.75 for el in island]
        island_y = [G.nodes[el]['y'] + 8.75 for el in island]
        
        # Scatter plot all electrodes in the island
        ax.scatter(island_x, island_y, color=color, s=20, zorder=5, alpha=0.9)
        
        # Create a subgraph and draw its edges to clearly show connectivity
        subgraph = G.subgraph(island)
        for edge in subgraph.edges():
            x0, y0 = G.nodes[edge[0]]['x'] + 8.75, G.nodes[edge[0]]['y'] + 8.75
            x1, y1 = G.nodes[edge[1]]['x'] + 8.75, G.nodes[edge[1]]['y'] + 8.75
            ax.plot([x0, x1], [y0, y1], color=color, alpha=0.5, linewidth=2, zorder=4)

    if output_dir is not None and output_fname is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_fullfname = os.path.join(output_dir, output_fname)
        print("Saving island plot to ", out_fullfname)
        fig.savefig(out_fullfname, dpi=300)
    plt.show()
    
    
    
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
    nas_dir = device_paths()[0]
    
    # fix seed
    np.random.seed(42)
    # implant_name = "4983"
    headstage_name = "MEA1K22"
    
    subdirs = [
        f"devices/headstage_devices/{headstage_name}/recordings/2026-04-16_12.09.11_SC_16ShankW5Bond_Tight_beforeHooking",
        f"devices/headstage_devices/{headstage_name}/recordings/2026-04-16_14.41.31_SC_SingleStim_260413_MEA1K22_S1688pad14shankB5",
        f"devices/headstage_devices/{headstage_name}/recordings/2026-04-16_16.24.06_SC_SingleStim_rec2_260413_MEA1K22_S1688pad14shankB5/",
    ]
    
    target_subdir = subdirs[0]
    output_dir = os.path.join(nas_dir, target_subdir, 'processed')
    output_fname = f"shortcuts_single_stim_{headstage_name}.png"
    
    # analyze_single_stim(os.path.join(nas_dir, target_subdir), deepdebug=False)
    
    
    # # el_config_S1D1650.raw.h5
    analyze_shorts(os.path.join(nas_dir, subdirs[0]), implant_name=headstage_name, 
                    debug=False, deepdebug=False)
    
    vis_shorts(os.path.join(nas_dir, target_subdir), 
               output_dir=output_dir, 
               output_fname=output_fname)
    connected_islands(os.path.join(nas_dir, target_subdir), 
                      output_dir=output_dir, 
                      output_fname=f"connected_islands_single_stim_{headstage_name}.png")
    plt.show()
    
if __name__ == "__main__":
    main()