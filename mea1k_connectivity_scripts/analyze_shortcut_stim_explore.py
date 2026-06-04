import os
import sys
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

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
            include_groups=False
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

def analyze_single_stim(subdir, deepdebug=False, draw_each_iteration=False,
                        draw_thr=0.3, implant_mapping=None, draw_pause_s=1.5):
    L = Logger()
    fnames, _ = get_hdf5_fnames_from_dir(subdir)
    aggr_imp_data = []
    draw_state = None
    cumulative_draw_data = []
    for i, fname in enumerate(fnames):
        L.logger.info(f"Config {i}/{len(fnames)}")
        
        stimulated = pd.read_csv(os.path.join(subdir, fname.replace(".raw.h5", ".csv")), 
                                 index_col=0)
        
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
        stimulated['sine_voltage_uV'] = mean_ampl
        
        stim_row = stimulated[stimulated.stim]
        if len(stim_row) == 0:
            L.logger.warning(f"No stim electrode found in config {fname}")
            continue
            
        stim_ampl = stim_row.sine_voltage_uV.values[0]
        stimulated['tile_connectivity'] = stimulated['sine_voltage_uV'] / stim_ampl
        stimulated['tile'] = 0 # Dummy tile for compatibility with visualization functions
        stim_el_ampl_mV = stimulated[stimulated.stim].sine_voltage_uV.values[0]/1000
        next_connected = (stimulated.sort_values('tile_connectivity', ascending=False).iloc[1:4].loc[:, 'tile_connectivity'].round(2)*100).to_list()
        L.logger.info(f"Config {i}/{len(fnames)}, sine ampl at stim el: {stim_el_ampl_mV:.0f} mV. Next ones: {next_connected}% of that.")
        if stim_el_ampl_mV < 50:
            L.logger.warning(f"Config {i}/{len(fnames)} has low stim amplitude ({stim_el_ampl_mV:.1f} mV), skipping...")
            continue

        if draw_each_iteration:
            current_for_draw = stimulated.copy()
            current_for_draw['config'] = fname.replace(".raw.h5", "")
            if 'electrode' not in current_for_draw.columns and 'el' in current_for_draw.columns:
                current_for_draw['electrode'] = current_for_draw['el']

            
            stim_els = current_for_draw[current_for_draw.stim].electrode.astype(int).tolist()

            cumulative_draw_data.append(current_for_draw)
            draw_input = current_for_draw
            draw_state = draw_mea1k_shorts(
                draw_input,
                thr=draw_thr,
                implant_mapping=implant_mapping,
                draw_state=draw_state,
                clear_axes=True,
                title=f"{fname} ({i+1}/{len(fnames)})",
                current_config=fname.replace(".raw.h5", ""),
                stim_electrodes=stim_els,
                show=True,
                annotate_mea1k_els=True,
            )
        
        
        stimulated.index = pd.MultiIndex.from_product([[fname.replace(".raw.h5","")],
                                                        stimulated.index], names=['config', 'el'])
        aggr_imp_data.append(stimulated)

    aggr_imp_data = pd.concat(aggr_imp_data)
    save_output(subdir, aggr_imp_data, "extracted_sine_voltages.csv")


def draw_mea1k_shorts(data, thr=0.3, implant_mapping=None, draw_state=None,
                      clear_axes=False, title=None, current_config=None, cmap_scaler=1,
                      stim_electrodes=None, show=False, annotate_mea1k_els=False):
    if draw_state is None or clear_axes:
        if draw_state is not None and clear_axes:
            plt.close(draw_state[0])
        (fig, ax), els = draw_mea1k()
    else:
        fig, ax, els = draw_state

    # for el_rec in els:
    #     el_rec.set_linewidth(0.5)

    draw_data = data.copy()
    if 'electrode' not in draw_data.columns and 'el' in draw_data.columns:
        draw_data['electrode'] = draw_data['el']
    if 'config' not in draw_data.columns:
        draw_data['config'] = 'single'
    if 'tile' not in draw_data.columns:
        draw_data['tile'] = 0

    if current_config is not None and 'config' in draw_data.columns:
        draw_data = draw_data[draw_data['config'] == current_config].copy()


    if {'x', 'y', 'stim', 'tile_connectivity'}.issubset(draw_data.columns):
        line_data = draw_data[(draw_data['tile_connectivity'] > thr) & (draw_data['tile_connectivity'] != np.inf)]
        for (_, _), group in line_data.groupby(['config', 'tile']):
            centers = group[group['stim']]
            if centers.empty:
                continue
            center = centers.iloc[0]
            for _, row in group[~group['stim']].iterrows():
                ax.plot(
                    (row.x + 8.75, center.x + 8.75),
                    (row.y + 8.75, center.y + 8.75),
                    color='red',
                    alpha=float(np.clip(row.tile_connectivity*cmap_scaler, 0, 1)),
                    linewidth=1.2,
                    zorder=3,
                )
                pass

    if implant_mapping is None:
        if title is not None:
            ax.set_title(title)
        if show:
            plt.show()
        return fig, ax, els


    mea1k_el_col = 'mea1k_el'
    shorted_el_ids = set(
        implant_mapping.groupby('el_id')['pad_id']
        .nunique()
        .loc[lambda s: s > 1]
        .index
    )

    pad_circles = []
    seen_pad_ids = []
    for el_i, el_rec in enumerate(els):
        if el_i not in implant_mapping[mea1k_el_col].values:
            continue

        el_entry = implant_mapping[implant_mapping[mea1k_el_col] == el_i].iloc[0]
        mea1k_el_conn = np.clip(el_entry.mea1k_connectivity, 0, 1)

        el_rec.set_alpha(.08)
        el_rec.set_facecolor((1, 1, 1))

        if pd.isna(el_entry.pad_id):
            continue

        col = implant_mapping[implant_mapping[mea1k_el_col] == el_i][['el_r', 'el_g', 'el_b']].values[0]
        el_rec.set_alpha(min(0.45, 0.1 + 0.35 * mea1k_el_conn))
        # el_rec.set_facecolor(col)

        if (el_entry.el_id in shorted_el_ids) and (el_entry.pad_id not in seen_pad_ids):
            pad_circles.append(plt.Circle((el_entry.pad_x_aligned, el_entry.pad_y_aligned),
                                          PAD_R, color=col,
                                          fill=False, linewidth=.3, alpha=.15))
            seen_pad_ids.append(el_entry.pad_id)

        if (el_entry.el_id in shorted_el_ids) and (mea1k_el_conn > CONNECTIVITY_THR) and ('shank_id' in el_entry.index) and el_entry['shank_id']:
            pad_circles.append(plt.Circle((el_entry.pad_x_aligned, el_entry.pad_y_aligned),
                                          PAD_R, color=col,
                                          fill=False, linewidth=.3, alpha=.35))
            
        
        if el_i in stim_electrodes:
            ax.scatter([el_rec.get_x() + 4.75], [el_rec.get_y() + 4.75], 
                       s=18, marker='o', facecolor='none', edgecolors='red', zorder=4, linewidths=0.6)
            
        if el_i in draw_data.electrode.values:
            ax.scatter([el_rec.get_x() + 4.75], [el_rec.get_y() + 4.75], 
                       s=8, marker='o', facecolor='none', edgecolors='gray', zorder=5)
            # add annotation
            if annotate_mea1k_els:
                ax.text(el_rec.get_x() + 10, el_rec.get_y() + 10, f"el{el_i}", fontsize=8, color='white',
                        ha='center', va='center', zorder=5)

    [ax.add_patch(pc) for pc in pad_circles]

    # Draw white dotted arcs between pads that share the same el_id (shorted groups)
    shorted_groups = implant_mapping[implant_mapping['el_id'].isin(shorted_el_ids)].drop_duplicates(subset=['el_id', 'pad_id'])
    for _, grp in shorted_groups.groupby('el_id'):
        pad_points = grp[['pad_x_aligned', 'pad_y_aligned']].to_numpy()
        for pair_i, ((x0, y0), (x1, y1)) in enumerate(combinations(pad_points, 2)):
            rad = 0.2 if pair_i % 2 == 0 else -0.2
            arc = FancyArrowPatch(
                (x0, y0), (x1, y1),
                arrowstyle='-',
                connectionstyle=f"arc3,rad={rad}",
                linestyle=':',
                linewidth=0.7,
                color='white',
                alpha=0.8,
                zorder=7,
            )
            ax.add_patch(arc)

    if show:
        plt.show()

    return fig, ax, els

def connected_islands(subdir, thr, implant_mapping=None, output_dir=None, output_fname=None,
                      symmetric_only=False):
    data = pd.read_csv(os.path.join(subdir, "processed", "extracted_sine_voltages.csv"))
    print(f"Total rows: {len(data)}")
    
    fig, ax = plt.subplots()
    def draw_hist(d):
        ax.hist(d['tile_connectivity'], histtype='step', bins=100, )
    data.groupby('config').apply(draw_hist)
    plt.show()
    
    centers = data[(data.stim) & data.tile_connectivity.notna()]
    center_shorts = data[(data.tile_connectivity > thr) & 
                         np.isin(data.tile, centers.tile) &
                         (data.tile_connectivity != np.inf) &
                         (data.sine_voltage_uV > 100) # valid stimulation should have ampl > 100 uV
                         ]

    # Build graph of connected electrodes
    G = nx.Graph()

    electrode_coords = (
        center_shorts[['electrode', 'x', 'y']]
        .drop_duplicates(subset=['electrode'])
        .set_index('electrode')[['x', 'y']]
        .to_dict('index')
    )

    # Build directed connections from center -> short
    directed_edges = set()

    # Add edges between center and its shorts
    grouped = center_shorts.groupby(['config', 'tile'])
    for (config, tile), group in grouped:
        stims = group[group.stim]
        shorts = group[~group.stim]
        
        if len(stims) == 0:
            continue
            
        center_el = stims.iloc[0].electrode
        for _, short_row in shorts.iterrows():
            directed_edges.add((center_el, short_row.electrode))

    if symmetric_only:
        undirected_edges = {
            tuple(sorted((src, dst)))
            for src, dst in directed_edges
            if (dst, src) in directed_edges
        }
        removed_count = len(directed_edges) - (2 * len(undirected_edges))
        print(f"Symmetric filter enabled. Removed {removed_count} one-way connections.")
    else:
        undirected_edges = {tuple(sorted((src, dst))) for src, dst in directed_edges}

    for node, coords in electrode_coords.items():
        G.add_node(node, x=coords['x'], y=coords['y'])

    G.add_edges_from(undirected_edges)

    # Find connected components (islands)
    islands = list(nx.connected_components(G))
    islands = [island for island in islands if len(island) > 1] # Only keep actual shorts
    print(f"Found {len(islands)} shorted islands.")
    
    (fig, ax), recs  = draw_mea1k()
    cmap = plt.cm.get_cmap('tab20')
    
    for i, island in enumerate(islands):
        color = cmap(i % 20)
        base_rgb = color[:3]

        island_x = [G.nodes[el]['x'] + 8.75 for el in island]
        island_y = [G.nodes[el]['y'] + 8.75 for el in island]
        island_alphas = [
            float(np.clip(center_shorts[center_shorts.electrode == el].tile_connectivity.max(), 0.0, 1.0))
            for el in island
        ]
        island_colors = [(*base_rgb, alpha) for alpha in island_alphas]

        if (implant_mapping is None):
            ax.scatter(
                island_x,
                island_y,
                color=[(*base_rgb, 0.95)] * len(island),
                s=28,
                edgecolors='white',
                linewidths=0.3,
                zorder=5,
            )
        else:
            ax.scatter(island_x, island_y, color=island_colors, s=20, zorder=5)

        subgraph = G.subgraph(island)
        for edge in subgraph.edges():
            x0, y0 = G.nodes[edge[0]]['x'] + 8.75, G.nodes[edge[0]]['y'] + 8.75
            x1, y1 = G.nodes[edge[1]]['x'] + 8.75, G.nodes[edge[1]]['y'] + 8.75
            if (implant_mapping is None):
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=color,
                    alpha=0.7,
                    linewidth=1.2,
                    zorder=4,
                )

    if implant_mapping is not None:
        # Reuse shared renderer for pad-short overlays (keeps one source of truth)
        draw_mea1k_shorts(
            data,
            thr=thr,
            implant_mapping=implant_mapping,
            draw_state=(fig, ax, recs),
            clear_axes=False,
            title=None,
            current_config=None,
            stim_electrodes=center_shorts[center_shorts.stim].loc[:,'electrode'].to_list(),
            show=False,
        )
    

    if output_dir is not None and output_fname is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_fullfname = os.path.join(output_dir, output_fname)
        print("Saving island plot to ", out_fullfname)
        fig.savefig(out_fullfname, dpi=300)
    plt.show()


    pad_map = implant_mapping[['mea1k_el', 'pad_id']].dropna().copy()
    pad_map['mea1k_el'] = pad_map['mea1k_el'].astype(int)
    pad_map['pad_id'] = pad_map['pad_id'].astype(int)

    all_pad_ids = sorted(pad_map['pad_id'].unique())
    el_to_pads = pad_map.groupby('mea1k_el')['pad_id'].apply(lambda s: sorted(set(s.tolist())))

    pad_short_map = {pad_id: set() for pad_id in all_pad_ids}
    for island in islands:
        island_pads = set()
        for el in island:
            if int(el) in el_to_pads.index:
                island_pads.update(el_to_pads.loc[int(el)])

        if len(island_pads) < 2:
            continue

        for pad_id in island_pads:
            pad_short_map[pad_id].update(island_pads.difference({pad_id}))

    shorted_pads = pd.DataFrame(index=pd.Index(all_pad_ids, name='pad_id'))
    shorted_pads['shorted_with_pads'] = [sorted(pad_short_map[pad_id]) for pad_id in shorted_pads.index]
    shorted_pads['is_pad_shorted'] = shorted_pads['shorted_with_pads'].apply(lambda x: len(x) > 0)
    print(shorted_pads)
    return shorted_pads

def evaluate_shorts(stim_data, shorted_pads, implant_mapping, stim_electrodes=None,
                    current_config=None, thr=0.3, show=True):
    
    
    
    fig, ax, recs = draw_mea1k_shorts(stim_data, thr=thr, implant_mapping=implant_mapping,cmap_scaler=.2,
                                      stim_electrodes=stim_data[stim_data.stim].electrode.astype(int).tolist(),
                                        show=False)
    
    # emphasize shorted pads with red circles
    for pad_id in shorted_pads[shorted_pads.is_pad_shorted].index:
        pad_info = implant_mapping[implant_mapping['pad_id'] == pad_id].iloc[0]
        ax.add_patch(plt.Circle((pad_info['pad_x_aligned'], pad_info['pad_y_aligned']),
                                PAD_R, color='red', fill=False, linewidth=0.8, alpha=0.9, zorder=6))
    
    plt.show()
  
# def evaluate_shorts(stim_data, shorted_pads, implant_mapping, stim_electrodes=None,
#                     current_config=None, thr=0.3, show=True):
#     # --- Build designed shorts lookup ---
#     designed_shorts = implant_mapping.groupby('el_id').agg({
#         'pad_id': lambda s: sorted(set(s.dropna().tolist())),
#         'pad_x_aligned': lambda s: s.dropna().tolist(),
#         'pad_y_aligned': lambda s: s.dropna().tolist(),
#     })
#     designed_shorts['is_shorted'] = designed_shorts['pad_id'].apply(lambda x: len(x) > 1)

#     # Set of all designed pad pairs (frozensets for undirected comparison)
#     designed_pairs = set()
#     for _, row in designed_shorts[designed_shorts['is_shorted']].iterrows():
#         for p0, p1 in combinations(row['pad_id'], 2):
#             designed_pairs.add(frozenset((p0, p1)))

#     # Set of detected shorted pad pairs from shorted_pads
#     # Expecting shorted_pads to have columns like 'pad_id_0', 'pad_id_1' or similar
#     detected_pairs = set()
#     if 'pad_id_0' in shorted_pads.columns and 'pad_id_1' in shorted_pads.columns:
#         for _, row in shorted_pads.iterrows():
#             detected_pairs.add(frozenset((row['pad_id_0'], row['pad_id_1'])))
#     elif 'pad_id' in shorted_pads.columns and 'pad_id_b' in shorted_pads.columns:
#         for _, row in shorted_pads.iterrows():
#             detected_pairs.add(frozenset((row['pad_id'], row['pad_id_b'])))

#     # Recorded pad ids (pads that were actually in the stim_data recording)
#     recorded_pads = set()
#     if 'pad_id' in stim_data.columns:
#         recorded_pads = set(stim_data['pad_id'].dropna().unique())
#     elif 'electrode' in stim_data.columns:
#         # Map electrodes back to pad_ids
#         el_to_pad = implant_mapping.dropna(subset=['pad_id']).set_index('mea1k_el')['pad_id']
#         recorded_pads = set(el_to_pad.reindex(stim_data['electrode'].unique()).dropna().unique())

#     # --- Build pad coordinate lookup ---
#     pad_coords = {}
#     for pad_id in implant_mapping['pad_id'].dropna().unique():
#         pad_info = implant_mapping[implant_mapping['pad_id'] == pad_id].iloc[0]
#         pad_coords[pad_id] = (pad_info['pad_x_aligned'], pad_info['pad_y_aligned'])

#     # --- Draw base MEA1K with mapping ---
#     (fig, ax), els = draw_mea1k(mapping=implant_mapping, cmap_scaler=0.5, fast=True)

#     # --- Draw recording config: filter & draw tile_connectivity lines (from draw_mea1k_shorts) ---
#     draw_data = stim_data.copy()
#     if 'electrode' not in draw_data.columns and 'el' in draw_data.columns:
#         draw_data['electrode'] = draw_data['el']
#     if 'config' not in draw_data.columns:
#         draw_data['config'] = 'single'
#     if 'tile' not in draw_data.columns:
#         draw_data['tile'] = 0

#     if current_config is not None and 'config' in draw_data.columns:
#         draw_data = draw_data[draw_data['config'] == current_config].copy()

#     if {'x', 'y', 'stim', 'tile_connectivity'}.issubset(draw_data.columns):
#         line_data = draw_data[
#             (draw_data['tile_connectivity'] > thr) & (draw_data['tile_connectivity'] != np.inf)
#         ]
#         for (_, _), group in line_data.groupby(['config', 'tile']):
#             centers = group[group['stim']]
#             if centers.empty:
#                 continue
#             center = centers.iloc[0]
#             for _, row in group[~group['stim']].iterrows():
#                 ax.plot(
#                     (row.x + 8.75, center.x + 8.75),
#                     (row.y + 8.75, center.y + 8.75),
#                     color='red', alpha=float(np.clip(row.tile_connectivity, 0, 1)),
#                     linewidth=1.2, zorder=3,
#                 )

#     # --- Draw red rings on stim electrodes ---
#     if stim_electrodes is not None:
#         for el_i, el_rec in enumerate(els):
#             if el_i in stim_electrodes:
#                 ax.scatter(
#                     [el_rec.get_x() + 4.75], [el_rec.get_y() + 4.75],
#                     s=18, marker='o', facecolor='none', edgecolors='red',
#                     zorder=4, linewidths=0.6,
#                 )

#     # --- Draw pad circles ---
#     for pad_id, (px, py) in pad_coords.items():
#         is_recorded = pad_id in recorded_pads
#         fc = 'lightgray' if is_recorded else 'dimgray'
#         ec = 'black'
#         ax.scatter(px, py, s=65, color=fc, edgecolors=ec, linewidths=0.5, zorder=6, alpha=.4)

#     # --- Draw designed short arcs ---
#     for _, row in designed_shorts[designed_shorts['is_shorted']].iterrows():
#         pad_ids = row['pad_id']
#         for pair_i, (p0, p1) in enumerate(combinations(pad_ids, 2)):
#             if p0 not in pad_coords or p1 not in pad_coords:
#                 continue
#             x0, y0 = pad_coords[p0]
#             x1, y1 = pad_coords[p1]

#             pair_key = frozenset((p0, p1))
#             both_recorded = (p0 in recorded_pads) and (p1 in recorded_pads)

#             if both_recorded and pair_key in detected_pairs:
#                 color, alpha = 'limegreen', 0.9   # designed & confirmed
#             elif both_recorded and pair_key not in detected_pairs:
#                 color, alpha = 'orange', 0.8       # designed but NOT detected (false neg)
#             else:
#                 color, alpha = 'white', 0.5        # not both recorded

#             rad = 0.25 if pair_i % 2 == 0 else -0.25
#             arc = FancyArrowPatch(
#                 (x0, y0), (x1, y1),
#                 arrowstyle='-', connectionstyle=f"arc3,rad={rad}",
#                 linestyle='-', linewidth=1.2,
#                 color=color, alpha=alpha, zorder=7,
#             )
#             ax.add_patch(arc)

#     # --- Draw unintended (detected but not designed) short arcs in red ---
#     unintended_pairs = detected_pairs - designed_pairs
#     for pair_i, pair in enumerate(unintended_pairs):
#         p0, p1 = tuple(pair)
#         if p0 not in pad_coords or p1 not in pad_coords:
#             continue
#         x0, y0 = pad_coords[p0]
#         x1, y1 = pad_coords[p1]

#         rad = 0.3 if pair_i % 2 == 0 else -0.3
#         arc = FancyArrowPatch(
#             (x0, y0), (x1, y1),
#             arrowstyle='-', connectionstyle=f"arc3,rad={rad}",
#             linestyle='--', linewidth=1.5,
#             color='red', alpha=0.9, zorder=8,
#         )
#         ax.add_patch(arc)

#     if show:
#         plt.show()

#     return fig, ax, els


  
    # draw_mea1k_shorts(
    #         pd.DataFrame(),
    #         thr=0,
    #         implant_mapping=implant_mapping,
    #         # draw_state=(fig, ax, recs),
    #         clear_axes=False,
    #         title=None,
    #         current_config=None,
    #         stim_electrodes=[],
    #         show=False,
    #     )
    
    

def vis_shorts(subdir, thr=0.3, implant_mapping=None, output_dir=None, output_fname=None):
    data = pd.read_csv(os.path.join(subdir, "processed", "extracted_sine_voltages.csv"))
    print(data)
    # fig, ax = plt.subplots()
    # # ax.set_yscale('log')
    # def draw_hist(d):
    #     ax.hist(d['sine_voltage_uV'], histtype='step', bins=100, )
    # data.groupby('config').apply(draw_hist)
    # plt.show()
    
    fig, ax, recs = draw_mea1k_shorts(data, thr=thr, implant_mapping=implant_mapping,
                                      stim_electrodes=data[data.stim].electrode.astype(int).tolist(),
                                        show=True)
    
    plt.show()
    if output_dir is not None and output_fname is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_fullfname = os.path.join(output_dir, output_fname)
        print("inkscape ", out_fullfname)
        fig.savefig(out_fullfname, dpi=300)
    
CONNECTIVITY_THR = .5
PAD_R = 20

def main(): 
    L = Logger()
    L.init_logger(None, None, "INFO")
    L.logger.debug("Starting shortcut analysis")
    nas_dir = device_paths()[0]
    
    # fix seed
    np.random.seed(42)
    # implant_name = "4983"
    headstage_name = "MEA1K24"
    
    subdirs = [
        # f"devices/headstage_devices/{headstage_name}/recordings/2026-04-13_15.58.04_SC_16ShankW5Bond_Tight",
        # f"devices/headstage_devices/{headstage_name}/recordings/2026-04-16_12.09.11_SC_16ShankW5Bond_Tight_beforeHooking",
        # f"devices/headstage_devices/{headstage_name}/recordings/2026-04-16_14.41.31_SC_SingleStim_260413_MEA1K22_S1688pad14shankB5",
        # f"devices/headstage_devices/{headstage_name}/recordings/2026-04-16_16.24.06_SC_SingleStim_rec2_260413_MEA1K22_S1688pad14shankB5/",
        # f"devices/headstage_devices/{headstage_name}/recordings/2026-04-22_09.22.37_SC_SingleStim_newConfigs1_260413_MEA1K22_S1688pad14shankB5/",
        # f"devices/headstage_devices/{headstage_name}/recordings/2026-04-22_10.01.26_SC_SingleStim_newConfigs2_260413_MEA1K22_S1688pad14shankB5/",
        # f"devices/headstage_devices/{headstage_name}/recordings/2026-04-23_22.23.08_SC_16ShankW3Bond3_first",
        f"devices/headstage_devices/{headstage_name}/recordings/2026-04-28_15.54.37_SC_3x3Corners",
        f"devices/headstage_devices/{headstage_name}/recordings/2026-04-28_16.02.43_SC_3x3Corners",
    ]
    
    target_subdir = subdirs[-1]
    output_dir = os.path.join(nas_dir, target_subdir, 'processed')
    output_fname = f"shortcuts_single_stim_{headstage_name}.png"
    
    implant_name = "260413_MEA1K22_S1688pad14shankB5"
    fullfname = os.path.join(nas_dir, "devices", "implant_devices", implant_name, "bonding", f"bonding_mapping_{implant_name}.csv")
    implant_mapping = pd.read_csv(fullfname)
    
    # analyze_single_stim(os.path.join(nas_dir, target_subdir), draw_each_iteration=True, 
    #                     implant_mapping=implant_mapping,)
    
    # analyze_shorts(os.path.join(nas_dir, target_subdir), implant_name=headstage_name, 
    #                 debug=False, deepdebug=False)
    
    # vis_shorts(os.path.join(nas_dir, target_subdir),
    #            thr=CONNECTIVITY_THR,
    #            implant_mapping=implant_mapping,
    #            output_dir=output_dir,
    #            output_fname=output_fname)
    only_symmetric_connections = False
    shorted_pads = connected_islands(os.path.join(nas_dir, target_subdir), thr=CONNECTIVITY_THR,
                                    # implant_mapping=implant_mapping,
                                    output_dir=output_dir, 
                                    output_fname=f"connected_islands_single_stim_{headstage_name}.png",
                                    symmetric_only=only_symmetric_connections)
    # save_output(os.path.join(nas_dir, target_subdir), shorted_pads, "extracted_pad_shorts.csv")
    
    stim_data_1 = pd.read_csv(os.path.join(nas_dir, subdirs[-1], "processed", "extracted_sine_voltages.csv"))
    stim_data_2 = pd.read_csv(os.path.join(nas_dir, subdirs[-2], "processed", "extracted_sine_voltages.csv"))
    stim_data = pd.concat([stim_data_1, stim_data_2])
    shorted_pads_1 = pd.read_csv(os.path.join(nas_dir, subdirs[-1], "processed", "extracted_pad_shorts.csv"), index_col=0)
    shorted_pads_2 = pd.read_csv(os.path.join(nas_dir, subdirs[-2], "processed", "extracted_pad_shorts.csv"), index_col=0)
    shorted_pads = pd.concat([shorted_pads_1, shorted_pads_2])
    evaluate_shorts(stim_data, shorted_pads, implant_mapping)
    
    
    plt.show()
    
if __name__ == "__main__":
    main()