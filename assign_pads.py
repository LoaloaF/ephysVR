import os
import sys
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import napari
from napari.utils import DirectLabelColormap

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

from mea1k_modules.mea1k_visualizations import draw_mea1k

import json

def _affine_cache_path(nas_dir, implant_device_name, device_name):
    bonding_path = os.path.join(nas_dir, 'devices', 'implant_devices', 
                                implant_device_name, 'bonding')
    os.makedirs(bonding_path, exist_ok=True)
    return os.path.join(bonding_path, f'affine_cache_{device_name}.json')


def _save_affine_cache(nas_dir, implant_device_name, device_name, affine_matrix):
    path = _affine_cache_path(nas_dir, implant_device_name, device_name)
    with open(path, 'w') as f:
        json.dump(affine_matrix.tolist(), f, indent=2)
    print(f"  Saved affine cache: {path}")


def _load_affine_cache(nas_dir, implant_device_name, device_name):
    path = _affine_cache_path(nas_dir, implant_device_name, device_name)
    if os.path.exists(path):
        with open(path, 'r') as f:
            matrix = np.array(json.load(f))
        print(f"  Loaded affine cache: {path}")
        return matrix
    return None

# def _mea1k_el_pixel_table(MEA1K_EL_WIDTH_MICROMETER=5.45, MEA1K_EL_HEIGHT_MICROMETER=9.3):
def _visualize_pad_alignment_debug(pad_id, pad_yx_data, pad_yx_world, pad_yx_pixels, 
                                   pad_yx_center, els_under_pad, el_pixel_lookup, 
                                   mea1k_connectivity_png, iteration):
    """Debug visualization for pad alignment process."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Compute zoom region around the pad  
    margin = 50
    y_min, y_max = pad_yx_center[0] - margin, pad_yx_center[0] + margin
    x_min, x_max = pad_yx_center[1] - margin, pad_yx_center[1] + margin
    
    # Show background connectivity image in zoomed region
    region = mea1k_connectivity_png[y_min:y_max, x_min:x_max]
    ax.imshow(region, cmap='gray', alpha=0.3, extent=[x_min, x_max, y_max, y_min])
    
    # Plot pad pixels (rounded from world space)
    ax.scatter(pad_yx_pixels[:, 1], pad_yx_pixels[:, 0], c='red', s=0.5, alpha=0.7, label='Pad pixels')
    
    # Plot median center
    ax.plot(pad_yx_center[1], pad_yx_center[0], 'g+', markersize=15, markeredgewidth=0.5, label='Median center')
    
    # Plot detected electrodes
    if els_under_pad:
        el_coords = []
        for (y, x), el_id in el_pixel_lookup.items():
            if el_id in els_under_pad:
                el_coords.append([y, x])
        if el_coords:
            el_coords = np.array(el_coords)
            ax.scatter(el_coords[:, 1], el_coords[:, 0], c='orange', s=1, alpha=0.5, label=f'Electrode pixels (n_els={len(els_under_pad)})')
        
        # Draw MEA1k electrode centers and boundaries for detected electrodes
        el_centers = {}
        for el_id in els_under_pad:
            el_pixels = []
            for (y, x), eid in el_pixel_lookup.items():
                if eid == el_id:
                    el_pixels.append([y, x])
            if el_pixels:
                el_pixels = np.array(el_pixels)
                el_center = el_pixels.mean(axis=0)
                el_centers[el_id] = el_center
                # Draw electrode as square (17.5x17.5 pixels)
                el_half = 17.5 / 2
                rect = plt.Rectangle((el_center[1] - el_half, el_center[0] - el_half), 17.5, 17.5,
                                    color='cyan', fill=False, linewidth=0.2, alpha=0.6)
                ax.add_patch(rect)
                # Mark center
                ax.plot(el_center[1], el_center[0], 'c.', markersize=2)
                # Label electrode ID
                ax.text(el_center[1], el_center[0], f'{el_id}', fontsize=5, 
                       ha='center', va='center', color='cyan', alpha=0.7)
    
    # Draw pad circle with diameter
    pad_radius = PAD_R
    circle = plt.Circle((pad_yx_center[1], pad_yx_center[0]), pad_radius, 
                        color='blue', fill=False, linewidth=0.3, alpha=0.8, label=f'Pad circle (d={2*pad_radius})')
    ax.add_patch(circle)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect('equal')
    ax.set_title(f'Iteration {iteration}: Pad {pad_id} | Center: ({pad_yx_center[0]}, {pad_yx_center[1]}) | N_els: {len(els_under_pad)}')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    
    plt.tight_layout()
    plt.show()


# real size of electrodes is this, but images use 9x9, so we match with 9x9 too 
# def _mea1k_el_pixel_table(MEA1K_EL_WIDTH_MICROMETER=5.45, MEA1K_EL_HEIGHT_MICROMETER=9.3):
def _mea1k_el_pixel_table(MEA1K_EL_WIDTH_MICROMETER=9, MEA1K_EL_HEIGHT_MICROMETER=9):
    def mea1k_el_center_table_micrometer():
        el_i = 0
        all_els = {}
        for y in np.arange(17.5/2, 2100, 17.5):
            for x in np.arange(17.5/2, 3850, 17.5):
                all_els[el_i] = (y, x)
                el_i += 1
        mea1k = pd.DataFrame(all_els).T
        mea1k.columns = ['y', 'x']
        mea1k.index.name = 'el'
        return mea1k

    all_el_pixels = []
    print("Generating MEA1k electrode pixel table...")
    for el_i, (y, x) in mea1k_el_center_table_micrometer().iterrows():
        all_y = np.arange(y - MEA1K_EL_HEIGHT_MICROMETER/2,
                          y + MEA1K_EL_HEIGHT_MICROMETER/2, 1)
        all_x = np.arange(x - MEA1K_EL_WIDTH_MICROMETER/2,
                          x + MEA1K_EL_WIDTH_MICROMETER/2, 1)
        el_i_yx = np.stack(np.meshgrid(all_y, all_x)).reshape(2, -1).round().astype(np.uint16)
        multiindex = pd.MultiIndex.from_arrays(el_i_yx, names=['y', 'x'])
        all_el_pixels.append(pd.Series([el_i]*len(el_i_yx.T), index=multiindex, name='el'))
    return pd.concat(all_el_pixels)


def _align(mapping, mea1k_connectivity, alignment, mea1k_connectivity_png=None):
    # load once and convert to dict for O(1) pixel lookups: {(y, x): electrode_id}
    el_pixel_lookup = _mea1k_el_pixel_table().to_dict()

    # extract affine matrix once; apply to all pixels in one matrix multiply
    # napari affine_matrix is (ndim+1 x ndim+1): world = A[:2,:2] @ data.T + A[:2,2]
    A = alignment.affine.affine_matrix

    pad_alignment = []
    for iteration, (pad_id, row) in enumerate(mapping.iterrows()):
        # pixel coords in data space where this pad label sits
        pad_yx_data = np.stack(np.where(alignment.data == pad_id)).T  # (N, 2)

        # vectorised data→world transform (replaces per-pixel data_to_world loop)
        pad_yx_world = (A[:2, :2] @ pad_yx_data.T).T + A[:2, 2]
        pad_yx_pixels = pad_yx_world.round().astype(int)            # (N, 2)
        pad_yx_center = np.median(pad_yx_pixels, axis=0).round().astype(int)

        aligned_pad = row.copy()
        aligned_pad["pad_id"] = pad_id
        aligned_pad["mea1k_el"] = np.nan
        aligned_pad["mea1k_connectivity"] = np.nan
        aligned_pad["pad_y_aligned"] = pad_yx_center[0]
        aligned_pad["pad_x_aligned"] = pad_yx_center[1]

        # Find all electrodes that overlap with ANY pad pixel (per-pixel overlap check)
        # For each pad pixel (y, x), look up which electrode (if any) it belongs to
        els_under_pad = {el_pixel_lookup.get((y, x))
                         for y, x in map(tuple, pad_yx_pixels)}
        els_under_pad.discard(None)  # Remove None entries (pad pixels not on any electrode)
        mea1k_el_underpad = np.array(sorted(els_under_pad), dtype=int)
        print(f"pad{pad_id:<4d}> -> n el:{len(mea1k_el_underpad):<4d}", end='...', flush=True)
        
        # Debug visualization for first 3 pads
        if iteration < 3 and mea1k_connectivity_png is not None:
            _visualize_pad_alignment_debug(pad_id, pad_yx_data, pad_yx_world, pad_yx_pixels,
                                          pad_yx_center, els_under_pad, el_pixel_lookup,
                                          mea1k_connectivity_png, iteration)

        dat = [aligned_pad] * max(1, len(mea1k_el_underpad))
        aligned_pad = pd.concat(dat, axis=1).T.reset_index(drop=True)
        if len(mea1k_el_underpad) > 0:
            aligned_pad.loc[:, "mea1k_el"] = mea1k_el_underpad
            aligned_pad.loc[:, "mea1k_connectivity"] = mea1k_connectivity.reindex(mea1k_el_underpad).values
        pad_alignment.append(aligned_pad)

    return pd.concat(pad_alignment, axis=0).reset_index(drop=True)


def _add_pads_to_napari(viewer, mea1k_connectivity_png, mapping, layer_name='Pads'):
    pad_colors = {-1: (0, 0, 0, 0)}
    pads_canvas = np.ones_like(mea1k_connectivity_png, dtype=np.int16) * -1
    for pad_id, row in mapping.iterrows():
        x, y = row[['pad_x', 'pad_y']] +10 # slightly offset to fit on canvas
        cv2.circle(pads_canvas, (x, y), PAD_R, pad_id, -1)
        pad_colors[pad_id] = (row[['pad_r', 'pad_g', 'pad_b']].values.astype(float) / 255.)
    pad_colors = DirectLabelColormap(color_dict=pad_colors)
    return viewer.add_labels(pads_canvas, colormap=pad_colors, name=layer_name)


def align_pads2mea1k(electrode_device_names, IMPLANT_DEVICE_NAME,
                     connectivty_measures_path, force_realign=False):
    if isinstance(electrode_device_names, str):
        electrode_device_names = [electrode_device_names]

    nas_dir = device_paths()[0]

    # --- load all mappings -------------------------------------------------------
    device_mappings = []
    for device_name in electrode_device_names:
        mapping_fullfname = os.path.join(
            nas_dir, 'devices', "electrode_devices", device_name,
            f"device_mapping_{device_name}.csv")
        mapping = pd.read_csv(mapping_fullfname, index_col=[0])
        mapping.index = mapping.pad_id.values
        device_mappings.append((device_name, mapping))

    # --- assert pad_id and RGB uniqueness across all devices ---------------------
    all_mappings = pd.concat([m for _, m in device_mappings])

    dup_ids = all_mappings.pad_id[all_mappings.pad_id.duplicated()]
    assert len(dup_ids) == 0, \
        f"Duplicate pad_ids found across devices: {sorted(dup_ids.tolist())}"

    rgb_tuples = list(zip(all_mappings['pad_r'], all_mappings['pad_g'], all_mappings['pad_b']))
    dup_rgbs = [rgb for rgb in set(rgb_tuples) if rgb_tuples.count(rgb) > 1]
    assert len(dup_rgbs) == 0, \
        f"Duplicate pad RGB colors found across devices: {dup_rgbs}"

    # --- load connectivity image and CSV -----------------------------------------
    connectivity_fullfname = os.path.join(
        nas_dir, connectivty_measures_path, 'processed', 'extr_connectivity.png')
    if not os.path.exists(connectivity_fullfname):
        raise FileNotFoundError(f"connectivity image not found: {connectivity_fullfname}")
    mea1k_connectivity_png = cv2.imread(connectivity_fullfname)[:, :, 0]

    path = connectivity_fullfname.replace(".png", ".csv")
    mea1k_connectivity = pd.read_csv(path, index_col=[0]).sort_index()
    mea1k_connectivity.rename(columns={'connectivity': 'mea1k_connectivity'}, inplace=True)
    mea1k_connectivity.drop(["ampl", "input_ampl_mV"], inplace=True, axis=1)
    mea1k_connectivity.set_index('el', inplace=True)
    mea1k_connectivity = mea1k_connectivity[~mea1k_connectivity.index.duplicated(keep='first')]
    mea1k_connectivity = mea1k_connectivity.reindex(np.arange(26400))
    
    # --- check if ALL devices have cached affines --------------------------------
    all_cached = not force_realign and all(
        _load_affine_cache(nas_dir, IMPLANT_DEVICE_NAME, dn) is not None
        for dn in electrode_device_names
    )

    viewer = napari.Viewer()
    viewer.add_image(mea1k_connectivity_png, name='Connectivity', colormap='gray_r', opacity=.5)

    alignments = []
    for device_name, mapping in device_mappings:
        layer = _add_pads_to_napari(viewer, mea1k_connectivity_png, mapping,
                                    layer_name=device_name)

        # apply cached affine immediately if available
        cached = _load_affine_cache(nas_dir, IMPLANT_DEVICE_NAME, device_name)
        if cached is not None and not force_realign:
            layer.affine = cached   # napari accepts ndarray directly
            print(f"  [{device_name}] Restored affine from cache.")
        else:
            print(f"  [{device_name}] No affine cache found — align manually in napari.")

        alignments.append((mapping, layer))

    if not all_cached or force_realign:
        print("Opening napari for manual alignment...")
    else:
        print("All affines loaded from cache — opening napari for review/adjustment.")
    
    napari.run()
    
    # --- save affines after user closes napari ---
    for device_name, (mapping, layer) in zip(electrode_device_names, alignments):
        _save_affine_cache(nas_dir, IMPLANT_DEVICE_NAME, device_name,
                           layer.affine.affine_matrix)
        
    # --- compute pad→electrode mapping for every device --------------------------
    all_pad_alignments = []
    for mapping, alignment in alignments:
        print(f"Aligning {len(mapping)} pads to MEA1k electrodes")
        all_pad_alignments.append(_align(mapping, mea1k_connectivity, alignment, mea1k_connectivity_png))
    pad_alignment = pd.concat(all_pad_alignments, axis=0).reset_index(drop=True)

    # --- Remove ambiguous mea1k_el assignments (appearing under multiple pads) ---
    print("\n--- Resolving ambiguous mea1k_el assignments ---")
    valid_rows = pad_alignment['pad_id'].notna() & pad_alignment['mea1k_el'].notna()
    pad_per_mea1k = pad_alignment[valid_rows].groupby('mea1k_el')['pad_id'].nunique()
    ambiguous_mea1k_els = pad_per_mea1k[pad_per_mea1k > 1].index

    if len(ambiguous_mea1k_els) > 0:
        # Count how many high-connectivity electrodes are being excluded
        ambiguous_rows = pad_alignment[pad_alignment['mea1k_el'].isin(ambiguous_mea1k_els)]
        high_conn_excluded = (ambiguous_rows['mea1k_connectivity'] > CONNECTIVITY_THR).sum()
        unique_high_conn_excluded = ambiguous_rows[ambiguous_rows['mea1k_connectivity'] > CONNECTIVITY_THR]['mea1k_el'].nunique()
        
        print(f"⚠️  Ambiguous assignments detected:")
        print(f"   {len(ambiguous_mea1k_els)} mea1k_el appear under multiple pads")
        print(f"   {len(ambiguous_rows)} total rows affected")
        print(f"   {high_conn_excluded} rows with connectivity > {CONNECTIVITY_THR} excluded")
        print(f"   {unique_high_conn_excluded} unique mea1k_el with connectivity > {CONNECTIVITY_THR} excluded")
        
        # Set pad_id and el_id to NaN for ambiguous assignments
        mask = pad_alignment['mea1k_el'].isin(ambiguous_mea1k_els)
        pad_alignment.loc[mask, 'pad_id'] = np.nan
        pad_alignment.loc[mask, 'el_id'] = np.nan
        # Also clear other pad/electrode specific fields
        pad_cols_to_clear = ['pad_y', 'pad_x', 'pad_r', 'pad_g', 'pad_b', 'pad_metal', 
                             'pad_y_aligned', 'pad_x_aligned', 'el_depth', 'shank_id', 
                             'shank_side', 'el_r', 'el_g', 'el_b']
        for col in pad_cols_to_clear:
            if col in pad_alignment.columns:
                pad_alignment.loc[mask, col] = np.nan
        
        # Drop duplicate ambiguous mea1k_el rows (keep first occurrence)
        pad_alignment = pad_alignment[~(mask & pad_alignment.duplicated(subset=['mea1k_el'], keep='first'))]
    # --- rank electrodes under each pad by connectivity --------------------------
    # --- rank polyimide electrodes connected to 1+ pads by connectivity --------
    # Create two separate connectivity rankings: pad_connectivity_rank and el_connectivity_rank
    pad_alignment['pad_connectivity_rank'] = np.nan
    pad_alignment['el_connectivity_rank'] = np.nan
    
    # pad_connectivity_rank: rank mea1k electrodes within each pad
    valid_pad_rows = pad_alignment['pad_id'].notna() & pad_alignment['mea1k_connectivity'].notna()
    if valid_pad_rows.any():
        pad_ranks = (pad_alignment[valid_pad_rows]
                     .sort_values(['pad_id', 'mea1k_connectivity'], ascending=[True, False])
                     .groupby('pad_id')['mea1k_connectivity']
                     .rank(method='first', ascending=False))
        pad_alignment.loc[valid_pad_rows, 'pad_connectivity_rank'] = pad_ranks

    # el_connectivity_rank: rank mea1k electrodes within each polyimide electrode
    valid_el_rows = pad_alignment['el_id'].notna() & pad_alignment['mea1k_connectivity'].notna()
    if valid_el_rows.any():
        el_ranks = (pad_alignment[valid_el_rows]
                    .sort_values(['el_id', 'mea1k_connectivity'], ascending=[True, False])
                    .groupby('el_id')['mea1k_connectivity']
                    .rank(method='first', ascending=False))
        pad_alignment.loc[valid_el_rows, 'el_connectivity_rank'] = el_ranks

    # --- append electrodes not under any pad -------------------------------------
    under_pad_els = mea1k_connectivity.index.intersection(pad_alignment.mea1k_el)
    mea1k_connectivity = mea1k_connectivity.drop(under_pad_els).reset_index()
    mea1k_connectivity.columns = ["mea1k_el", "mea1k_connectivity", 'input_mV']
    pad_alignment = pd.concat([pad_alignment, mea1k_connectivity], axis=0)

    order = ['pad_id', 'pad_y', 'pad_x', 'pad_r', 'pad_g', 'pad_b', 'pad_metal',
             'pad_y_aligned', 'pad_x_aligned',
             'el_depth', 'shank_id', 'shank_side',
             'el_r', 'el_g', 'el_b',
             'mea1k_el', 'el_id', 'mea1k_connectivity', 'pad_connectivity_rank', 'el_connectivity_rank']
    missing_els = np.setdiff1d(np.arange(26400), pad_alignment.mea1k_el)
    pad_alignment = pd.concat(
        [pad_alignment, pd.DataFrame({"mea1k_el": missing_els})], axis=0)
    pad_alignment = (pad_alignment[order]
                     .sort_values(['pad_id', 'mea1k_el'])
                     .reset_index(drop=True))
    print('\n======\n', pad_alignment)

    bonding_path = os.path.join(
        nas_dir, 'devices', "implant_devices", IMPLANT_DEVICE_NAME, "bonding")
    os.makedirs(bonding_path, exist_ok=True)
    fullfname = os.path.join(bonding_path, f'bonding_mapping_{IMPLANT_DEVICE_NAME}.csv')
    pad_alignment.to_csv(fullfname, index=False)

def integrate_impedance(bonding_mappin_fullfname, impedance_csv_fullfname):
    bonding_mapping = pd.read_csv(bonding_mappin_fullfname)
    impedance = pd.read_csv(impedance_csv_fullfname)
    if 'impedance_Ohm' in bonding_mapping.columns:
        print(f"Impedance column already exists in {bonding_mappin_fullfname}, skipping integration.")
        return
    impedance.set_index('electrode', inplace=True)
    bonding_mapping.set_index('mea1k_el', inplace=True)
    combined = bonding_mapping.join(impedance['impedance_Ohm'], how='left')
    combined.reset_index(inplace=True)
    combined.to_csv(bonding_mappin_fullfname)

def plot_pad_alignment(IMPLANT_DEVICE_NAME, white_bg=False):
    nas_dir = device_paths()[0]
    fullfname = os.path.join(nas_dir, "devices", "implant_devices",
                             IMPLANT_DEVICE_NAME, "bonding",
                             f'bonding_mapping_{IMPLANT_DEVICE_NAME}.csv')
    pad_alignment = pd.read_csv(fullfname)
    
    print(pad_alignment, flush=True)
    # how many pads are connected
    n_pads = pad_alignment['pad_id'].nunique()
    n_connected_pads = pad_alignment[pad_alignment['mea1k_connectivity'] > CONNECTIVITY_THR]['pad_id'].nunique()
    print(f"Connected pads: {n_connected_pads}/{n_pads} ({n_connected_pads/n_pads:.1%})")
    # how many polyimide electrodes have good connectivity
    n_connected_els = pad_alignment[pad_alignment['mea1k_connectivity'] > CONNECTIVITY_THR]['el_id'].nunique()
    print(f"Connected electrodes: {n_connected_els}/{len(pad_alignment.el_id.unique())} ({n_connected_els/len(pad_alignment.el_id.unique()):.1%})")
    
    
    plt.hist(pad_alignment['mea1k_connectivity'], bins=50)
    plt.show()

    (fig, ax), els = draw_mea1k(bg='white' if white_bg else 'black')

    pad_circles = []
    seen_pad_ids = []
    for el_i, el_rec in enumerate(els):
        if el_i not in pad_alignment.mea1k_el.values:
            continue
        #     print(f"EL{el_i} not under any pad")

        el_entry = pad_alignment[pad_alignment['mea1k_el'] == el_i].iloc[0]
        mea1k_el_conn = el_entry.mea1k_connectivity.clip(0, 1)  # ensure between 0 and 1 for alpha
        
        # for all set color and alpha to baseline, see all
        el_rec.set_alpha(.15)
        if white_bg:
            el_rec.set_facecolor((0.5, 0.5, 0.5))
        else:
            el_rec.set_facecolor((1, 1, 1))
        
        # continue
        if pd.isna(el_entry.pad_id):
            # print(f"EL{el_i} under pad_id NaN")
            continue
        
        # each electrod under a pad gets colored by its pad's RGB color, regardless of connectivity
        col = pad_alignment[pad_alignment['mea1k_el'] == el_i][['el_r', 'el_g', 'el_b']].values[0]
        # col = pad_alignment[pad_alignment['mea1k_el'] == el_i][['pad_r', 'pad_g', 'pad_b']].values[0] / 255
        el_rec.set_alpha(min(1, mea1k_el_conn+.2)) # set alpha to connectivity, see connectivity pattern
        
        # color each electrode by its pad
        el_rec.set_facecolor(col)
        
        # col = pad_alignment[pad_alignment['mea1k_el'] == el_i][['el_r', 'el_g', 'el_b']].values[0] / 255
        # whiteness = np.clip(el_entry.mea1k_connectivity * 1, 0, 1)
        # el_rec.set_facecolor((whiteness, whiteness, whiteness))

        if el_entry.pad_id not in seen_pad_ids:
            pad_circles.append(plt.Circle((el_entry.pad_x_aligned, el_entry.pad_y_aligned),
                                            PAD_R, color=col,
                                            fill=False, linewidth=.5, alpha=.25))
            seen_pad_ids.append(el_entry.pad_id)
        # if (el_entry.el_connectivity_rank == 1) and (el_entry.mea1k_connectivity > CONNECTIVITY_THR) and el_entry.notna()["shank_id"]:
        if (mea1k_el_conn > CONNECTIVITY_THR) and el_entry["shank_id"]:
            # el_rec.set_alpha(1)
            # el_rec.set_facecolor(col)
            pad_circles.append(plt.Circle((el_entry.pad_x_aligned, el_entry.pad_y_aligned),
                                          PAD_R, color=col,
                                          fill=False, linewidth=.5, alpha=1))
            # if el_entry.el_connectivity_rank == 1:
            #     el_rec.set_facecolor('green')
            #     el_rec.set_alpha(1)
            #     el_rec.set_zorder(10)
                
            # # annotate el_entry with connectivity order
            # # get x y of recatngle patch
            # x,y = el_rec.get_xy()
            # ax.text(x, y, f"{int(el_entry.el_connectivity_rank)}",
            #         fontsize=8, ha='center', va='center', color='white', alpha=0.8)
    [ax.add_patch(pc) for pc in pad_circles]
    if white_bg:
        fullfname = fullfname.replace(".csv", "_inverted.csv")
    fig.savefig(fullfname.replace("csv", "png"), dpi=300, transparent=False,
                bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_pad_shorts(IMPLANT_DEVICE_NAME, white_bg=False):
    nas_dir = device_paths()[0]
    
    fullfname = os.path.join(nas_dir, "devices", "implant_devices",
                             IMPLANT_DEVICE_NAME, "bonding",
                             f"bonding_mapping_{IMPLANT_DEVICE_NAME}.csv")
    pad_alignment = pd.read_csv(fullfname)
    pads = pad_alignment.dropna(subset=['pad_id', 'pad_x_aligned', 'pad_y_aligned', 'el_id']).copy()
    pads = pads.drop_duplicates(subset=['pad_id'])
    shorted_el_ids = pads.groupby('el_id')['pad_id'].nunique().loc[lambda s: s > 1].index
    pads = pads[pads['el_id'].isin(shorted_el_ids)].copy()

    base_colors = list(plt.cm.get_cmap('tab20').colors)
    # spread colors to avoid adjacent similar shades from tab20
    color_order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    contrast_colors = [base_colors[i] for i in color_order]
    group_color = {
        el_id: contrast_colors[i % len(contrast_colors)]
        for i, el_id in enumerate(sorted(shorted_el_ids))
    }

    (fig, ax), els = draw_mea1k(bg='white' if white_bg else 'black')

    conn_lookup = (pad_alignment[['mea1k_el', 'mea1k_connectivity']]
                   .dropna(subset=['mea1k_el'])
                   .drop_duplicates(subset=['mea1k_el'])
                   .set_index('mea1k_el')['mea1k_connectivity'])
    for el_i, el_rec in enumerate(els):
        el_rec.set_alpha(.15)
        if white_bg:
            el_rec.set_facecolor((0.5, 0.5, 0.5))
        else:
            el_rec.set_facecolor((1, 1, 1))
        if el_i in conn_lookup.index and pd.notna(conn_lookup.loc[el_i]):
            el_rec.set_alpha(min(1, conn_lookup.loc[el_i] + .2))

    for el_id, grp in pads.groupby('el_id'):
        col = group_color[el_id]
        ax.scatter(grp['pad_x_aligned'], grp['pad_y_aligned'], s=20, color=col, edgecolors='none', linewidths=0.3)

        pts = grp[['pad_x_aligned', 'pad_y_aligned']].to_numpy()
        for pair_i in range(len(pts)):
            for pair_j in range(pair_i + 1, len(pts)):
                x0, y0 = pts[pair_i]
                x1, y1 = pts[pair_j]
                rad = 0.2 if (pair_i + pair_j) % 2 == 0 else -0.2
                arc = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-',
                                      connectionstyle=f"arc3,rad={rad}", linestyle=':',
                                      linewidth=2, color=col, alpha=0.85)
                ax.add_patch(arc)

        # cx, cy = grp['pad_x_aligned'].mean(), grp['pad_y_aligned'].mean()
        # ax.text(cx + 10, cy + 10, f"el_id={int(el_id)}", fontsize=7, color=col)

    fig.savefig(fullfname.replace(".csv", "_shorts.png"), dpi=300, transparent=False,
                bbox_inches='tight', pad_inches=0)
    plt.show()

CONNECTIVITY_THR = .2
PAD_R = 22

def main():
    nas_dir = device_paths()[0]
    # NewGen 2026 - combined device from two sub-devices
    ELECTRODE_DEVICE_NAMES = ['S0844pad8shank','S0844pad6shank']  # combined = S1688pad14shank
    HEADSTAGE_DEVICE_NAME = 'MEA1K23'
    date = '260625'
    batch = 6
    IMPLANT_DEVICE_NAME = f"{date}_{HEADSTAGE_DEVICE_NAME}_S1688pad14shankB{batch}"

    IMPLANT_DEVICE_NAME = "260713_MEA1K23_J1688pad2shankB1"
    rec_dir_name = f'2026-07-13_15.50_JDesignFirstSoldered2_VRef15_PT2'
    # rec_dir_name = f'2026-07-15_13.59_ReEtched14ShankReGoldPlated_VRef15_PT3'
    connectivity_rec_path = os.path.join(nas_dir, 'devices', 'implant_devices',
                                         IMPLANT_DEVICE_NAME, 'recordings', rec_dir_name)
    # tileshorts_path = os.path.join(nas_dir, 'devices', 'implant_devices',
    #                                IMPLANT_DEVICE_NAME, 'recordings', 
    #                                '2026-07-09_09.48.23_SC_JFirstTrySoldered_PT4')
    # align_pads2mea1k_from_tileshorts(ELECTRODE_DEVICE_NAMES, IMPLANT_DEVICE_NAME, tileshorts_path, force_realign=False)
    
    # align_pads2mea1k(ELECTRODE_DEVICE_NAMES, IMPLANT_DEVICE_NAME, connectivity_rec_path, force_realign=False)
    # integrate_impedance(
    #     os.path.join(nas_dir, 'devices', 'implant_devices', IMPLANT_DEVICE_NAME, "bonding", f'bonding_mapping_{IMPLANT_DEVICE_NAME}.csv'),
    #     os.path.join(nas_dir, 'devices', 'implant_devices', IMPLANT_DEVICE_NAME, "recordings", 'impedance', 'processed', "all_impedance.csv")
    # )
    plot_pad_alignment(IMPLANT_DEVICE_NAME, white_bg=True)
    # plot_pad_shorts(IMPLANT_DEVICE_NAME)



if __name__ == '__main__':
    main()