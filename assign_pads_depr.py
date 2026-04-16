import os
import sys
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import napari
from napari.utils import DirectLabelColormap

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

from mea1k_modules.mea1k_visualizations import draw_mea1k


def _mea1k_el_pixel_table(MEA1K_EL_WIDTH_MICROMETER=5.45, MEA1K_EL_HEIGHT_MICROMETER=9.3):
    code_dir = device_paths()[2]
    if code_dir is None:
        return None
    cached_fullfname = os.path.join(code_dir, 'ephysVR', 'assets', "mea1k_el_pixel_table.pkl")
    if os.path.exists(cached_fullfname):
        return pd.read_pickle(cached_fullfname)

    def mea1k_el_center_table_micrometer():
        el_i = 0
        all_els = {}
        for y in np.arange(17.5/4, 2100, 17.5):
            for x in np.arange(17.5/4, 3850, 17.5):
                all_els[el_i] = (y, x)
                el_i += 1
        mea1k = pd.DataFrame(all_els).T
        mea1k.columns = ['y', 'x']
        mea1k.index.name = 'el'
        return mea1k

    all_el_pixels = []
    for el_i, (y, x) in mea1k_el_center_table_micrometer().iterrows():
        all_y = np.arange(y - MEA1K_EL_HEIGHT_MICROMETER/2,
                          y + MEA1K_EL_HEIGHT_MICROMETER/2, 1)
        all_x = np.arange(x - MEA1K_EL_WIDTH_MICROMETER/2,
                          x + MEA1K_EL_WIDTH_MICROMETER/2, 1)
        el_i_yx = np.stack(np.meshgrid(all_y, all_x)).reshape(2, -1).round().astype(np.uint16)
        multiindex = pd.MultiIndex.from_arrays(el_i_yx, names=['y', 'x'])
        all_el_pixels.append(pd.Series([el_i]*len(el_i_yx.T), index=multiindex, name='el'))
    pd.to_pickle(pd.concat(all_el_pixels), cached_fullfname)
    return pd.concat(all_el_pixels)


def _align(mapping, mea1k_connectivity, alignment):
    # load once and convert to dict for O(1) pixel lookups: {(y, x): electrode_id}
    el_pixel_lookup = _mea1k_el_pixel_table().to_dict()

    # extract affine matrix once; apply to all pixels in one matrix multiply
    # napari affine_matrix is (ndim+1 x ndim+1): world = A[:2,:2] @ data.T + A[:2,2]
    A = alignment.affine.affine_matrix

    pad_alignment = []
    for pad_id, row in mapping.iterrows():
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

        # O(1) dict lookup per pixel instead of isin over the full MultiIndex
        els_under_pad = {el_pixel_lookup.get((y, x))
                         for y, x in map(tuple, pad_yx_pixels)}
        els_under_pad.discard(None)
        mea1k_el_underpad = np.array(sorted(els_under_pad), dtype=int)
        print(f"pad{pad_id} -> n el:{len(mea1k_el_underpad)}", end='...', flush=True)

        dat = [aligned_pad] * max(1, len(mea1k_el_underpad))
        aligned_pad = pd.concat(dat, axis=1).T.reset_index(drop=True)
        if len(mea1k_el_underpad) > 0:
            aligned_pad.loc[:, "mea1k_el"] = mea1k_el_underpad
            aligned_pad.loc[:, "mea1k_connectivity"] = mea1k_connectivity.reindex(mea1k_el_underpad).values
        pad_alignment.append(aligned_pad)

    return pd.concat(pad_alignment, axis=0).reset_index(drop=True)


def _add_pads_to_napari(viewer, mea1k_connectivity_png, mapping, scale_up_pads, layer_name='Pads'):
    pad_colors = {-1: (0, 0, 0, 0)}
    pads_canvas = np.ones_like(mea1k_connectivity_png, dtype=np.int16) * -1
    for pad_id, row in mapping.iterrows():
        x, y = row[['pad_x', 'pad_y']]
        pad_r = 18
        cv2.circle(pads_canvas, (x, y), pad_r, pad_id, -1)
        pad_colors[pad_id] = (row[['pad_r', 'pad_g', 'pad_b']].values.astype(float) / 255.)
    pad_colors = DirectLabelColormap(color_dict=pad_colors)
    return viewer.add_labels(pads_canvas, colormap=pad_colors, name=layer_name)


def align_pads2mea1k(electrode_device_names, IMPLANT_DEVICE_NAME,
                     connectivty_measures_path, scale_up_pads=2.7):
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
        device_mappings.append((device_name, mapping))

    # --- assert pad_id and RGB uniqueness across all devices ---------------------
    all_mappings = pd.concat([m for _, m in device_mappings])
    print(all_mappings)

    dup_ids = all_mappings.pad_id[all_mappings.pad_id.duplicated()]
    assert len(dup_ids) == 0, \
        f"Duplicate pad_ids found across devices: {sorted(dup_ids.tolist())}"
    all_mappings.index = all_mappings.pad_id.values

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
    mea1k_connectivity.drop("ampl", inplace=True, axis=1)
    mea1k_connectivity.set_index('el', inplace=True)
    mea1k_connectivity = mea1k_connectivity[~mea1k_connectivity.index.duplicated(keep='first')]
    mea1k_connectivity = mea1k_connectivity.reindex(np.arange(26400))
    
    plt.hist(mea1k_connectivity.mea1k_connectivity, bins=50)
    plt.axvline(CONNECTIVITY_THR, color='red', linestyle='dashed', 
                label=f"Connectivity Threshold ({CONNECTIVITY_THR})")
    plt.xlabel("Connectivity")
    plt.ylabel("Count")
    plt.title("Distribution of Connectivity Measures")
    plt.legend()
    plt.yscale('log')
    plt.show()

    # --- build napari viewer, one layer per device -------------------------------
    viewer = napari.Viewer()
    viewer.add_image(mea1k_connectivity_png, name='Connectivity', colormap='gray_r', opacity=.5)

    alignments = []
    for device_name, mapping in device_mappings:
        alignment = _add_pads_to_napari(viewer, mea1k_connectivity_png, mapping,
                                        scale_up_pads=scale_up_pads, layer_name=device_name)
        alignments.append((mapping, alignment))

    napari.run()

    # --- compute pad→electrode mapping for every device --------------------------
    all_pad_alignments = []
    for mapping, alignment in alignments:
        all_pad_alignments.append(_align(mapping, mea1k_connectivity, alignment))
    pad_alignment = pd.concat(all_pad_alignments, axis=0).reset_index(drop=True)

    # --- rank electrodes under each pad by connectivity --------------------------
    ranks = (pad_alignment[['pad_id', 'mea1k_connectivity']]
             .sort_values(['pad_id', 'mea1k_connectivity'], ascending=[True, False])
             .groupby('pad_id').rank(ascending=False)
             .sort_index())
    pad_alignment['connectivity_order'] = ranks['mea1k_connectivity']

    # --- append electrodes not under any pad -------------------------------------
    under_pad_els = mea1k_connectivity.index.intersection(pad_alignment.mea1k_el)
    mea1k_connectivity = mea1k_connectivity.drop(under_pad_els).reset_index()
    mea1k_connectivity.columns = ["mea1k_el", "mea1k_connectivity", 'input_mV']
    pad_alignment = pd.concat([pad_alignment, mea1k_connectivity], axis=0)

    order = ['pad_id', 'pad_y', 'pad_x', 'pad_r', 'pad_g', 'pad_b', 'pad_metal',
             'pad_y_aligned', 'pad_x_aligned',
             'el_depth', 'shank_id', 'shank_side',
             'el_r', 'el_g', 'el_b',
             'mea1k_el', 'mea1k_connectivity', 'connectivity_order']
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


def plot_pad_alignment(IMPLANT_DEVICE_NAME):
    nas_dir = device_paths()[0]
    fullfname = os.path.join(nas_dir, "devices", "implant_devices",
                             IMPLANT_DEVICE_NAME, "bonding",
                             f'bonding_mapping_{IMPLANT_DEVICE_NAME}.csv')
    pad_alignment = pd.read_csv(fullfname)

    (fig, ax), els = draw_mea1k()

    pad_circles = []
    for el_i, el_rec in enumerate(els):
        if el_i not in pad_alignment.mea1k_el.values:
            continue
        el_entry = pad_alignment[pad_alignment['mea1k_el'] == el_i].iloc[0]
        el_rec.set_alpha(min(1, el_entry.mea1k_connectivity/3))
        if pd.isna(el_entry.pad_id):
            continue

        col = pad_alignment[pad_alignment['mea1k_el'] == el_i][['pad_r', 'pad_g', 'pad_b']].values[0] / 255
        whiteness = np.clip(el_entry.mea1k_connectivity * 1, 0, 1)
        el_rec.set_facecolor((whiteness, whiteness, whiteness))

        if (el_entry.connectivity_order == 1) and (el_entry.mea1k_connectivity > CONNECTIVITY_THR) and el_entry.notna()["shank_id"]:
            el_rec.set_alpha(1)
            el_rec.set_facecolor(col)
            pad_circles.append(plt.Circle((el_entry.pad_x_aligned, el_entry.pad_y_aligned),
                                          PAD_R*1.7, color=col,
                                          fill=False, linewidth=.8, alpha=.5))
    [ax.add_patch(pc) for pc in pad_circles]
    fig.savefig(fullfname.replace("csv", "png"), dpi=300, transparent=False,
                bbox_inches='tight', pad_inches=0)
    plt.show()


CONNECTIVITY_THR = .7
PAD_R = 18

def main():
    nas_dir = device_paths()[0]
    # NewGen 2026 - combined device from two sub-devices
    ELECTRODE_DEVICE_NAMES = ['S0844pad6shank', 'S0844pad8shank']  # combined = S1688pad14shank
    HEADSTAGE_DEVICE_NAME = 'MEA1K22'
    date = '260320'
    batch = 1
    IMPLANT_DEVICE_NAME = f"{date}_{HEADSTAGE_DEVICE_NAME}_S1688pad14shankB{batch}"
    rec_dir_name = 'testBond4_ShubhamW1_14Shank_VrefFPGAStim_ampl15'
    connectivity_rec_path = os.path.join(nas_dir, 'devices', 'implant_devices',
                                         IMPLANT_DEVICE_NAME, 'recordings', rec_dir_name)
    align_pads2mea1k(ELECTRODE_DEVICE_NAMES, IMPLANT_DEVICE_NAME, connectivity_rec_path)
    plot_pad_alignment(IMPLANT_DEVICE_NAME)


if __name__ == '__main__':
    main()