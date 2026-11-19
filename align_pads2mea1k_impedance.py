"""Basic demo: align pads to MEA1k electrodes using an impedance measurement
instead of a connectivity-shortcut measurement.

Reuses the alignment machinery from assign_pads.py as-is. The only things that
change are (1) the background image shown in napari during manual alignment,
and (2) where the real per-electrode values come from. Both come from
`all_impedance.csv` (columns: electrode, connectivity, impedance_Ohm, ...) —
the background image is just a rasterization of one of those columns for
visual reference, never the source of numeric values.

No implant-name bookkeeping here: this just produces a plain mapping CSV of
pad -> mea1k electrode, saved next to the impedance data.
"""
import os
import sys

import numpy as np
import pandas as pd
import napari

sys.path.append(os.path.dirname(__file__))
from assign_pads import (
    _align, _add_pads_to_napari, _save_affine_cache, _load_affine_cache, PAD_R,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_functionality import device_paths

# same grid the MEA1k electrode canvas uses everywhere else in this codebase
# (see mea1k_modules.mea1k_visualizations.draw_mea1k / CHIP_W, CHIP_H)
CHIP_COLS, CHIP_ROWS = 220, 120
PITCH_UM = 17.5
CANVAS_SHAPE = (2100, 3850)  # (rows, cols) in pixels, matches assign_pads._mea1k_el_pixel_table


def _el_center_yx(el):
    """Pixel center of electrode `el`, matching draw_mea1k's layout."""
    y = (el // CHIP_COLS) * PITCH_UM + PITCH_UM / 2
    x = (el % CHIP_COLS) * PITCH_UM + PITCH_UM / 2
    return y, x


def build_background_image(el_values, canvas_shape=CANVAS_SHAPE):
    """Rasterize a per-electrode value Series (indexed by el) onto the MEA1k
    pixel canvas, purely as a visual backdrop for manual napari alignment."""
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    vals = el_values.clip(lower=0).fillna(0)
    vmax = vals.max() or 1
    vals = (vals / vmax * 255).astype(np.uint8)
    half = int(PITCH_UM // 2)
    for el, v in vals.items():
        y, x = _el_center_yx(el)
        y, x = int(y), int(x)
        canvas[y - half:y + half, x - half:x + half] = v
    return canvas


def align_pads2mea1k_from_impedance(electrode_device_names, impedance_csv_fullfname,
                                     value_col='connectivity', out_csv_fullfname=None,
                                     affine_cache_dir=None, force_realign=False):
    if isinstance(electrode_device_names, str):
        electrode_device_names = [electrode_device_names]

    # nas_dir = device_paths()[0]
    path = '/Users/loaloa/Documents/sqaure_interconnects/'

    # --- load pad mappings (device geometry, unrelated to image/values) --------
    device_mappings = []
    for device_name in electrode_device_names:
        mapping_fullfname = os.path.join(
            path, device_name,
            f"device_mapping_{device_name}.csv")
        mapping = pd.read_csv(mapping_fullfname, index_col=[0])
        mapping.index = mapping.pad_id.values
        device_mappings.append((device_name, mapping))

    # --- real values come from the table, never the image ----------------------
    impedance = pd.read_csv(impedance_csv_fullfname)
    impedance = impedance.rename(columns={'electrode': 'el'})
    impedance = impedance[~impedance['el'].duplicated(keep='first')]
    impedance = impedance.set_index('el').reindex(np.arange(26400))
    mea1k_connectivity = impedance[[value_col]].rename(columns={value_col: 'mea1k_connectivity'})

    # --- background image is just a rasterization of that same column ----------
    mea1k_connectivity_png = build_background_image(mea1k_connectivity['mea1k_connectivity'])

    # affine cache lives next to the impedance data unless told otherwise
    affine_cache_dir = affine_cache_dir or os.path.dirname(impedance_csv_fullfname)
    os.makedirs(affine_cache_dir, exist_ok=True)

    viewer = napari.Viewer()
    viewer.add_image(mea1k_connectivity_png, name='Impedance connectivity',
                     colormap='gray_r', opacity=.5)

    alignments = []
    for device_name, mapping in device_mappings:
        layer = _add_pads_to_napari(viewer, mea1k_connectivity_png, mapping,
                                    layer_name=device_name, square_size=40)

        cache_path = os.path.join(affine_cache_dir, f'affine_cache_{device_name}.json')
        cached = None
        if os.path.exists(cache_path) and not force_realign:
            import json
            with open(cache_path) as f:
                cached = np.array(json.load(f))
        if cached is not None:
            layer.affine = cached
            print(f"  [{device_name}] Restored affine from cache.")
        else:
            print(f"  [{device_name}] No affine cache found — align manually in napari.")
        alignments.append((device_name, mapping, layer))

    print("Opening napari for manual alignment...")
    napari.run()

    # --- save affines after user closes napari ---
    for device_name, mapping, layer in alignments:
        cache_path = os.path.join(affine_cache_dir, f'affine_cache_{device_name}.json')
        import json
        with open(cache_path, 'w') as f:
            json.dump(layer.affine.affine_matrix.tolist(), f, indent=2)
        print(f"  Saved affine cache: {cache_path}")

    # --- compute pad -> electrode mapping for every device ----------------------
    all_pad_alignments = []
    for device_name, mapping, layer in alignments:
        print(f"Aligning {len(mapping)} pads to MEA1k electrodes ({device_name})")
        all_pad_alignments.append(_align(mapping, mea1k_connectivity, layer))
    pad_alignment = pd.concat(all_pad_alignments, axis=0).reset_index(drop=True)

    out_csv_fullfname = out_csv_fullfname or os.path.join(
        os.path.dirname(impedance_csv_fullfname), 'pad2el_mapping.csv')
    pad_alignment.to_csv(out_csv_fullfname, index=False)
    print(f"Saved pad->electrode mapping: {out_csv_fullfname}")
    return pad_alignment


if __name__ == '__main__':
    ELECTRODE_DEVICE_NAMES = ['35_36_768pads',]
    IMPEDANCE_CSV = ("/Users/loaloa/homedataAir/phd/ratvr/mea1k_paper/bonding_yield/"
                     "100k_bonding_yield/test8/test8_MEA23_36umP_t00_PT2_imp/all_impedance.csv")
    IMPEDANCE_CSV = ("/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/devices/headstage_devices/MEA1K27/recordings/2026-10-28_14.57_test11_100k_36umP_PT7_Imp/processed/all_impedance.csv")
    IMPEDANCE_CSV = ("/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/devices/headstage_devices/MEA1K24/recordings/2026-11-11_15.10_test17_100_36umP_PT5_Imp/processed/all_impedance.csv")

    align_pads2mea1k_from_impedance(ELECTRODE_DEVICE_NAMES, IMPEDANCE_CSV,
                                    value_col='connectivity', force_realign=False)
