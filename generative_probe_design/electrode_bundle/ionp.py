"""IONP (iron-oxide nanoparticle) well patterns on the electrode bundle.

For each of the 64 electrode contacts, a vertical column of small circular wells is laid
down the straight-shank region above the contact (from just above the contact up to where
the fanout begins ~4990um), keeping only the wells that fall inside the normalized stripe
bands in `cfg.pattern`. Wells go onto a new `pattern_ionp` layer.

`add_ionp_wells` takes the electrode contact locations in memory (from BundleResult), so in
the `all` pipeline there is no DXF round-trip. Standalone, `electrode_locs_from_dxf` reads
them back off an existing DXF's `Electrodes` layer instead.
"""
import json
from typing import Any, Dict, Tuple

import numpy as np

from . import dxf_io as io
from .config import (IonpConfig, ID_TOP_ANCHOR_CENTER, ID_SLOT_CENTERS, id_codeword)


def filter_to_pattern(ylocs: np.ndarray, pattern: np.ndarray,
                      y_top: float, y_bottom: float) -> np.ndarray:
    """Keep only y positions inside the normalized stripe intervals."""
    yrange = y_top - y_bottom
    mask = np.zeros_like(ylocs, dtype=bool)
    for start, end in pattern:
        mask |= (ylocs >= start * yrange + y_bottom) & (ylocs <= end * yrange + y_bottom)
    return ylocs[mask]


def calc_ionp_well_locs(el_locs, pattern, y_top, y_bottom,
                        well_distance, first_well_distance) -> Tuple[np.ndarray, np.ndarray]:
    """One vertical column of wells above each electrode, clipped to the pattern."""
    xs, ys = [], []
    for el_x, el_y in el_locs:
        col_y = np.arange(el_y + first_well_distance, y_top, well_distance)
        col_y = filter_to_pattern(col_y, pattern, y_top, y_bottom)
        xs.append(np.full_like(col_y, el_x))
        ys.append(col_y)
    return np.concatenate(xs), np.concatenate(ys)


def drop_wells_near_contacts(xs: np.ndarray, ys: np.ndarray,
                             el_locs: np.ndarray, radius: float
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Remove wells within `radius` (um) of any contact centre. radius<=0 -> no-op."""
    if radius <= 0 or len(xs) == 0:
        return xs, ys
    el = np.asarray(el_locs, dtype=float)
    keep = np.ones(len(xs), dtype=bool)
    r2 = radius * radius
    for cx, cy in el:
        keep &= (xs - cx) ** 2 + (ys - cy) ** 2 > r2
    return xs[keep], ys[keep]


def effective_y_top(el_locs, cfg: IonpConfig) -> float:
    """Top of the ID region (norm 1.0). `id_top_margin` um above the highest contact, or
    `cfg.y_top` (the fan) when `id_top_margin is None`."""
    if cfg.id_top_margin is None:
        return cfg.y_top
    return float(np.asarray(el_locs, dtype=float)[:, 1].max()) + cfg.id_top_margin


def _band_wells(band, el_locs, y_top, y_bottom, cfg: IonpConfig) -> int:
    """Number of wells a single normalized band would hold (after the contact keep-out)."""
    xs, ys = calc_ionp_well_locs(el_locs, np.array([band]), y_top, y_bottom,
                                 cfg.ionp_well_distance, cfg.ionp_firstwell_distance)
    xs, ys = drop_wells_near_contacts(xs, ys, el_locs, cfg.keepout_radius)
    return len(xs)


def _grow_band(center, el_locs, y_top, y_bottom, cfg: IonpConfig,
               start_h_um, step_um=20.0, max_h_um=3000.0):
    """Symmetric band about `center` (normalized), grown from `start_h_um` until it holds
    >= cfg.min_n_wells wells. Anchors pass a thick start height; data slots start thin."""
    yr = y_top - y_bottom
    h = start_h_um
    while True:
        half = (h / yr) / 2.0
        band = [center - half, center + half]
        if _band_wells(band, el_locs, y_top, y_bottom, cfg) >= cfg.min_n_wells or h >= max_h_um:
            return band
        h += step_um


def _grow_tip_anchor(el_locs, y_top, y_bottom, cfg: IonpConfig,
                     start_h_um, step_um=20.0, max_h_um=3000.0):
    """Bottom anchor pinned at the tip: band [y_bottom, y_bottom+h] grown UP from `start_h_um`
    until it holds >= cfg.min_n_wells wells (the sparse tip makes it the fattest stripe)."""
    yr = y_top - y_bottom
    h = start_h_um
    while True:
        band = [0.0, h / yr]
        if _band_wells(band, el_locs, y_top, y_bottom, cfg) >= cfg.min_n_wells or h >= max_h_um:
            return band
        h += step_um


def build_id_pattern(i: int, el_locs, cfg: IonpConfig) -> np.ndarray:
    """Geometry-aware barcode for ID i in 0..15: a tip-pinned bottom anchor + a top anchor
    bracketing the 7 Hamming data slots present for `i` (min pairwise distance 3). Every stripe
    auto-sized to clear the MRI well floor. Returns normalized [start, end] bands, sorted."""
    assert 0 <= i < 16, f"ID {i} out of range 0..15"
    el_locs = np.asarray(el_locs, dtype=float)
    y_top, yb = effective_y_top(el_locs, cfg), cfg.y_bottom
    _SLOT_SEED_H_UM = 20.0   # data slots start thin and grow up to the well floor
    bands = [_grow_tip_anchor(el_locs, y_top, yb, cfg, cfg.id_bot_anchor_h_um)]
    code = id_codeword(i)
    for bit, center in zip(code, ID_SLOT_CENTERS):
        if bit:
            bands.append(_grow_band(center, el_locs, y_top, yb, cfg, _SLOT_SEED_H_UM))
    bands.append(_grow_band(ID_TOP_ANCHOR_CENTER, el_locs, y_top, yb, cfg, cfg.id_top_anchor_h_um))
    return np.array(sorted(bands, key=lambda b: b[0]))


def check_design_rules(pattern, y_top, y_bottom, ionp_ylocs, cfg: IonpConfig) -> None:
    yrange = y_top - y_bottom
    prev_y_end = None
    total_ok = True
    for i, (start, end) in enumerate(pattern):
        y_start = start * yrange + y_bottom
        y_end = end * yrange + y_bottom
        n = int(np.sum((ionp_ylocs >= y_start) & (ionp_ylocs <= y_end)))
        print(f"strip {i}: y {y_start:7.1f}-{y_end:7.1f} um  ({y_end-y_start:6.1f} um)  {n:6d} wells")
        if prev_y_end is not None and y_start - prev_y_end < cfg.min_strip_distance:
            print(f"  WARNING: strip {i} only {y_start-prev_y_end:.0f}um from strip {i-1} (<{cfg.min_strip_distance})")
            total_ok = False
        if n < cfg.min_n_wells or n > cfg.max_n_wells:
            print(f"  WARNING: {n} wells outside recommended {cfg.min_n_wells}-{cfg.max_n_wells}")
            total_ok = False
        prev_y_end = y_end
    print(f"design rules: {'OK' if total_ok else 'see warnings above'}")


def write_pattern_summary_to_json(pattern, y_top, y_bottom, el_locs,
                                  ionp_x, ionp_y, cfg: IonpConfig, out_fname,
                                  flex_mapping: Dict[str, Any] = None) -> Dict[str, Any]:
    """Write per-strip IONP metadata to JSON, matching the shanks/*_info.json schema.

    `flex_mapping`, if given, is the channel->flex-pad summary from mapping.build_mapping;
    it is embedded as a top-level `flex_mapping` block (right after `electrodes`)."""
    yrange = y_top - y_bottom
    strips = {}
    for i, (start, end) in enumerate(pattern):
        y_start = start * yrange + y_bottom
        y_end = end * yrange + y_bottom
        n = int(np.sum((ionp_y >= y_start) & (ionp_y <= y_end)))
        strips[str(i)] = {
            "name": f"ionp_pattern_{i}",
            "normalized": {"start": float(start), "end": float(end)},
            "real_y_um": {"start": float(y_start), "end": float(y_end),
                          "height": float(y_end - y_start)},
            "n_wells": n,
        }
    summary = {
        "electrodes": {
            "n_electrodes": int(len(el_locs)),
            "locations_um": [{"x": float(x), "y": float(y)} for x, y in el_locs],
        },
    }
    if flex_mapping is not None:
        summary["flex_mapping"] = flex_mapping
    summary.update({
        "ionp_pattern": {
            "name": f"ionp_pattern_{cfg.pattern_i}",
            "n_strips": int(len(pattern)),
            "total_wells": int(len(ionp_x)),
            "normalized_range": {"bottom": 0.0, "top": 1.0},
            "real_y_range_um": {"bottom": float(y_bottom), "top": float(y_top),
                                "height": float(yrange)},
            "strips": strips,
        },
        "metadata": {
            "ionp_pattern_i": int(cfg.pattern_i),
            "ionp_well_d_um": float(cfg.ionp_well_d),
            "ionp_well_distance_um": float(cfg.ionp_well_distance),
            "ionp_firstwell_distance_um": float(cfg.ionp_firstwell_distance),
        },
    })
    with open(out_fname, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {out_fname}")
    return summary


def electrode_locs_from_dxf(msp, cfg: IonpConfig) -> np.ndarray:
    """Read the contact centres off an existing DXF's Electrodes layer (standalone mode)."""
    el_locs = io.extract_electrode_centroids_from_msp(msp, cfg.el_layer)
    print(f"Found {len(el_locs)} electrodes")
    return el_locs


def add_ionp_wells(doc, msp, electrode_locs, cfg: IonpConfig = None):
    """
    Stamp IONP wells onto the open document from in-memory contact centres. Returns
    (ionp_x, ionp_y, el_locs) so the caller can also render / write the summary JSON.
    """
    cfg = cfg or IonpConfig()
    el_locs = np.asarray(electrode_locs, dtype=float)
    y_top = effective_y_top(el_locs, cfg)
    pattern = cfg.pattern if cfg.pattern is not None else build_id_pattern(cfg.pattern_i, el_locs, cfg)
    ionp_x, ionp_y = calc_ionp_well_locs(
        el_locs, pattern, y_top, cfg.y_bottom,
        cfg.ionp_well_distance, cfg.ionp_firstwell_distance)
    ionp_x, ionp_y = drop_wells_near_contacts(ionp_x, ionp_y, el_locs, cfg.keepout_radius)
    print(f"Total wells: {len(ionp_x)}")
    check_design_rules(pattern, y_top, cfg.y_bottom, ionp_y, cfg)
    io.write_wells_into_dxf(doc, msp, ionp_x, ionp_y,
                            cfg.ionp_well_d, cfg.ionp_layer, cfg.well_resolution)
    return ionp_x, ionp_y, el_locs, pattern
