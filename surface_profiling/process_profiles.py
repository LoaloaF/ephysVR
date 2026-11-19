import sys
import time
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from export_vk_to_xyz import read_vk, export_xyz, relevel_plane, relevel_plane_points, read_optical_from_cag
from export_obj import export_obj
from export_stl import export_stl
from mea_grid import run_mea_grid, cell_outlines

# ── root for all scan sessions (subdir per file below) ───────────────────────
SCAN_ROOT = Path('/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/devices/surface_profiles/')

# ── global settings (all files) ───────────────────────────────────────────────
SIGMA        = 0.0    # Gaussian smoothing (px); 0 = off
Z_SCALE      = 1.0    # height exaggeration for OBJ/STL
HEATMAP_CMAP = 'viridis'

SHOW_HEATMAP   = True
SHOW_HISTOGRAM = True

DO_EXPORT_XYZ = False
DO_EXPORT_OBJ = True
DO_EXPORT_STL = False

DO_MEA_GRID   = False   # per-electrode crops + features for files with mea_grid=True

# ── per-file config ───────────────────────────────────────────────────────────
# subdir   : folder name under SCAN_ROOT (required per file)
# relevel  : False | 'plane' (fit all pixels) | 'points' (use relevel_points)
# vmin     : subtracted from z; sets colorscale floor to 0.  None = 1st percentile
# vmax     : colorscale ceiling in µm (after vmin shift).    None = 99th percentile
# subarea  : (x, y, w, h) µm — second histogram overlay + rectangle on heatmap
# mea_grid : True | dict(kwargs for run_mea_grid) — fit 220×120 electrode lattice
#            (needs el 0 top-left after rotate_180, and ~0.7 µm/px resolution)
FILES = {
    # # 26-07-17
    # 'B7reEtched_spring_interconnect_full_1821.vk6': dict(
    #     name           = 'B7reEtched_spring_interconnect',
    #     subdir         = '2026-07-17_Profilometer',
    #     relevel        = 'points',
    #     relevel_points = [(51, 135), (2700, 135), (55, 1215), (2747, 1203)],
    #     rotate_180     = False,
    #     zero_min       = True,
    #     vmin           = 3.2,
    #     vmax           = 3.5,
    #     subarea        = None,
    # ),
    
    
    
    # # # 26-07-17 mea1k 24
    # 'MEA1K24.vk4': dict(
    #     name           = 'MEA1K24_half_half_goldplating',
    #     subdir         = '2026-07-17_Profilometer',
    #     relevel        = 'points',
    #     relevel_points = [(323, 210), (4101, 239), (306, 2292), (4065, 2270)],
    #     rotate_180     = False,
    #     zero_min       = True,
    #     vmin           = 3.7,
    #     vmax           = None,
    #     # subarea        = [1000, 1000, 2000, 1000],
    #     mea_grid       = True,
        
    # ),
    # # 26-07-28 mea1k 23
    # 'MEA1K23.vk6': dict(
    #     name           = 'MEA1K23_best_bonding_results',
    #     subdir         = '2026-07-28_Profilometer',
    #     relevel        = 'points',
    #     relevel_points = [(400, 199), (4178, 201), (363, 2227), (4176, 2200)],
    #     rotate_180     = True,
    #     zero_min       = True,
    #     vmin           = 10.5,
    #     vmax           = 3.5,
    #     subarea        = [1000, 1000, 2000, 1000],
    #     mea_grid       = True,
        
    # ),
    
    
    # # 26-08-24 — J1 device, new, metal down
    # 'J1_device_metaldown.vk4': dict(
    #     subdir         = '2026-08-24_profilometer',
    #     relevel        = 'points',
    #     relevel_points = [(300, 40), (4107, 40), (354, 1021), (4006, 1080)],
    #     zero_min       = True,
    #     vmin           = 1.3,
    #     vmax           = 6,
    #     subarea        = [3200, 50, 800, 800],
    # ),
    
    # # 26-08-24
    # 'noPitMEA_metaldown_J1device.vk4': dict(
    #     subdir         = '2026-08-24_profilometer',
    #     relevel        = 'plane',
    #     rotate_180     = True,
    #     subarea        = [3200, 450, 1000, 1000],
    # ),
    # # 26-08-24
    # 'MEA1K22.vk4': dict(
    #     subdir         = '2026-08-24_profilometer',
    #     relevel_points = [(314, 194), (4105, 184), (308,2227), (4101,2220)],
    #     relevel        = 'points',
    #     vmin           = 4.4,
    #     vmax           = 3.5,
    #     zero_min       = True,
    #     rotate_180     = True, # put points like this, now have to use it, but not "real"
    #     subarea        = [50, 50, 1000, 1000],
    #     mea_grid       = True,
    # ),
    # 26-08-24
    # 'MEA1K_24_after_bond_6x.vk4': dict(
    #     name           = 'MEA1K_24_after_bond_bottomBadResults',
    #     subdir         = '2026-08-24_profilometer',
    #     relevel_points = [[417, 250], [4032, 166], [295,979], [3628,875]],
    #     relevel        = 'points',
    #     vmin           = 6.5,
    #     vmax           = 3.5,
    #     zero_min       = True,
    #     subarea        = [1800, 50, 600, 600]
    # ),
    # 'MEA1K_24_after_bond_8x.vk4': dict(
    #     name           = 'MEA1K_24_after_bond_TopGoodResults',
    #     subdir         = '2026-08-24_profilometer',
    #     relevel_points = [[566, 481], [4077, 488], [578,1320], [4278,1218]],
    #     relevel        = 'points',
    #     vmin           = 4.9,
    #     vmax           = 3.5,
    #     zero_min       = True,
    #     subarea        = [1800, 350, 600, 600]
    # ),
    
    # # # 26-10-30
    # 'MEA1K_23.vk4': dict(
    #     subdir         = '2026-10-30_profilometer',
    #     relevel_points = [(369, 362), (4164, 174), (408,2210), (4184,2218)],
    #     relevel        = 'points',
    #     vmin           = 15.5,
    #     vmax           = 3.5,
    #     zero_min       = True,
    #     # rotate_180     = True, # put points like this, now have to use it, but not "real"
    #     # subarea        = [50, 50, 1000, 1000],
    #     mea_grid       = True,
    # ),
    # # # 26-10-30
    # 'MEA1K_24.vk4': dict(
    #     subdir         = '2026-10-30_profilometer',
    #     relevel_points = [(250, 154), (4060, 167), (286,2220), (4062,2219)],
    #     relevel        = 'points',
    #     vmin           = 3.75,
    #     vmax           = 3.5,
    #     zero_min       = True,
    #     # rotate_180     = True, # put points like this, now have to use it, but not "real"
    #     # subarea        = [50, 50, 1000, 1000],
    #     mea_grid       = True,
    # ),
    # # # 26-10-30
    'MEA1K_27.vk4': dict(
        subdir         = '2026-10-30_profilometer',
        relevel_points = [(294, 166), (4105, 165), (292,2229), (4103,2232)],
        relevel        = 'points',
        vmin           = 14,
        vmax           = 3.5,
        zero_min       = True,
        # rotate_180     = True, # put points like this, now have to use it, but not "real"
        subarea        = [150, 2100, 2000, 250],
        mea_grid       = True,
    ),
    
    # # # 26-10-30
    # 'Interconnect_edge.vk6': dict(
    #     subdir         = '2026-10-30_profilometer',
    # ),
    
    
    
    
    
    
    
    
}

# ── progress log ──────────────────────────────────────────────────────────────
_T0 = time.time()

def log(msg):
    """Timestamped progress line (seconds since start), flushed immediately."""
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def show(what):
    """plt.show() blocks until the window is closed — say so, and time it."""
    log(f"showing {what} — close the window to continue")
    t = time.time()
    plt.show()
    log(f"{what} window closed after {time.time() - t:.0f}s")


# ── main loop ─────────────────────────────────────────────────────────────────
for fname, cfg in FILES.items():
    if 'subdir' not in cfg:
        raise ValueError(f"{fname}: missing 'subdir' key in FILES config")
    base_dir = SCAN_ROOT / cfg['subdir']
    out      = base_dir / 'exported'
    p        = base_dir / fname
    out.mkdir(parents=True, exist_ok=True)

    relevel        = cfg.get('relevel',        False)
    relevel_points = cfg.get('relevel_points', None)
    rotate_180     = cfg.get('rotate_180',     False)
    zero_min       = cfg.get('zero_min',       True)
    vmin           = cfg.get('vmin',           None)
    vmax           = cfg.get('vmax',           None)
    subarea        = cfg.get('subarea',        None)

    # ── read ──────────────────────────────────────────────────────────────────
    print()
    log(f"Reading {p.name} ...")
    data = read_vk(p)
    log("read done")
    m    = data['meta']
    print(f"  Grid        : {m['nx']} × {m['ny']} pixels")
    print(f"  Pixel size  : dx={m['dx_um']:.4f} µm  dy={m['dy_um']:.4f} µm")
    print(f"  Scan area   : {m['x_range_um']:.1f} × {m['y_range_um']:.1f} µm")
    print(f"  z range     : {m['z_min_um']:.4f} … {m['z_max_um']:.4f} µm  (span {m['z_range_um']:.4f} µm)")
    print(f"  z=0 pixels  : {(data['z'] == 0).sum():,} / {data['z'].size:,}")

    log("checking for companion .cag ...")
    optical = read_optical_from_cag(p, m['nx'], m['ny'])
    if optical is not None:
        print(f"  Optical reflectance loaded  (shape {optical.shape})")
    else:
        print("  No .cag found — using false-color height texture")

    # ── transforms ────────────────────────────────────────────────────────────
    if rotate_180:
        data['z'] = data['z'][::-1, ::-1].copy()
        print("\nRotated 180°")

    if relevel:
        print()
        log("releveling ...")
        if relevel == 'points':
            if not relevel_points:
                raise ValueError(f"{fname}: relevel='points' requires relevel_points")
            data = relevel_plane_points(data, relevel_points)
        else:
            data = relevel_plane(data)
        m = data['meta']
        print(f"  z range after relevel: {m['z_min_um']:.4f} … {m['z_max_um']:.4f} µm  (span {m['z_range_um']:.4f} µm)")

    z_shift = 0.0                     # total offset subtracted after releveling (for mea_grid)
    if zero_min:
        z_off = float(np.nanmin(data['z']))
        data['z'] -= z_off
        z_shift += z_off
        data['meta'].update(z_min_um=0.0,
                            z_max_um=float(np.nanmax(data['z'])),
                            z_range_um=float(np.nanmax(data['z'])))
        m = data['meta']
        print(f"\nZero-min: subtracted {z_off:.4f} µm  →  z now {m['z_min_um']:.4f} … {m['z_max_um']:.4f} µm")

    if vmin is not None:
        data['z'] -= vmin
        z_shift += vmin
        data['meta'].update(z_min_um=float(np.nanmin(data['z'])),
                            z_max_um=float(np.nanmax(data['z'])),
                            z_range_um=float(np.nanmax(data['z']) - np.nanmin(data['z'])))
        m = data['meta']
        print(f"VMIN shift: subtracted {vmin:.4f} µm  →  z now {m['z_min_um']:.4f} … {m['z_max_um']:.4f} µm")

    # ── shared colorscale ─────────────────────────────────────────────────────
    z_all   = data['z'].ravel()
    z_valid = z_all[np.isfinite(z_all)]
    _vmin = 0.0  if vmin is not None else float(np.nanpercentile(z_valid, 1))
    _vmax = vmax if vmax is not None else float(np.nanpercentile(z_valid, 99))

    # ── histogram ─────────────────────────────────────────────────────────────
    if SHOW_HISTOGRAM:
        log("histogram ...")
        bins = np.linspace(z_valid.min(), z_valid.max(), 201)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(z_valid, bins=bins, color='steelblue', edgecolor='none', alpha=0.7, label='full scan')

        if subarea is not None:
            sx, sy, sw, sh = subarea
            mask  = ((data['x'] >= sx) & (data['x'] <= sx + sw) &
                     (data['y'] >= sy) & (data['y'] <= sy + sh))
            z_sub = data['z'][mask]
            z_sub = z_sub[np.isfinite(z_sub)]
            ax.hist(z_sub, bins=bins, color='tomato', edgecolor='none', alpha=0.7,
                    label=f'subarea x={sx}+{sw} y={sy}+{sh} µm  (n={len(z_sub):,})')
            print(f"\nSubarea z: {z_sub.min():.4f} … {z_sub.max():.4f} µm  median {np.median(z_sub):.4f} µm")

        ax.axvline(np.nanmedian(z_valid), color='orange', lw=1.2,
                   label=f'median {np.nanmedian(z_valid):.3f} µm')
        ax.axvline(_vmin, color='darkgray',  lw=1.2, ls='--', label=f'cmap min {_vmin:.3f} µm')
        ax.axvline(_vmax, color='lightgray', lw=1.2, ls='--', label=f'cmap max {_vmax:.3f} µm')
        ax.set_xlabel('Height (µm)')
        ax.set_ylabel('Pixel count')
        ax.set_title(f'{cfg.get("name", p.stem)}  —  height histogram{"  [releveled]" if relevel else ""}')
        ax.legend(fontsize=9)
        plt.tight_layout()
        show("histogram")

    # ── MEA electrode grid ────────────────────────────────────────────────────
    mea_grid  = cfg.get('mea_grid', False)
    mea_cells = None
    if DO_MEA_GRID and mea_grid:
        print()
        log("MEA electrode grid ...")
        mea_table, _, mea_lat = run_mea_grid(data, out, p.stem, z_ref_offset_um=z_shift,
                                             **(mea_grid if isinstance(mea_grid, dict) else {}))
        mea_cells = cell_outlines(mea_table, mea_lat)
        log("MEA electrode grid done")

    # ── heatmap ───────────────────────────────────────────────────────────────
    if SHOW_HEATMAP:
        log("heatmap ...")
        steps = []
        if relevel:           steps.append(f'releveled ({relevel})')
        if zero_min:          steps.append('zero-min')
        if vmin is not None:  steps.append(f'floor −{vmin} µm')
        subtitle = '  |  '.join(steps) if steps else 'raw'

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data['z'].T, origin='upper', cmap=HEATMAP_CMAP, vmin=_vmin, vmax=_vmax,
                       extent=[0, m['x_range_um'], m['y_range_um'], 0])
        cb = plt.colorbar(im, ax=ax, label='Height (µm)', shrink=0.8, pad=0.02)
        cb.ax.tick_params(labelsize=9)

        if 'relevel_ref_points' in m:
            _dx2, _dy2 = m['dx_um'] / 2, m['dy_um'] / 2
            for i, (rpx, rpy) in enumerate(m['relevel_ref_points']):
                ax.plot(rpx + _dx2, rpy + _dy2, 'o', ms=6, mec='cyan', mfc='none', mew=1.5)
                ax.text(rpx + _dx2, rpy + _dy2, f'{i+1}', color='cyan', fontsize=7,
                        ha='center', va='center')

        if subarea is not None:
            from matplotlib.patches import Rectangle
            sx, sy, sw, sh = subarea
            ax.add_patch(Rectangle((sx, sy), sw, sh, linewidth=1.5,
                                   edgecolor='tomato', facecolor='none',
                                   label=f'subarea {sw}×{sh} µm'))
            ax.legend(fontsize=9, loc='upper right')

        ax.set_xlabel('x (µm)', fontsize=11)
        ax.set_ylabel('y (µm)', fontsize=11)
        ax.set_title(cfg.get('name', p.stem), fontsize=12, fontweight='bold')
        ax.set_title(subtitle, fontsize=9, color='gray', loc='right')
        ax.tick_params(labelsize=9)
        plt.tight_layout()

        heatmap_path = out / f"{p.stem}_heatmap.png"
        log(f"saving {heatmap_path.name} (300 dpi) ...")
        fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"\nHeatmap colorscale: {_vmin:.4f} … {_vmax:.4f} µm")
        print(f"  → {heatmap_path.name}  (300 dpi)")
        show("heatmap")

    # ── MEA grid check: full heatmap with cell outlines + zooms on the array corners ──
    if mea_cells is not None:
        log("grid check figure ...")
        from matplotlib.collections import LineCollection
        cells = mea_cells + (m['dx_um'] / 2, m['dy_um'] / 2)   # grid coords are pixel centres
        flag  = mea_table['flag'].to_numpy()
        ext   = [0, m['x_range_um'], m['y_range_um'], 0]
        fig   = plt.figure(figsize=(20, 15))
        gs    = fig.add_gridspec(2, 5, height_ratios=[2.2, 1])
        ax    = fig.add_subplot(gs[0, :])
        ax.imshow(data['z'].T, origin='upper', cmap=HEATMAP_CMAP, vmin=_vmin, vmax=_vmax, extent=ext)
        ax.add_collection(LineCollection(cells[~flag], colors='white', linewidths=0.1))
        ax.add_collection(LineCollection(cells[flag],  colors='red',   linewidths=0.2))
        ax.set_title(f"{cfg.get('name', p.stem)} — fitted electrode grid (red = flagged); zoom in, "
                     f"corners below", fontsize=11, fontweight='bold')
        W = 120                                                  # zoom window (µm)
        last = len(cells) - 1
        for k, (lbl, el, sx, sy) in enumerate([('el 0 (top-left)', 0, -1, -1),
                                                ('el 219 (top-right)', 219, 1, -1),
                                                ('centre', 120 // 2 * 220 + 110, 0, 0),
                                                (f'el {last - 219} (bottom-left)', last - 219, -1, 1),
                                                (f'el {last} (bottom-right)', last, 1, 1)]):
            axz = fig.add_subplot(gs[1, k])
            axz.imshow(data['z'].T, origin='upper', cmap=HEATMAP_CMAP, vmin=_vmin, vmax=_vmax, extent=ext,
                       interpolation='nearest')
            axz.add_collection(LineCollection(cells[~flag], colors='white', linewidths=0.7))
            axz.add_collection(LineCollection(cells[flag],  colors='red',   linewidths=0.9))
            axz.add_collection(LineCollection(cells[[el]], colors='magenta', linewidths=2))
            cx_, cy_ = cells[el, :4].mean(0)
            x0_ = cx_ - W / 2 + sx * W / 4; y0_ = cy_ - W / 2 + sy * W / 4   # corner cell offset inwards
            axz.set_xlim(x0_, x0_ + W); axz.set_ylim(y0_ + W, y0_)
            axz.set_title(f'{lbl}  (magenta)', fontsize=9)
            axz.tick_params(labelsize=7)
        plt.tight_layout()
        grid_path = out / f"{p.stem}_grid_check.png"
        log(f"saving {grid_path.name} (300 dpi) ...")
        fig.savefig(grid_path, dpi=300, bbox_inches='tight')
        print(f"  → {grid_path.name}")
        show("grid check")

    # ── export ────────────────────────────────────────────────────────────────
    def _do_exports(d, stem, opt=None):
        if opt is not None:
            nx_o, ny_o = opt.shape
            opt_path = out / f"{stem}_optical.png"
            fig, ax = plt.subplots(figsize=(nx_o / 100, ny_o / 100), dpi=100)
            ax.imshow(opt.T, cmap='gray', vmin=0, vmax=1, origin='upper', aspect='auto')
            ax.axis('off')
            fig.subplots_adjust(0, 0, 1, 1)
            fig.savefig(opt_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"    → {opt_path.name}  (optical reflectance)")
        if DO_EXPORT_XYZ:
            log("XYZ export ...")
            export_xyz(d, stem, out)
        if DO_EXPORT_OBJ:
            log(f"OBJ export ({d['z'].size:,} vertices) ...")
            export_obj(d, stem, out, sigma=SIGMA, z_scale=Z_SCALE,
                       cmap=HEATMAP_CMAP, vmin=_vmin, vmax=_vmax)
        if DO_EXPORT_STL:
            log("STL export ...")
            export_stl(d, stem, out, sigma=SIGMA, z_scale=Z_SCALE)

    print()
    log(f"Exporting to {out}/  (σ={SIGMA} px, Z×{Z_SCALE})")
    if subarea is not None:
        sx, sy, sw, sh = subarea
        ix0 = int(np.clip(round(sx / m['dx_um']),        0, m['nx']))
        ix1 = int(np.clip(round((sx + sw) / m['dx_um']), 0, m['nx']))
        iy0 = int(np.clip(round(sy / m['dy_um']),        0, m['ny']))
        iy1 = int(np.clip(round((sy + sh) / m['dy_um']), 0, m['ny']))
        sub = {'x': data['x'][ix0:ix1, iy0:iy1],
               'y': data['y'][ix0:ix1, iy0:iy1],
               'z': data['z'][ix0:ix1, iy0:iy1],
               'meta': m}
        sub_opt = optical[ix0:ix1, iy0:iy1] if optical is not None else None
        sub_stem = f"{p.stem}_sub"
        print(f"Subarea only ({ix1-ix0}×{iy1-iy0} px)  →  {sub_stem}.*")
        _do_exports(sub, sub_stem, opt=sub_opt)
    else:
        _do_exports(data, p.stem, opt=optical)

    log(f"{p.name} done.")
