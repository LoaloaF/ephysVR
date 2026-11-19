"""
Bump analysis of the per-electrode cells exported by process_profiles.py
(mea_grid.py → <stem>_mea_cells.csv / .npz in each scan's exported/ folder).

Per chip: what do the gold bumps look like (typical shape + spread), how high are they,
does the shape change with height, and how many bumps have real shape defects. Heights
are in the releveled frame: the relevel points sit on the trench floor between the
electrodes (same level as the flat field around the array), so 0 = trench floor. All
height plots share one scale, so chips are comparable.

Steps
  1. load      — crops + feature table
  2. seams     — stitch-field seams: regular grid (field size SEAM_FIELD_UM) fitted to jumps in
                 the per-cell shifts and steps in the features and the pixel noise
  3. shape     — per denoised cell: align the median bump to it (shift ≤ DEFECT_SHIFT_PX,
                 own height scale + offset) and take the residual; its RMS and the largest
                 lump / dent (contiguous patch deviating by more than lump_dev_um). A bump that
                 is only off-centre does not deviate.
  4. status    — each electrode is included or excluded for the first reason that applies
                 (thresholds in EXCLUDE, overridable on the command line):
                   grid fit failed  correlation with the lattice template < fit_corr_min
                   stitch seam      within seam_margin cells of a stitch seam
                   height outlier   bump top more than height_iqr IQR outside the quartiles
                   pixel noise      jitter_um (from mea_grid) > noise_max_um; noise-free scan
                                    ≈ 0.02–0.03 µm. The noise comes from the per-tile brightness
                                    settings (ND filter off + high PMT gain), not from the chip
                   irregular shape  residual RMS > shape_rms_max_um, or a lump / dent
                                    ≥ lump_area_um2
                 Every statistic and figure below uses only the included electrodes; the
                 'electrodes used' figure shows each metric's histogram with its cut-off.
  5. figures   — bump distribution (low / median / high bump, where bumps differ, profiles),
                 3D median bump (+ rotating video), shape vs height, electrodes used (map +
                 shares), examples per category
  6. compare   — between chips: height distributions, median shapes, difference,
                 profiles (absolute + normalised), bump position, smoothness

Usage:
    python analyze_mea_cells.py                          # everything → SCAN_ROOT/mea_cell_analysis/
    python analyze_mea_cells.py <out_dir>
    python analyze_mea_cells.py --compare                # only the comparison figures
    python analyze_mea_cells.py --chips                  # only the per-chip figures
    python analyze_mea_cells.py noise_max_um=0.15 shape_rms_max_um=0.4   # override thresholds

Output: <out_dir>/<chip>/ per-chip figures (+ video), <out_dir>/comparison/ comparison figures,
<out_dir>/mea_cell_quality.csv status + features of every electrode of every chip.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from scipy.ndimage import gaussian_filter, median_filter, label

# ── config ────────────────────────────────────────────────────────────────────
SCAN_ROOT = Path('/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/devices/surface_profiles/')

# chip label → exported stem (relative to SCAN_ROOT); all chips must share the crop grid
CHIPS = {
    # 'MEA1K22': '2026-08-24_profilometer/exported/MEA1K22',
    # 'MEA1K23': '2026-07-28_profilometer/exported/MEA1K23',
    # 'MEA1K24': '2026-07-17_Profilometer/exported/MEA1K24',   # 2.77 µm/px (6×6 px crops) — rescan

    # 'MEA1K_23': '2026-10-30_profilometer/exported/MEA1K_23', # 10.30
    'MEA1K_24': '2026-10-30_profilometer/exported/MEA1K_24',
    # 'MEA1K_27': '2026-10-30_profilometer/exported/MEA1K_27',

}

OUT_DIR       = SCAN_ROOT / 'mea_cell_analysis'
DO_CHIPS      = True            # per-chip figures      (--chips:   only these)
DO_COMPARE    = True            # comparison figures    (--compare: only these)

# exclusion thresholds; override any of them on the command line as key=value
EXCLUDE = dict(
    fit_corr_min     = 0.5,     # grid fit failed: correlation with the lattice template below this
    seam_margin      = 1,       # stitch seam: cells excluded on each side of a seam (0 = none)
    height_iqr       = 3.0,     # height outlier: bump top more than this many IQR outside the quartiles
    noise_max_um     = 0.15,    # pixel noise: jitter_um above this
    shape_rms_max_um = 0.30,    # irregular shape: RMS deviation from the aligned median bump above this ...
    lump_dev_um      = 1.0,     # ... or a lump / dent deviating by more than this ...
    lump_area_um2    = 10.0,    # ... over a contiguous patch at least this large
)
for arg in sys.argv[1:]:
    if arg in ('--chips', '--compare'):
        DO_CHIPS, DO_COMPARE = arg == '--chips', arg == '--compare'
        continue
    if '=' not in arg:
        OUT_DIR = Path(arg)
        continue
    k, v = arg.split('=', 1)
    if k not in EXCLUDE:
        raise SystemExit(f"unknown threshold {k!r} — choose from {', '.join(EXCLUDE)}")
    EXCLUDE[k] = type(EXCLUDE[k])(v)

SEAM_FIELD_UM = (625, 469)      # profilometer field size (x, y); seam grid is searched ±15 %
SEAMS_MANUAL  = {}              # e.g. {'MEA1K22': dict(cols=[20, 55], rows=[20])} — boundary after
                                # col/row index; overrides the fitted seams for that chip

HEIGHT_MAX_UM = None            # shared colour / axis ceiling for heights; None = 99th pct of all chips
HLBL          = 'height above trench floor (µm)'

DENOISE_MEDIAN_PX = 3           # median filter (removes single-pixel spikes) ...
DENOISE_SIGMA_UM  = 1.0         # ... then Gaussian smoothing, before defect detection / bump top
DEFECT_SHIFT_PX   = 2           # median bump is shifted up to this many px to match each cell
DEFECT_RING_UM    = 1.5         # cell border ignored for shape deviations (trench position jitter)

N_PROFILES    = 80              # single-cell lines in the profile plots
N_EXAMPLES    = 14              # example cells per gallery row
VIDEO         = True            # rotating 3D median bump → <chip>_bump_3d.mp4 (needs ffmpeg)
SEED          = 0

plt.rcParams.update({'figure.titlesize': 18, 'figure.titleweight': 'bold', 'axes.titlesize': 14,
                     'axes.labelsize': 12, 'legend.fontsize': 11})

N_COLS, N_ROWS = 220, 120
BAND_COL  = '#2a78d6'
STATUS    = [('included', '#2e8b57'), ('stitch seam', '#ffffff'), ('pixel noise', '#eb6834'),
             ('irregular shape', '#d6336c'), ('height outlier', '#8c564b'),
             ('grid fit failed', '#7b2cbf')]                                     # status code = index


# ── helpers ───────────────────────────────────────────────────────────────────
def _save(fig, name, sub, dpi=200):
    """Save into OUT_DIR/<sub>/ (a chip label or 'comparison')."""
    path = OUT_DIR / sub / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  → {path.name}")
    plt.show()


def _robust_z(x):
    med = np.median(x)
    return (x - med) / (1.4826 * np.median(np.abs(x - med)) + 1e-9)


def _seam_evidence(t, axis):
    """Evidence for a seam between cells i and i+1 along columns (axis=0) or rows (axis=1):
    jumps in the per-cell recentring shift plus steps in height / gap level / fit quality, and
    (double weight) in pixel noise, which changes sharply between scan fields."""
    sx = t['shift_x_um'].to_numpy().reshape(N_ROWS, N_COLS)
    sy = t['shift_y_um'].to_numpy().reshape(N_ROWS, N_COLS)
    ev = [_robust_z(np.median(np.hypot(np.diff(sx, axis=1 - axis), np.diff(sy, axis=1 - axis)), axis=axis))]
    w = [1]
    for f, wf in (('h_top_um', 1), ('z_gap_um', 1), ('corr', 1), ('jitter_um', 2)):
        if f not in t:
            continue
        p = np.median(t[f].to_numpy(float).reshape(N_ROWS, N_COLS), axis=axis)
        ev.append(_robust_z(np.array([abs(p[i + 1:i + 4].mean() - p[max(i - 2, 0):i + 1].mean())
                                      for i in range(len(p) - 1)])))
        w.append(wf)
    return np.clip(np.average(ev, axis=0, weights=w), 0, None)


def _fit_seams(e, period_cells):
    """Regular seam comb (period ±15 %, any phase) maximising the evidence at its teeth."""
    best = (-np.inf, None)
    for p in np.arange(0.85 * period_cells, 1.15 * period_cells, 0.05):
        for ph in np.arange(0, p, 0.25):
            pos = np.round(ph + np.arange(len(e) / p + 1) * p - 0.5).astype(int)
            pos = pos[(pos >= 1) & (pos < len(e) - 1)]
            if len(pos) < 2:
                continue
            sc = np.mean([e[q - 1:q + 2].max() for q in pos])               # ±1 cell tolerance
            if sc > best[0]:
                best = (sc, pos)
    return best[1]


def _denoise(R, dx):
    return gaussian_filter(median_filter(R, size=(1, DENOISE_MEDIAN_PX, DENOISE_MEDIAN_PX)),
                           (0, DENOISE_SIGMA_UM / dx, DENOISE_SIGMA_UM / dx))


def _defects(G, ref, dx):
    """Fit each denoised cell G (N, n, n) as its own height scale × ref (+ offset), with ref shifted
    by up to DEFECT_SHIFT_PX to the best match; returns the largest contiguous patch (µm²)
    deviating by more than lump_dev_um, the lump mask (patches ≥ lump_area_um2) and the
    residual RMS (µm)."""
    N, n, _ = G.shape
    o = (np.arange(n) - (n - 1) / 2) * dx
    inner = np.maximum(*np.meshgrid(np.abs(o), np.abs(o), indexing='ij')) <= n * dx / 2 - DEFECT_RING_UM
    S = DEFECT_SHIFT_PX
    rp = np.pad(ref, S, mode='edge')
    A = G[:, inner] - G[:, inner].mean(1, keepdims=True)
    best, res = np.full(N, np.inf), np.zeros_like(A)
    for sx in range(-S, S + 1):
        for sy in range(-S, S + 1):
            B = rp[S + sx:S + sx + n, S + sy:S + sy + n][inner]; B = B - B.mean()
            r = A - ((A * B).sum(1) / (B ** 2).sum())[:, None] * B
            sse = (r ** 2).sum(1); b = sse < best
            best[b], res[b] = sse[b], r[b]
    big = np.zeros(G.shape, bool); big[:, inner] = np.abs(res) > EXCLUDE['lump_dev_um']
    lab, _ = label(big, structure=np.pad(np.ones((1, 3, 3), bool), ((1, 1), (0, 0), (0, 0))))   # 2D per cell
    size = np.bincount(lab.ravel()); size[0] = 0
    patch = size[lab] * dx * dx
    return patch.reshape(N, -1).max(1), patch >= EXCLUDE['lump_area_um2'], np.sqrt((res ** 2).mean(1))


def _cell_map(ax, values, title, **kw):
    im = ax.imshow(np.asarray(values, float).reshape(N_ROWS, N_COLS), interpolation='nearest', **kw)
    ax.set_title(title); ax.set_xlabel('column'); ax.set_ylabel('row')
    return im


def _draw_seams(ax, seams, lw=0.8):
    for c in seams['cols']:
        ax.axvline(c + 0.5, color='k', lw=lw, ls=':')
    for r in seams['rows']:
        ax.axhline(r + 0.5, color='k', lw=lw, ls=':')


def _clean(ax):
    ax.spines[['top', 'right']].set_visible(False)


# ── 1. load ───────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

data, dx = {}, None
for chip, stem in CHIPS.items():
    t = pd.read_csv(SCAN_ROOT / f'{stem}_mea_cells.csv')
    d = np.load(SCAN_ROOT / f'{stem}_mea_cells.npz')
    c = d['crops'].astype(np.float32)
    if dx is None:
        dx, shape = float(d['dx_um']), c.shape[1:]
    if c.shape[1:] != shape or not np.isclose(float(d['dx_um']), dx, rtol=1e-3):
        raise ValueError(f"{chip}: crop grid {c.shape[1:]} @ {float(d['dx_um']):.3f} µm/px differs from "
                         f"{shape} @ {dx:.3f} µm/px — chips must share the scan resolution")
    if 'z_frame' not in d:
        print(f"WARNING {chip}: export predates the releveled-frame heights — absolute heights include this file's "
              f"zero_min / vmin offsets and are NOT comparable between chips. Re-run process_profiles.py.")
    # H: height in the releveled frame (relevel plane on the trench floor = 0)
    # R: relief above the cell's own gap ring — used for shape / defects only
    if 'jitter_um' not in t:
        print(f"WARNING {chip}: export has no smoothness columns (jitter_um) — re-run process_profiles.py; "
              f"no electrode is excluded for pixel noise")
    data[chip] = dict(t=t, H=c, R=c - t['z_gap_um'].to_numpy(np.float32)[:, None, None])
n   = shape[0]
SMOOTH = all('jitter_um' in D['t'] for D in data.values())
ext = n * dx / 2
o   = (np.arange(n) - (n - 1) / 2) * dx


# ── 2. seams + 3. defects + 4. status per electrode ──────────────────────────
for chip, D in data.items():
    t, R = D['t'], D['R']
    print(f"\n{chip}")
    if chip in SEAMS_MANUAL:
        seams = dict(cols=list(SEAMS_MANUAL[chip].get('cols', [])), rows=list(SEAMS_MANUAL[chip].get('rows', [])))
    else:
        P = float(t['x_um'].diff().median())                                 # cell pitch (µm)
        seams = dict(cols=[int(c) for c in _fit_seams(_seam_evidence(t, 0), SEAM_FIELD_UM[0] / P)],
                     rows=[int(r) for r in _fit_seams(_seam_evidence(t, 1), SEAM_FIELD_UM[1] / P)])
    print(f"  seams after column {seams['cols']}, after row {seams['rows']}")
    near, M = np.zeros((N_ROWS, N_COLS), bool), EXCLUDE['seam_margin']
    for c in seams['cols'] if M else []:
        near[:, max(c - M + 1, 0):c + M + 1] = True
    for r in seams['rows'] if M else []:
        near[max(r - M + 1, 0):r + M + 1, :] = True
    t['seam'] = near.ravel()
    # scan fields (between seams) and their median pixel noise
    t['field'] = (np.searchsorted(np.array(seams['rows']) + 0.5, t['row']) * 100
                  + np.searchsorted(np.array(seams['cols']) + 0.5, t['col']))
    t['fit_failed'] = t['corr'] < EXCLUDE['fit_corr_min']
    fit_ok = ~t['fit_failed'].to_numpy() & ~t['seam'].to_numpy()
    noisy = (t['jitter_um'] > EXCLUDE['noise_max_um']).to_numpy() if SMOOTH else np.zeros(len(t), bool)
    t['field_noise_um'] = t['field'].map(t.loc[fit_ok].groupby('field')['jitter_um'].median()) if SMOOTH else np.nan

    G = _denoise(R, dx)
    ref = np.median(G[fit_ok & ~noisy] if (fit_ok & ~noisy).any() else G[fit_ok], 0)
    area, mask, rms = _defects(G, ref, dx)
    t['defect_area_um2'], t['shape_rms_um'] = area, rms
    t['defect'] = (area >= EXCLUDE['lump_area_um2']) | (rms > EXCLUDE['shape_rms_max_um'])
    # heights in the releveled frame; bump top from the denoised cell
    t['h_top_um']     = np.percentile(G.reshape(len(G), -1), 95, axis=1) + t['z_gap_um']
    t['h_core_um']    = t['z_core_um']
    t['gap_level_um'] = t['z_gap_um']
    t['volume_um3']   = np.clip(D['H'], 0, None).sum((1, 2)) * dx * dx      # gold above the trench floor

    q1, q3 = np.percentile(t.loc[fit_ok, 'h_top_um'], [25, 75])
    k = EXCLUDE['height_iqr']
    h_lim = (q1 - k * (q3 - q1), q3 + k * (q3 - q1))
    t['height_outlier'] = (t['h_top_um'] < h_lim[0]) | (t['h_top_um'] > h_lim[1])

    # status: first reason that applies (code = index in STATUS)
    t['status'] = np.select([t['fit_failed'], t['seam'], t['height_outlier'], noisy, t['defect']],
                            [5, 1, 4, 2, 3], 0)
    t['status_name'] = t['status'].map(dict(enumerate(s for s, _ in STATUS)))
    use = (t['status'] == 0).to_numpy()
    print('  ' + ', '.join(f"{nm} {(t['status'] == k).mean() * 100:.1f}%" for k, (nm, _) in enumerate(STATUS)))
    D.update(seams=seams, use=use, ref=ref, mask=mask, h_lim=h_lim)

HMAX = HEIGHT_MAX_UM or float(np.percentile(np.concatenate([D['H'][D['use']].ravel() for D in data.values()]), 99))
HNORM = Normalize(0, HMAX)
print(f"\nshared height scale: 0 … {HMAX:.2f} µm above the trench floor (releveled frame)")
for D in data.values():
    Hu = D['H'][D['use']]
    D['p10'], D['p50'], D['p90'] = np.percentile(Hu, [10, 50, 90], axis=0)
    D['mad'] = np.median(np.abs(Hu - D['p50']), 0)          # half of all bumps lie within ± this
MADMAX = max(float(np.percentile(D['mad'], 99)) for D in data.values())
if SMOOTH:
    JMAX = float(np.percentile(np.concatenate([D['t'].loc[D['use'], 'jitter_um'] for D in data.values()]), 99))


# ── 5. figures per chip ───────────────────────────────────────────────────────
for chip, D in (data.items() if DO_CHIPS else []):
    t, use, seams, p50, mad = D['t'], D['use'], D['seams'], D['p50'], D['mad']
    H = D['H']
    Hs = H[use]
    print(f"\n━━ {chip} ━━")
    by = np.argmax(p50.max(0)); bx = np.argmax(p50.max(1))                 # profile lines through the bump

    # ── bump distribution: low / median / high bump, where bumps differ, profiles
    fig = plt.figure(figsize=(20, 10), layout='constrained')
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9])
    map_axs = []
    for j, (img, ttl) in enumerate([(D['p10'], 'low bumps (10th percentile)'), (p50, 'median bump'),
                                    (D['p90'], 'high bumps (90th percentile)')]):
        ax = fig.add_subplot(gs[0, j]); map_axs.append(ax)
        im = ax.imshow(img.T, cmap='viridis', norm=HNORM, extent=[-ext, ext, ext, -ext])
        ax.set_title(ttl); ax.set_xlabel('x (µm)')
    map_axs[0].set_ylabel('y (µm)')
    fig.colorbar(im, ax=map_axs, shrink=0.85, label=HLBL)
    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(mad.T, cmap='magma', vmin=0, vmax=MADMAX, extent=[-ext, ext, ext, -ext])
    ax.set_title('where bumps differ most'); ax.set_xlabel('x (µm)')
    fig.colorbar(im, ax=ax, shrink=0.85, label='typical deviation from the median (± µm)')
    w = 1                                                                    # strip half-width (px)
    pick = rng.choice(len(Hs), min(N_PROFILES, len(Hs)), replace=False)
    for j, (prof, ttl) in enumerate([(Hs[:, :, max(by - w, 0):by + w + 1].mean(2), 'profile along x'),
                                     (Hs[:, max(bx - w, 0):bx + w + 1, :].mean(1), 'profile along y')]):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        for k, i in enumerate(pick):
            ax.plot(o, prof[i], color='0.2', lw=0.6, alpha=0.08, label='single bumps' if k == 0 else None)
        q = np.percentile(prof, [5, 25, 50, 75, 95], axis=0)
        ax.fill_between(o, q[0], q[4], color=BAND_COL, alpha=0.15, lw=0, label='middle 90 %')
        ax.fill_between(o, q[1], q[3], color=BAND_COL, alpha=0.3, lw=0, label='middle 50 %')
        ax.plot(o, q[2], color=BAND_COL, lw=3, label='median')
        ax.set_ylim(-0.5, HMAX * 1.3); ax.set_xlabel('µm from cell centre'); ax.set_ylabel(HLBL)
        ax.set_title(ttl); _clean(ax)
    ax.legend(frameon=False, loc='upper right')
    fig.suptitle(f'{chip}: bump shape')
    _save(fig, f'{chip}_bump_distribution.png', chip)

    # ── 3D median bump coloured by height (+ rotating video)
    X, Y = np.meshgrid(o, o, indexing='ij')

    def _bump3d(ax, elev, azim):
        ax.plot_surface(X, Y, p50, facecolors=plt.cm.viridis(HNORM(p50)), rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)
        ax.set_zlim(0, HMAX); ax.view_init(elev, azim)
        ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)'); ax.set_zlabel('height (µm)')
        ax.invert_yaxis()

    for video in ((False, True) if VIDEO else (False,)):
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection='3d')
        _bump3d(ax, 30, -60)
        fig.colorbar(ScalarMappable(HNORM, 'viridis'), ax=ax, shrink=0.6, pad=0.1, label=HLBL)
        fig.suptitle(f'{chip}: median bump')
        if not video:
            _save(fig, f'{chip}_bump_3d.png', chip)
            continue
        anim = FuncAnimation(fig, lambda k: ax.view_init(30, -60 + k * 1.5), frames=240)
        try:
            anim.save(OUT_DIR / chip / f'{chip}_bump_3d.mp4', writer='ffmpeg', fps=30, dpi=150)
            print(f"  → {chip}_bump_3d.mp4")
        except Exception as e:                                               # no ffmpeg
            print(f"  video skipped ({e})")
        plt.close(fig)

    # ── shape vs height: median bump per height quartile
    h = t.loc[use, 'h_top_um'].to_numpy()
    qs = np.quantile(h, np.linspace(0, 1, 5))
    qi = np.clip(np.searchsorted(qs, h, side='right') - 1, 0, 3)
    names = ['lowest 25 %', '25–50 %', '50–75 %', 'highest 25 %']
    seq = plt.cm.RdPu(np.linspace(0.35, 0.9, 4))
    fig = plt.figure(figsize=(18, 9), layout='constrained')
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.85])
    map_axs = [fig.add_subplot(gs[0, k]) for k in range(4)]
    prof_axs = [fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2:])]
    for k in range(4):
        med = np.median(Hs[qi == k], 0)
        im = map_axs[k].imshow(med.T, cmap='viridis', norm=HNORM, extent=[-ext, ext, ext, -ext])
        map_axs[k].set_title(names[k]); map_axs[k].set_xlabel('x (µm)')
        for sp in map_axs[k].spines.values():
            sp.set_color(seq[k]); sp.set_linewidth(4)
        prof_axs[0].plot(o, med[:, by], color=seq[k], lw=3, label=names[k])
        prof_axs[1].plot(o, med[bx, :], color=seq[k], lw=3)
    map_axs[0].set_ylabel('y (µm)')
    fig.colorbar(im, ax=map_axs, shrink=0.8, label=HLBL)
    for ax, ttl in zip(prof_axs, ('profile along x', 'profile along y')):
        ax.set_ylim(-0.5, HMAX * 1.2); ax.set_xlabel('µm from cell centre'); ax.set_ylabel(HLBL)
        ax.set_title(ttl); _clean(ax)
    prof_axs[0].legend(frameon=False, title='bumps by height')
    fig.suptitle(f'{chip}: bump shape by bump height')
    _save(fig, f'{chip}_bump_by_height.png', chip)

    # ── which electrodes entered the analysis, why the others were excluded, and where each cut is
    status = t['status'].to_numpy()
    cmap = ListedColormap([c for _, c in STATUS])
    fig = plt.figure(figsize=(26, 10), layout='constrained')
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.6])
    ax = fig.add_subplot(gs[0])
    _cell_map(ax, status, 'electrodes in the analysis', cmap=cmap, vmin=-0.5, vmax=len(STATUS) - 0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec='0.4', lw=0.5) for _, c in STATUS]
    ax.legend(handles, [s_ for s_, _ in STATUS], loc='upper center', bbox_to_anchor=(0.5, -0.1),
              ncol=3, frameon=False, fontsize=13)
    right = gs[1].subgridspec(3, 2)
    ax = fig.add_subplot(right[0, 0])
    pct = np.array([(status == k).mean() * 100 for k in range(len(STATUS))])
    ax.barh(range(len(STATUS)), pct, color=[c for _, c in STATUS], edgecolor='0.4', lw=0.5)
    for k, v in enumerate(pct):
        ax.text(v + 2, k, f'{v:.0f} %', va='center', fontsize=12)
    ax.set_yticks(range(len(STATUS)), [s_ for s_, _ in STATUS]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel('% of electrodes'); ax.set_title('share of electrodes'); _clean(ax)
    # metric histograms (cells not on a seam); excluded side in the category colour
    cand = ~t['seam'].to_numpy()
    lo_h, hi_h = D['h_lim']
    # (panel, metric, status code, x label, keep range (lo, hi) — None = open, log y)
    hists = [(right[0, 1], 'corr', 5, 'grid fit: correlation', (EXCLUDE['fit_corr_min'], None), False),
             (right[1, 0], 'h_top_um', 4, 'bump top (µm)', (lo_h, hi_h), False),
             (right[1, 1], 'jitter_um', 2, 'pixel noise (µm)', (None, EXCLUDE['noise_max_um']), False),
             (right[2, 0], 'shape_rms_um', 3, 'shape deviation, RMS (µm)', (None, EXCLUDE['shape_rms_max_um']), False),
             (right[2, 1], 'defect_area_um2', 3, f'largest lump / dent > {EXCLUDE["lump_dev_um"]:g} µm (µm²)',
              (None, EXCLUDE['lump_area_um2']), True)]
    for spec, f, k, lbl, keep, logy in hists:
        if f not in t:
            continue
        ax = fig.add_subplot(spec)
        v = t.loc[cand, f].to_numpy(float)
        cuts = [c_ for c_ in keep if c_ is not None]
        lo_v = min([np.nanpercentile(v, 0.5)] + cuts); hi_v = max([np.nanpercentile(v, 99.5)] + [c_ * 1.15 for c_ in cuts])
        bins = np.linspace(lo_v, hi_v, 80)
        out = ((v < keep[0]) if keep[0] is not None else False) | ((v > keep[1]) if keep[1] is not None else False)
        ax.hist(np.clip(v[~out], lo_v, hi_v), bins=bins, color='0.7')
        ax.hist(np.clip(v[out], lo_v, hi_v), bins=bins, color=STATUS[k][1])
        for c_ in cuts:
            ax.axvline(c_, color='k', lw=1.5, ls='--')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel(lbl); ax.set_ylabel('electrodes')
        cut = ' or '.join(f'{op} {c_:.3g}' for op, c_ in zip(('<', '>'), keep) if c_ is not None)
        ax.set_title(f'{STATUS[k][0]}: {cut}', color=STATUS[k][1], fontweight='bold'); _clean(ax)
    fig.suptitle(f'{chip}: electrodes used')
    _save(fig, f'{chip}_cells_used.png', chip)

    # ── many small examples per category
    rows = [(0, 'included')] * 2 + [(k, s) for k, (s, _) in enumerate(STATUS) if k > 0]
    picks = {k: rng.permutation(np.where(status == k)[0]) for k in range(len(STATUS))}
    taken = {k: 0 for k in picks}
    fig, axs = plt.subplots(len(rows), N_EXAMPLES, figsize=(1.45 * N_EXAMPLES + 1.5, 1.55 * len(rows) + 0.6),
                            layout='constrained', squeeze=False)
    for r, (k, lbl) in enumerate(rows):
        for j, ax in enumerate(axs[r]):
            ax.set_xticks([]); ax.set_yticks([])
            if taken[k] >= len(picks[k]):
                for sp in ax.spines.values():
                    sp.set_visible(False)
                continue
            i = picks[k][taken[k]]; taken[k] += 1
            ax.imshow(H[i].T, cmap='viridis', norm=HNORM)
            for sp in ax.spines.values():
                sp.set_color(STATUS[k][1] if k != 1 else '0.6'); sp.set_linewidth(3)
        if r == 0 or rows[r - 1][0] != k:
            axs[r, 0].set_ylabel(lbl.replace(' / ', '\n/ '), fontsize=12)
    fig.colorbar(ScalarMappable(HNORM, 'viridis'), ax=axs, shrink=0.4, label=HLBL)
    fig.suptitle(f'{chip}: examples of included and excluded electrodes')
    _save(fig, f'{chip}_cells_examples.png', chip)


# ── 6. comparison between chips: height + bump shape ─────────────────────────
if DO_COMPARE and len(data) > 1:
    print("\n━━ chip comparison ━━")
    CHIP_COL = dict(zip(data, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']))   # fixed categorical order
    names = list(data)
    summ = pd.DataFrame({ch: {f: D['t'].loc[D['use'], f].median() for f in
                              ('h_core_um', 'h_top_um', 'gap_level_um', 'volume_um3', 'area_halfmax_um2', 'centroid_x_um',
                               'centroid_y_um', 'field_noise_um')}
                         | {'included_pct': D['use'].mean() * 100}
                         for ch, D in data.items()})
    print("median per chip (used cells):")
    print(summ.round(2).to_string())

    # height distributions
    fig, axs = plt.subplots(1, 5, figsize=(24, 4.2))
    for ax, (f, lbl, lo, hi) in zip(axs, [('h_core_um', 'active-area height above trench floor (µm)', 0, HMAX * 1.2),
                                          ('h_top_um', 'bump top above trench floor (µm)', 0, HMAX * 1.3),
                                          ('gap_level_um', 'cell edge level above trench floor (µm)', -0.5, HMAX),
                                          ('volume_um3', 'gold volume above trench floor (µm³)', 0, None),
                                          ('area_halfmax_um2', 'area above half height (µm²)', 0, None)]):
        allv = np.concatenate([D['t'].loc[D['use'], f] for D in data.values()])
        bins = np.linspace(lo, hi if hi else np.percentile(allv, 99.5), 70)
        for ch, D in data.items():
            v = D['t'].loc[D['use'], f]
            ax.hist(v, bins=bins, histtype='step', lw=2, density=True, color=CHIP_COL[ch], label=ch)
            ax.axvline(v.median(), color=CHIP_COL[ch], lw=1, ls='--')
        ax.set_xlabel(lbl); ax.set_ylabel('density'); ax.spines[['top', 'right']].set_visible(False)
    axs[0].legend(frameon=False)
    fig.suptitle('Chip comparison: height and gold amount (dashed = median; included electrodes)',
                 fontweight='bold')
    plt.tight_layout()
    _save(fig, 'compare_height.png', 'comparison')

    # bump shape: median cells, difference, profiles (absolute and normalised), bump position
    meds = {ch: D['p50'] for ch, D in data.items()}
    qs = {ch: np.percentile(D['H'][D['use']], [25, 75], axis=0) for ch, D in data.items()}
    nc = len(names)
    fig = plt.figure(figsize=(max(4.2 * (nc + 2), 18), 9.5), layout='constrained')
    top = fig.add_gridspec(2, nc + 2)
    med_axs = []
    for j, ch in enumerate(names):
        ax = fig.add_subplot(top[0, j]); med_axs.append(ax)
        im = ax.imshow(meds[ch].T, cmap='viridis', norm=HNORM, extent=[-ext, ext, ext, -ext])
        ax.set_title(f'{ch} median cell', fontsize=10); ax.set_xlabel('x (µm)')
        for sp in ax.spines.values():
            sp.set_color(CHIP_COL[ch]); sp.set_linewidth(3)
    med_axs[0].set_ylabel('y (µm)')
    fig.colorbar(im, ax=med_axs, shrink=0.8, label=HLBL)
    ax = fig.add_subplot(top[0, nc])
    diff = meds[names[-1]] - meds[names[0]]
    a_ = np.abs(diff).max()
    im = ax.imshow(diff.T, cmap='RdBu_r', vmin=-a_, vmax=a_, extent=[-ext, ext, ext, -ext])
    ax.set_title(f'{names[-1]} − {names[0]}', fontsize=10); ax.set_xlabel('x (µm)')
    fig.colorbar(im, ax=ax, shrink=0.8, label='height difference (µm)')
    ax = fig.add_subplot(top[0, nc + 1])
    for ch, D in data.items():
        tt = D['t'].loc[D['use']]
        sub = tt.sample(min(3000, len(tt)), random_state=SEED)
        ax.scatter(sub['centroid_x_um'], sub['centroid_y_um'], s=3, alpha=0.3, color=CHIP_COL[ch], label=ch,
                   rasterized=True)
        ax.plot(tt['centroid_x_um'].median(), tt['centroid_y_um'].median(), 'o', ms=9, mfc=CHIP_COL[ch],
                mec='white', mew=1.5)
    ax.axhline(0, color='0.7', lw=0.8); ax.axvline(0, color='0.7', lw=0.8)
    ax.set_xlim(-3, 3); ax.set_ylim(3, -3); ax.set_aspect('equal')
    ax.set_xlabel('bump centroid x (µm)'); ax.set_ylabel('bump centroid y (µm)')
    ax.set_title('bump position in the cell (dot = median)', fontsize=10)
    ax.legend(frameon=False, markerscale=4, fontsize=8)
    bot = top[1, :].subgridspec(1, 4)
    for j, (axis, lbl) in enumerate([(1, 'x'), (0, 'y')]):            # mean over the other axis
        ax_abs, ax_nrm = fig.add_subplot(bot[0, j]), fig.add_subplot(bot[0, 2 + j])
        for ch in names:
            m_ = meds[ch].mean(axis); lo_, hi_ = qs[ch][0].mean(axis), qs[ch][1].mean(axis)
            ax_abs.fill_between(o, lo_, hi_, color=CHIP_COL[ch], alpha=0.2, lw=0)
            ax_abs.plot(o, m_, color=CHIP_COL[ch], lw=2, label=f'{ch} median (band = IQR)')
            ax_nrm.plot(o, m_ / m_.max(), color=CHIP_COL[ch], lw=2, label=ch)
        ax_abs.set_ylim(-0.5, HMAX); ax_abs.set_ylabel(HLBL)
        ax_abs.set_title(f'profile along {lbl} (mean over {"y" if lbl == "x" else "x"})', fontsize=10)
        ax_abs.legend(frameon=False, fontsize=8)
        ax_nrm.set_ylim(-0.3, 1.1); ax_nrm.set_ylabel('height / own max')
        ax_nrm.set_title(f'shape only: profile along {lbl}, normalised', fontsize=10)
        ax_nrm.legend(frameon=False, fontsize=8)
        for ax in (ax_abs, ax_nrm):
            ax.set_xlabel('µm from cell centre'); ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Chip comparison: bump shape (included electrodes)', fontweight='bold')
    _save(fig, 'compare_shape.png', 'comparison')

    # smoothness: same pixel size required (jitter is per pixel)
    if SMOOTH:
        JIT = [('jitter_core_um', 'active area'), ('jitter_gap_um', 'gap ring')]
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
        bins = np.linspace(0, JMAX * 1.3, 70)
        for ax, (f, lbl) in zip(axs, JIT):
            for ch, D in data.items():
                v = D['t'].loc[D['use'], f]
                ax.hist(v, bins=bins, histtype='step', lw=2, density=True, color=CHIP_COL[ch], label=ch)
                ax.axvline(v.median(), color=CHIP_COL[ch], lw=1, ls='--')
            ax.axvspan(0, 0.03, color='0.9', zorder=0)
            ax.set_xlabel(f'{lbl}: equivalent pixel noise (µm)'); ax.set_ylabel('density')
            ax.spines[['top', 'right']].set_visible(False)
        axs[0].legend(frameon=False)
        fig.suptitle('Chip comparison: smoothness (dashed = median; grey = noise-free floor)', fontweight='bold')
        _save(fig, 'compare_smoothness.png', 'comparison')

out_csv = OUT_DIR / 'mea_cell_quality.csv'
cols = ['el', 'row', 'col', 'fit_failed', 'seam', 'height_outlier', 'field', 'field_noise_um', 'status_name', 'defect', 'defect_area_um2', 'shape_rms_um',
        'h_top_um', 'h_core_um', 'gap_level_um', 'volume_um3', 'area_halfmax_um2', 'centroid_x_um', 'centroid_y_um']
if SMOOTH:
    cols += ['jitter_um', 'jitter_core_um', 'jitter_gap_um']
pd.concat([D['t'][cols].assign(chip=chip) for chip, D in data.items()])[['chip'] + cols].to_csv(out_csv, index=False)
print(f"\n  → {out_csv.name}")
print("Done.")
