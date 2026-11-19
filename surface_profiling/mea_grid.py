"""
Fit a rigid 220×120 electrode lattice to an MEA1K profilometer scan and cut out
one aligned crop per electrode (el = row*220 + col, el 0 top-left as displayed
by process_profiles.py, i.e. smallest x and y).

Pipeline (all lengths in µm):
  1. rough array bbox  — where the axis-median profiles carry lattice modulation
  2. rigid lattice     — rotation + pitch per axis, maximising the folded gap depth
                         of rotated projections; gap phase from the folded profile
  3. array window      — exhaustive search over integer lattice offsets for the
                         220×120 window with highest per-cell template correlation
  4. recentring        — per-cell template match (≤ max_shift), replaced by the
                         median of confident neighbours so only the scan distortion
                         (stitching) is removed, not the bump's own offset
  5. crops + features  — one pitch×pitch crop per electrode + feature table
"""

import time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

N_COLS, N_ROWS = 220, 120
PITCH_UM       = 17.5


# ── helpers ───────────────────────────────────────────────────────────────────
def _longest_run(mask):
    """(start, stop) of the longest True run."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    starts, stops = np.where(d == 1)[0], np.where(d == -1)[0]
    i = np.argmax(stops - starts)
    return starts[i], stops[i]


def _project(Z, Xc, Yc, theta_deg, axis, bin_um):
    """Mean of Z binned along u (axis='x', normal of the column lines) or v (axis='y')."""
    t = np.radians(theta_deg)
    u = Xc * np.cos(t) + Yc * np.sin(t) if axis == 'x' else -Xc * np.sin(t) + Yc * np.cos(t)
    umin = u.min()
    b = ((u - umin) / bin_um).astype(np.int64).ravel()
    s, n = np.bincount(b, Z.ravel()), np.bincount(b)
    return s / np.maximum(n, 1), n, umin


def _fold(prof, cnt, umin, bin_um, pitch, nb=70):
    """Fold a projected profile with period `pitch` → mean profile over one period (phase 0..1)."""
    u  = umin + (np.arange(len(prof)) + 0.5) * bin_um
    ph = ((u / pitch) % 1 * nb).astype(int)
    return np.bincount(ph, prof * cnt, nb) / np.maximum(np.bincount(ph, cnt, nb), 1)


def _ncc(A, T):
    """Normalised cross-correlation of each crop in A (N,h,w) with template T (h,w)."""
    A = A - A.mean((1, 2), keepdims=True)
    T = T - T.mean()
    return (A * T).sum((1, 2)) / np.sqrt((A ** 2).sum((1, 2)) * (T ** 2).sum() + 1e-12)


# ── 1. rough array bbox ──────────────────────────────────────────────────────
def find_array_bbox(zz, dx, dy, pitch=PITCH_UM):
    """Rough array extent (x0, x1, y0, y1) in µm from the local amplitude of the
    column-lattice modulation (complex demodulation at 1/pitch, ~2-pitch tiles).
    Unlike full-axis mean profiles this is not smeared by a slight rotation."""
    x = (np.arange(zz.shape[0]) * dx)[:, None]
    c = (zz * np.exp(-2j * np.pi * x / pitch)).astype(np.complex64)
    s = (2 * pitch / dx, 2 * pitch / dy)
    amp  = np.hypot(gaussian_filter(c.real, s), gaussian_filter(c.imag, s))
    mask = amp > 0.4 * np.percentile(amp, 95)
    out = []
    for frac, d in ((mask.mean(1), dx), (mask.mean(0), dy)):
        a, b = _longest_run(frac > 0.5 * frac.max())
        out += [a * d, b * d]
    return tuple(out)


def _window_bbox(lat, i0, j0):
    """(x0, x1, y0, y1) µm spanned by the cell centres of the N_COLS×N_ROWS window."""
    xs, ys = lattice_xy(lat, [i0, i0 + N_COLS - 1, i0, i0 + N_COLS - 1],
                             [j0, j0, j0 + N_ROWS - 1, j0 + N_ROWS - 1])
    return xs.min(), xs.max(), ys.min(), ys.max()


# ── 2. rigid lattice ─────────────────────────────────────────────────────────
def fit_lattice(zz, dx, dy, bbox, pitch0=PITCH_UM, bin_um=0.1):
    """Fit rotation, pitch and gap phase separately for column lines (axis 'x') and
    row lines (axis 'y') inside bbox. Returns dict of lattice parameters."""
    x0, x1, y0, y1 = bbox
    m  = 2 * pitch0                                        # stay clear of the array edge
    ix = slice(int((x0 + m) / dx), int((x1 - m) / dx))
    iy = slice(int((y0 + m) / dy), int((y1 - m) / dy))
    Z  = zz[ix, iy]
    X  = np.arange(zz.shape[0])[ix][:, None] * dx
    Y  = np.arange(zz.shape[1])[iy][None, :] * dy
    xm, ym = float(X.mean()), float(Y.mean())
    Xc, Yc = X - xm, Y - ym

    lat = dict(origin=(xm, ym))
    for axis in ('x', 'y'):
        best = (-1, 0.0, pitch0)
        for thetas, pitches in [
                (np.arange(-0.5, 0.5001, 0.02), np.arange(pitch0 - 0.2, pitch0 + 0.2001, 0.005)),
                (None, None)]:
            if thetas is None:                              # fine search around coarse optimum
                thetas  = best[1] + np.arange(-0.03, 0.0301, 0.002)
                pitches = best[2] + np.arange(-0.01, 0.01001, 0.0005)
            for th in thetas:
                prof, cnt, umin = _project(Z, Xc, Yc, th, axis, bin_um)
                for P in pitches:
                    s = _fold(prof, cnt, umin, bin_um, P).std()
                    if s > best[0]:
                        best = (s, th, P)
        _, th, P = best
        prof, cnt, umin = _project(Z, Xc, Yc, th, axis, bin_um)
        nb = 70
        fp = _fold(prof, cnt, umin, bin_um, P, nb)
        k  = 2 * int(round(1.0 / P * nb)) + 1                # ~2 µm circular smoothing (odd: no shift)
        fps = np.convolve(np.r_[fp[-k:], fp, fp[:k]], np.ones(k) / k, 'same')[k:-k]
        gap = (np.argmin(fps) + 0.5) / nb * P               # gap line position (mod P)
        lat[axis] = dict(theta_deg=float(th), pitch=float(P), gap=float(gap),
                         centre=float((gap + P / 2) % P), depth=float(np.ptp(fp)),
                         folded=fp)
        print(f"    {axis}-lattice: angle {th:+.3f}°, pitch {P:.4f} µm, "
              f"gap depth {np.ptp(fp):.3f} µm, gap at {gap:.2f} µm (mod pitch)")
    print(f"    rotation {(lat['x']['theta_deg'] + lat['y']['theta_deg']) / 2:+.3f}°, "
          f"shear {lat['x']['theta_deg'] - lat['y']['theta_deg']:+.3f}°")
    return lat


def lattice_xy(lat, i, j):
    """Cell centre (x, y) µm for lattice indices i (column), j (row)."""
    tx, ty = np.radians(lat['x']['theta_deg']), np.radians(lat['y']['theta_deg'])
    M  = np.array([[np.cos(tx), np.sin(tx)], [-np.sin(ty), np.cos(ty)]])
    Mi = np.linalg.inv(M)
    u  = lat['x']['centre'] + np.asarray(i, float) * lat['x']['pitch']
    v  = lat['y']['centre'] + np.asarray(j, float) * lat['y']['pitch']
    x  = Mi[0, 0] * u + Mi[0, 1] * v + lat['origin'][0]
    y  = Mi[1, 0] * u + Mi[1, 1] * v + lat['origin'][1]
    return x, y


def _cell_axes(lat):
    """Unit vectors along a column step (+i) and a row step (+j) in (x, y)."""
    x0, y0 = lattice_xy(lat, 0, 0)
    xi, yi = lattice_xy(lat, 1, 0)
    xj, yj = lattice_xy(lat, 0, 1)
    a = np.array([xi - x0, yi - y0]); b = np.array([xj - x0, yj - y0])
    return a / np.linalg.norm(a), b / np.linalg.norm(b)


def sample_cells(img, dx, dy, lat, cx, cy, n, chunk=4000, order=1):
    """Bilinear crops (N, n, n) of img centred on (cx, cy), sampled at spacing dx in the
    lattice frame. Crop index [s, t] follows the z[ix, iy] convention (s along columns).
    order=0: nearest pixel instead (keeps native pixel values, e.g. for noise statistics)."""
    ea, eb = _cell_axes(lat)
    o = (np.arange(n) - (n - 1) / 2) * dx
    S, T = np.meshgrid(o, o, indexing='ij')
    px = S * ea[0] + T * eb[0]
    py = S * ea[1] + T * eb[1]
    cx, cy = np.ravel(cx), np.ravel(cy)
    out = np.empty((len(cx), n, n), np.float32)
    for a in range(0, len(cx), chunk):
        X = (cx[a:a + chunk, None, None] + px) / dx
        Y = (cy[a:a + chunk, None, None] + py) / dy
        out[a:a + chunk] = map_coordinates(img, [X, Y], order=order, mode='constant', cval=np.nan)
    return out


# ── 3. array window ──────────────────────────────────────────────────────────
def locate_array(zz, dx, dy, lat, bbox, n):
    """Score every lattice cell in the scan against the median cell, then pick the
    N_COLS×N_ROWS window with the highest summed score by exhaustive search."""
    nx, ny = zz.shape
    # lattice index range covering the scan
    corners = [(0, 0), (nx * dx, 0), (0, ny * dy), (nx * dx, ny * dy)]
    tx, ty = np.radians(lat['x']['theta_deg']), np.radians(lat['y']['theta_deg'])
    us = [(x - lat['origin'][0]) * np.cos(tx) + (y - lat['origin'][1]) * np.sin(tx) for x, y in corners]
    vs = [-(x - lat['origin'][0]) * np.sin(ty) + (y - lat['origin'][1]) * np.cos(ty) for x, y in corners]
    i_rng = np.arange(int(np.floor((min(us) - lat['x']['centre']) / lat['x']['pitch'])),
                      int(np.ceil((max(us) - lat['x']['centre']) / lat['x']['pitch'])) + 1)
    j_rng = np.arange(int(np.floor((min(vs) - lat['y']['centre']) / lat['y']['pitch'])),
                      int(np.ceil((max(vs) - lat['y']['centre']) / lat['y']['pitch'])) + 1)
    I, J = np.meshgrid(i_rng, j_rng, indexing='ij')
    cx, cy = lattice_xy(lat, I, J)
    crops = sample_cells(zz, dx, dy, lat, cx, cy, n)
    valid = ~np.isnan(crops).any((1, 2))

    x0, x1, y0, y1 = bbox
    m = 3 * PITCH_UM
    inner = valid & (cx.ravel() > x0 + m) & (cx.ravel() < x1 - m) & (cy.ravel() > y0 + m) & (cy.ravel() < y1 - m)
    tmpl = np.median(crops[inner], 0)
    C = np.full(len(crops), -1.0)                          # cells outside the scan count as -1
    C[valid] = _ncc(crops[valid], tmpl)
    C = C.reshape(I.shape)

    if C.shape[0] < N_COLS or C.shape[1] < N_ROWS:
        raise ValueError(f"scan only spans {C.shape} lattice cells, need {N_COLS}×{N_ROWS}")
    cs = np.pad(C.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    W  = (cs[N_COLS:, N_ROWS:] - cs[:-N_COLS, N_ROWS:] - cs[N_COLS:, :-N_ROWS] + cs[:-N_COLS, :-N_ROWS])
    a, b = np.unravel_index(np.argmax(W), W.shape)
    i0, j0 = int(i_rng[a]), int(j_rng[b])

    # margin: mean score of the edge line kept vs the one gained when shifting by ±1
    win = C[a:a + N_COLS, b:b + N_ROWS]
    def _line(arr):
        return float(np.mean(arr)) if arr.size else np.nan
    edges = {
        'left':   (_line(win[0]),     _line(C[a - 1, b:b + N_ROWS]) if a > 0 else np.nan),
        'right':  (_line(win[-1]),    _line(C[a + N_COLS, b:b + N_ROWS]) if a + N_COLS < C.shape[0] else np.nan),
        'top':    (_line(win[:, 0]),  _line(C[a:a + N_COLS, b - 1]) if b > 0 else np.nan),
        'bottom': (_line(win[:, -1]), _line(C[a:a + N_COLS, b + N_ROWS]) if b + N_ROWS < C.shape[1] else np.nan),
    }
    print(f"    array window: lattice col {i0}…{i0 + N_COLS - 1}, row {j0}…{j0 + N_ROWS - 1}")
    for k, (inside, outside) in edges.items():
        warn = '' if not (outside > 0.5 * inside) else '   <-- WEAK EDGE, check QC corners'
        print(f"      {k:6s} edge: mean corr inside {inside:.2f}  vs next line outside {outside:.2f}{warn}")
    # cross-check against the envelope bbox
    ex0, ey0 = lattice_xy(lat, i0 - 0.5, j0 - 0.5)
    ex1, ey1 = lattice_xy(lat, i0 + N_COLS - 0.5, j0 + N_ROWS - 0.5)
    dev = np.abs(np.array([ex0 - x0, ex1 - x1, ey0 - y0, ey1 - y1])) / PITCH_UM
    print(f"    window vs envelope bbox: edge deviation {np.round(dev, 1)} pitches")
    if dev.max() > 3:
        print("    WARNING: lattice window and envelope bbox disagree by > 3 pitches")

    return dict(i0=i0, j0=j0, tmpl=tmpl, score=C, i_rng=i_rng, j_rng=j_rng, edges=edges)


# ── 4. recentring ────────────────────────────────────────────────────────────
def recentre(zz, dx, dy, lat, cx, cy, pitch, max_shift_um=3.0, min_contrast=0.5, nbh=5,
             step_um=0.1):
    """Shift each cell (≤ max_shift_um per axis) so its two edges sit on the lowest heights
    of the profile across the cell (gaps). The applied shift is the median of the
    confident shifts in an nbh×nbh neighbourhood, i.e. only the scan distortion is removed;
    noisy cells inherit it from their neighbours."""
    n = int(round(pitch / dx))
    m = int(np.ceil((max_shift_um + 2) / dx))
    big = sample_cells(zz, dx, dy, lat, cx, cy, n + 2 * m)
    L = big.shape[1]
    o = (np.arange(L) - (L - 1) / 2) * dx
    inner = np.abs(o) < pitch / 2 - 1.5                     # stay off the gaps of the other axis
    shifts = np.arange(-max_shift_um, max_shift_um + step_um / 2, step_um)

    sh = np.zeros((len(big), 2)); contrast = np.zeros((len(big), 2)); at_border = np.zeros((len(big), 2), bool)
    for ax in (0, 1):
        prof = np.nanmedian(big[:, :, inner] if ax == 0 else big[:, inner, :], axis=2 - ax)   # (N, L)
        prof = gaussian_filter1d(prof, 0.75 / dx, axis=1)    # flat gap bottom → minimum at its middle
        cost = np.empty((len(big), len(shifts)))
        for k, s_ in enumerate(shifts):                     # height at both edges, linear interp
            e = (np.array([-pitch / 2, pitch / 2]) + s_ - o[0]) / dx
            i0 = np.floor(e).astype(int); f = e - i0
            cost[:, k] = ((1 - f) * prof[:, i0] + f * prof[:, i0 + 1]).sum(1) / 2
        kb = np.argmin(cost, 1)
        sh[:, ax] = shifts[kb]
        contrast[:, ax] = np.median(prof[:, np.abs(o) < pitch / 2], 1) - cost[np.arange(len(big)), kb]
        at_border[:, ax] = (kb == 0) | (kb == len(shifts) - 1)
    conf = (contrast >= min_contrast * np.median(contrast, 0)) & ~at_border   # per axis

    S = sh.reshape(N_COLS, N_ROWS, 2).copy()
    S[~conf.reshape(N_COLS, N_ROWS, 2)] = np.nan
    pad = nbh // 2
    Sp = np.pad(S, ((pad, pad), (pad, pad), (0, 0)), constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(Sp, (nbh, nbh), axis=(0, 1))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        applied = np.nanmedian(win.reshape(N_COLS, N_ROWS, 2, -1), -1)
    applied = np.nan_to_num(applied).reshape(-1, 2)

    ea, eb = _cell_axes(lat)
    ncx = cx + applied[:, 0] * ea[0] + applied[:, 1] * eb[0]
    ncy = cy + applied[:, 0] * ea[1] + applied[:, 1] * eb[1]
    print(f"    recentring on gaps: confident x {conf[:, 0].mean() * 100:.1f}%, y {conf[:, 1].mean() * 100:.1f}%; "
          f"applied |shift| median {np.median(np.abs(applied), 0).round(2)} µm, "
          f"99th pct {np.percentile(np.abs(applied), 99, axis=0).round(2)} µm (x, y)")
    return ncx, ncy, applied, sh, conf.all(1)


# ── 5. features ──────────────────────────────────────────────────────────────
def cell_features(crops, dx, active_um=(9.0, 5.0), ring_um=1.5):
    """Per-cell height features from crops (N, n, n) of the processed height z.
    active_um: (along columns/x, along rows/y) size of the central active area."""
    N, n, _ = crops.shape
    o = (np.arange(n) - (n - 1) / 2) * dx
    S, T = np.meshgrid(o, o, indexing='ij')
    half = n * dx / 2
    ring = (np.abs(S) > half - ring_um) | (np.abs(T) > half - ring_um)
    z_gap  = np.median(crops[:, ring], 1)
    rel    = crops - z_gap[:, None, None]
    pos    = np.clip(rel, 0, None)

    # the bump is not centred in the cell (gap-to-gap); centre the active area on the
    # bump of the median cell instead
    med  = np.clip(np.nanmedian(rel, 0), 0, None)
    med  = np.where(med > 0.5 * med.max(), med, 0)
    bx, by = (med * S).sum() / med.sum(), (med * T).sum() / med.sum()
    core = (np.abs(S - bx) <= active_um[0] / 2) & (np.abs(T - by) <= active_um[1] / 2)
    print(f"    active area {active_um[0]}×{active_um[1]} µm centred on median bump at "
          f"({bx:+.2f}, {by:+.2f}) µm from cell centre")
    h_top  = np.percentile(rel.reshape(N, -1), 95, axis=1)
    mass   = pos.sum((1, 2))
    df = pd.DataFrame(dict(
        z_gap_um       = z_gap,
        z_core_um      = np.median(crops[:, core], 1),
        h_core_um      = np.median(rel[:, core], 1),
        h_top_um       = h_top,
        volume_um3     = mass * dx * dx,
        area_halfmax_um2 = (rel > 0.5 * h_top[:, None, None]).sum((1, 2)) * dx * dx,
        centroid_x_um  = (pos * S).sum((1, 2)) / np.maximum(mass, 1e-12),
        centroid_y_um  = (pos * T).sum((1, 2)) / np.maximum(mass, 1e-12),
    ))
    return df, (bx, by)


def cell_smoothness(z, dx, dy, lat, cx, cy, n, bump_xy, active_um=(9.0, 5.0), ring_um=1.5):
    """Pixel-to-pixel smoothness per cell, on the native pixel grid (no resampling).
    Second difference to the two neighbours, d2 = z[i-1] - 2 z[i] + z[i+1], along x and y:
    a plane or a gently curved bump gives d2 ≈ 0, measurement noise does not. For white
    noise of RMS σ, median|d2| = 0.6745·√6·σ, so jitter = median|d2| / (0.6745·√6) reads as
    the equivalent pixel noise in µm. Medians keep dust spikes and the steep trench walls
    (a minority of pixels) from dominating. Depends on the pixel size: compare scans of
    the same resolution only.
    Returns jitter over the whole cell, the active area and the gap ring (µm)."""
    k = 1 / (0.6745 * np.sqrt(6))
    d2 = np.full((2,) + z.shape, np.nan, np.float32)
    d2[0, 1:-1, :] = np.abs(z[:-2, :] - 2 * z[1:-1, :] + z[2:, :])
    d2[1, :, 1:-1] = np.abs(z[:, :-2] - 2 * z[:, 1:-1] + z[:, 2:])
    A = np.concatenate([sample_cells(d, dx, dy, lat, cx, cy, n, order=0) for d in d2], axis=2)   # (N, n, 2n)
    o = (np.arange(n) - (n - 1) / 2) * dx
    S, T = np.meshgrid(o, o, indexing='ij')
    half = n * dx / 2
    core = (np.abs(S - bump_xy[0]) <= active_um[0] / 2) & (np.abs(T - bump_xy[1]) <= active_um[1] / 2)
    ring = (np.abs(S) > half - ring_um) | (np.abs(T) > half - ring_um)
    core, ring = np.tile(core, 2), np.tile(ring, 2)
    return (k * np.nanmedian(A.reshape(len(A), -1), 1),
            k * np.nanmedian(A[:, core], 1),
            k * np.nanmedian(A[:, ring], 1))


# ── outlines for plotting ────────────────────────────────────────────────────
def cell_outlines(table, lat):
    """Closed polygons (N, 5, 2) in µm of each electrode cell as cropped (rotated square
    of one pitch around the recentred cell centre), e.g. for a LineCollection."""
    P = (lat['x']['pitch'] + lat['y']['pitch']) / 2
    ea, eb = _cell_axes(lat)
    sq = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5], [-.5, -.5]]) * P @ np.array([ea, eb])
    return table[['x_um', 'y_um']].to_numpy()[:, None, :] + sq[None]


# ── driver ───────────────────────────────────────────────────────────────────
def run_mea_grid(data, out, stem, max_shift_um=3.0, min_contrast=0.5,
                 active_um=(9.0, 5.0), save_crops=True, z_ref_offset_um=0.0):
    """Full pipeline on a data dict from read_vk (after rotation / releveling).
    Writes {stem}_mea_cells.csv (+ .npz crops) to `out`. Use cell_outlines(table, lat)
    to draw the cells on a heatmap.
    z_ref_offset_um: offset subtracted from data['z'] after releveling (zero_min, vmin);
    it is added back so all exported heights are in the releveled frame (relevel plane = 0)."""
    t0 = time.time()
    def step(msg):
        print(f"  [{time.time() - t0:5.0f}s] {msg}", flush=True)
    m  = data['meta']
    dx, dy = m['dx_um'], m['dy_um']
    z  = data['z'].astype(np.float64) + z_ref_offset_um
    if PITCH_UM / dx < 12:
        print(f"  WARNING: only {PITCH_UM / dx:.1f} px per pitch — lattice and shapes will be unreliable")

    zz = z - gaussian_filter(z, 40 / dx)                    # keep only cell-scale structure
    lo, hi = np.percentile(zz, [1, 99])
    zz = np.clip(zz, lo, hi)                                 # dust / spikes

    step("[1] envelope bbox ...")
    bbox = find_array_bbox(zz, dx, dy)
    print(f"    x {bbox[0]:.0f}…{bbox[1]:.0f} µm, y {bbox[2]:.0f}…{bbox[3]:.0f} µm  "
          f"(≈ {(bbox[1] - bbox[0]) / PITCH_UM:.0f} × {(bbox[3] - bbox[2]) / PITCH_UM:.0f} pitches)")
    step("[2] rigid lattice fit ...")
    lat = fit_lattice(zz, dx, dy, bbox)
    n = int(round((lat['x']['pitch'] + lat['y']['pitch']) / 2 / dx))
    step("[3] array window ...")
    res = locate_array(zz, dx, dy, lat, bbox, n)

    # second pass: refit the lattice on exactly the found array, then re-locate the window
    x0, x1, y0, y1 = _window_bbox(lat, res['i0'], res['j0'])
    bbox = (x0 - PITCH_UM / 2, x1 + PITCH_UM / 2, y0 - PITCH_UM / 2, y1 + PITCH_UM / 2)
    step("[2b] lattice refit on the 220×120 window ...")
    lat = fit_lattice(zz, dx, dy, bbox)
    step("[3b] array window ...")
    res = locate_array(zz, dx, dy, lat, bbox, n)

    col, row = np.meshgrid(np.arange(N_COLS), np.arange(N_ROWS), indexing='ij')   # (col, row) order
    cx, cy = lattice_xy(lat, res['i0'] + col.ravel(), res['j0'] + row.ravel())
    step("[4] recentring ...")
    P = (lat['x']['pitch'] + lat['y']['pitch']) / 2
    cx, cy, applied, raw, conf = recentre(zz, dx, dy, lat, cx, cy, P,
                                          max_shift_um=max_shift_um, min_contrast=min_contrast)

    step("[5] crops + features ...")
    crops_hp = sample_cells(zz, dx, dy, lat, cx, cy, n)
    tmpl     = np.nanmedian(crops_hp, 0)
    corr     = _ncc(np.nan_to_num(crops_hp), tmpl)
    crops    = sample_cells(z, dx, dy, lat, cx, cy, n)

    # reorder (col, row) → electrode order el = row*N_COLS + col
    order = (row.ravel() * N_COLS + col.ravel()).argsort()
    crops, crops_hp = crops[order], crops_hp[order]
    table = pd.DataFrame(dict(el=np.arange(N_COLS * N_ROWS), row=row.ravel()[order], col=col.ravel()[order],
                              x_um=cx[order], y_um=cy[order],
                              shift_x_um=applied[order, 0], shift_y_um=applied[order, 1],
                              own_shift_x_um=raw[order, 0], own_shift_y_um=raw[order, 1],
                              shift_confident=conf[order], corr=corr[order]))
    feats, bump_xy = cell_features(crops, dx, active_um)
    table = pd.concat([table, feats], axis=1)
    step("[6] smoothness ...")
    jit = cell_smoothness(z, dx, dy, lat, cx[order], cy[order], n, bump_xy, active_um)
    table['jitter_um'], table['jitter_core_um'], table['jitter_gap_um'] = jit
    print(f"    smoothness (equivalent pixel noise): median {np.median(jit[0]):.3f} µm "
          f"(active area {np.median(jit[1]):.3f}, gap {np.median(jit[2]):.3f})")
    h = table['h_top_um']
    q1, q3 = np.percentile(h, [25, 75])
    table['flag'] = (table['corr'] < 0.5) | (h > q3 + 3 * (q3 - q1)) | (h < q1 - 3 * (q3 - q1))
    print(f"    {len(table)} cells, corr median {table['corr'].median():.2f}, "
          f"flagged {table['flag'].sum()} ({table['flag'].mean() * 100:.1f}%)")

    step("[7] writing outputs ...")
    table.to_csv(out / f'{stem}_mea_cells.csv', index=False)
    print(f"    → {stem}_mea_cells.csv")
    if save_crops:
        np.savez_compressed(out / f'{stem}_mea_cells.npz', crops=crops, crops_hp=crops_hp,
                            dx_um=dx, el=table['el'].to_numpy(), z_frame='releveled',
                            z_ref_offset_um=z_ref_offset_um,
                            lattice=np.array([lat['x']['theta_deg'], lat['x']['pitch'], lat['x']['centre'],
                                              lat['y']['theta_deg'], lat['y']['pitch'], lat['y']['centre'],
                                              *lat['origin'], res['i0'], res['j0']]))
        print(f"    → {stem}_mea_cells.npz  (crops {crops.shape}, z and high-passed z)")
    step("done")
    return table, crops, lat
