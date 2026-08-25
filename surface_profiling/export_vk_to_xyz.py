"""
Export Keyence VK4/VK6 profilometer files to XYZ coordinate arrays.

Output units are micrometers (µm). Each file produces a .npz (numpy) and a .csv.

Usage:
    python export_vk_to_xyz.py                               # exports all vk4/vk6 in Alexei_one/
    python export_vk_to_xyz.py /path/to/data/dir             # scan a different directory
    python export_vk_to_xyz.py path/to/file.vk4             # single file
    python export_vk_to_xyz.py --relevel path/to/file.vk4   # with plane leveling
"""

import math
import sys
import zipfile
import numpy as np
from pathlib import Path
from SurfaceTopography.IO import VKReader

PM_TO_UM = 1e-6  # picometers → micrometers

# Defaults — paths relative to the project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Alexei_one"
DEFAULT_OUT_DIR  = PROJECT_ROOT / "output"


def read_vk(path: Path) -> dict:
    """Read a VK4/VK6 file and return x, y, z arrays (all in µm) plus metadata."""
    reader = VKReader(str(path))
    topo = reader.topography(channel_index=0)

    nx, ny = topo.nb_grid_pts
    sx, sy = topo.physical_sizes  # in pm

    sx_um = sx * PM_TO_UM
    sy_um = sy * PM_TO_UM

    x = np.linspace(0, sx_um, nx, endpoint=False)
    y = np.linspace(0, sy_um, ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')  # shape (nx, ny)

    z = topo.heights() * PM_TO_UM  # pm → µm

    meta = {
        'file': str(path),
        'nx': nx, 'ny': ny,
        'dx_um': sx_um / nx,
        'dy_um': sy_um / ny,
        'x_range_um': sx_um,
        'y_range_um': sy_um,
        'z_min_um': float(np.nanmin(z)),
        'z_max_um': float(np.nanmax(z)),
        'z_range_um': float(np.nanmax(z) - np.nanmin(z)),
        'unit': 'um',
    }

    return {'x': xx, 'y': yy, 'z': z, 'meta': meta}


def read_optical_from_cag(vk_path: Path, nx: int, ny: int) -> np.ndarray:
    """Extract the laser reflectance channel from a companion .cag file.

    The .cag is a KPK0+ZIP container.  The optical data is stored as a flat
    uint32 array of exactly nx*ny*4 bytes — same grid as the height data.
    Returns a float32 array of shape (nx, ny) normalised to [0, 1], or None
    if no .cag exists or no matching blob is found.
    """
    cag_path = vk_path.with_suffix('.cag')
    if not cag_path.exists():
        return None

    target = nx * ny * 4  # bytes
    try:
        with zipfile.ZipFile(cag_path) as z:
            for name in z.namelist():
                if z.getinfo(name).file_size == target:
                    raw = z.read(name)
                    # Native blob layout is (ny, nx) row-major; reshape accordingly then
                    # transpose to (nx, ny) to match heights() coordinate convention.
                    raw8 = np.frombuffer(raw, dtype=np.uint8).reshape((ny, nx, 4))
                    b_max = [int(raw8[:, :, c].max()) for c in range(4)]
                    b_mean = [round(float(raw8[:, :, c].mean()), 1) for c in range(4)]
                    print(f"    Optical bytes: max={b_max}  mean={b_mean}")
                    img = (raw8[:, :, 0].astype(np.float32)
                           + raw8[:, :, 1].astype(np.float32) * 256.0
                           + raw8[:, :, 2].astype(np.float32) * 65536.0).T  # → (nx, ny)
                    p1, p99 = float(np.percentile(img, 1)), float(np.percentile(img, 99))
                    img = np.clip((img - p1) / max(p99 - p1, 1.0), 0.0, 1.0)
                    print(f"    Optical: {name.split('/')[-1][:8]}…  "
                          f"raw range [{p1:.0f}, {p99:.0f}]  →  normalised [0, 1]")
                    return img
    except zipfile.BadZipFile:
        pass
    return None


def export_xyz(data: dict, stem: str, out_dir: Path):
    """Write compressed NPZ for a data dict returned by read_vk()."""
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, x=data['x'], y=data['y'], z=data['z'])
    print(f"    → {npz_path.name}  (npz)")


def relevel_plane_points(data: dict, points_um: list) -> dict:
    """Fit and subtract a plane through N reference points (≥3) assumed to be at equal height.

    points_um: list of (x, y) tuples in µm — z is sampled from the nearest grid cell.
    """
    x, y, z = data['x'], data['y'], data['z']
    m = data['meta']

    pts_xyz = []
    for px, py in points_um:
        ix = int(np.clip(round(px / m['dx_um']), 0, m['nx'] - 1))
        iy = int(np.clip(round(py / m['dy_um']), 0, m['ny'] - 1))
        pz = float(z[ix, iy])
        if np.isfinite(pz):
            pts_xyz.append((float(x[ix, iy]), float(y[ix, iy]), pz))
            print(f"    ({px:.1f}, {py:.1f}) µm  →  grid [{ix},{iy}]  z = {pz:.4f} µm")
        else:
            print(f"    ({px:.1f}, {py:.1f}) µm  →  NaN — skipped")

    if len(pts_xyz) < 3:
        raise ValueError(f"Need ≥3 valid reference points, got {len(pts_xyz)}")

    pts = np.array(pts_xyz)
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    coeffs, _, _, _ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    a, b, c = coeffs

    print(f"    Plane ({len(pts)} pts):  "
          f"x-slope {a*1e3:.4f} nm/µm ({math.degrees(math.atan(a)):.4f}°)"
          f",  y-slope {b*1e3:.4f} nm/µm ({math.degrees(math.atan(b)):.4f}°)"
          f",  offset {c:.4f} µm")
    z_pred = a * pts[:, 0] + b * pts[:, 1] + c
    residuals = pts[:, 2] - z_pred
    print(f"    Residuals: max |δz| {np.abs(residuals).max():.4f} µm  "
          f"rms {np.sqrt((residuals**2).mean()):.4f} µm")

    z_new = z - (a * x + b * y + c)
    meta = dict(data['meta'])
    meta['z_min_um']      = float(np.nanmin(z_new))
    meta['z_max_um']      = float(np.nanmax(z_new))
    meta['z_range_um']    = float(np.nanmax(z_new) - np.nanmin(z_new))
    meta['relevel']       = f'points({len(pts)}) a={a:.6e} b={b:.6e} c={c:.6e} (µm/µm)'
    meta['relevel_ref_points'] = [(p[0], p[1]) for p in pts_xyz]  # (x_um, y_um)

    return {'x': x, 'y': y, 'z': z_new, 'meta': meta}


def relevel_plane(data: dict) -> dict:
    """Fit and subtract a best-fit plane (z = a·x + b·y + c) to remove sample tilt."""
    x = data['x'].ravel()
    y = data['y'].ravel()
    z = data['z'].ravel()

    valid = np.isfinite(z)
    A = np.column_stack([x[valid], y[valid], np.ones(valid.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(A, z[valid], rcond=None)
    a, b, c = coeffs

    print(f"    Plane fit:  x-slope {a*1e3:.4f} nm/µm ({math.degrees(math.atan(a)):.4f}°)"
          f",  y-slope {b*1e3:.4f} nm/µm ({math.degrees(math.atan(b)):.4f}°)"
          f",  offset {c:.4f} µm")

    z_new = data['z'] - (a * data['x'] + b * data['y'] + c)

    meta = dict(data['meta'])
    meta['z_min_um']   = float(np.nanmin(z_new))
    meta['z_max_um']   = float(np.nanmax(z_new))
    meta['z_range_um'] = float(np.nanmax(z_new) - np.nanmin(z_new))
    meta['relevel']    = f'plane a={a:.6e} b={b:.6e} c={c:.6e} (µm/µm)'

    return {'x': data['x'], 'y': data['y'], 'z': z_new, 'meta': meta}


def _resolve_targets(args):
    relevel = '--relevel' in args
    args = [a for a in args if a != '--relevel']

    if len(args) == 0:
        data_dir = DEFAULT_DATA_DIR
        return sorted(data_dir.rglob("*.vk4")) + sorted(data_dir.rglob("*.vk6")), DEFAULT_OUT_DIR, relevel
    arg = Path(args[0])
    if arg.is_dir():
        return sorted(arg.rglob("*.vk4")) + sorted(arg.rglob("*.vk6")), DEFAULT_OUT_DIR, relevel
    # One or more explicit files
    return [Path(p) for p in args], DEFAULT_OUT_DIR, relevel


def main():
    targets, out_dir, relevel = _resolve_targets(sys.argv[1:])

    if not targets:
        print("No .vk4 or .vk6 files found.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {len(targets)} file(s) → {out_dir}/  {'[plane leveling ON]' if relevel else ''}\n")

    for p in targets:
        print(f"  Reading {p.name} ...", end=' ', flush=True)
        try:
            data = read_vk(p)
            m = data['meta']
            print(f"grid {m['nx']}x{m['ny']}, z range {m['z_range_um']:.3f} µm")
            if relevel:
                data = relevel_plane(data)
            export_xyz(data, p.stem, out_dir)
        except Exception as e:
            print(f"\n  ERROR on {p.name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
