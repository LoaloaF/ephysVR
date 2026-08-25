"""
Export Keyence VK4/VK6 files as binary STL surface meshes.

Each STL is a closed solid: top surface + flat bottom + 4 side walls.
Coordinates in µm. Gaussian smoothing applied before meshing.

Usage:
    python export_stl.py                      # all vk4/vk6 in Alexei_one/
    python export_stl.py /path/to/data/dir    # scan a different directory
    python export_stl.py path/to/file.vk4     # single file
"""

import sys
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, distance_transform_edt

from export_vk_to_xyz import read_vk, relevel_plane, DEFAULT_DATA_DIR, DEFAULT_OUT_DIR

SIGMA   = 0.1   # px — light smoothing to remove pixel staircasing
Z_SCALE = 1.0   # height exaggeration


def export_stl(data: dict, stem: str, out_dir: Path,
               sigma: float = SIGMA, z_scale: float = Z_SCALE):
    """Write a binary STL for a data dict from read_vk()."""
    z = data['z'].astype(np.float32)
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32)

    # Fill NaN pixels before smoothing
    nan_mask = np.isnan(z)
    if nan_mask.any():
        _, nn_idx = distance_transform_edt(nan_mask, return_indices=True)
        z[nan_mask] = z[tuple(nn_idx[:, nan_mask])]
        print(f"    Filled {nan_mask.sum():,} NaN pixels")

    z = gaussian_filter(z, sigma=sigma) * z_scale
    nx, ny = z.shape

    z_bottom = float(z.min()) - 1.0
    print(f"    Grid {nx}×{ny} | Z {z.min():.3f}…{z.max():.3f} µm (×{z_scale})")

    # ── build vertex grid using actual µm coordinates ─────────────────────────
    V = np.stack([x, y, z], axis=-1)  # (nx, ny, 3)

    tri_list = []

    # Top surface
    i_idx = np.arange(nx - 1)
    j_idx = np.arange(ny - 1)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')

    v00 = V[ii,     jj    ].reshape(-1, 3)
    v10 = V[ii,     jj + 1].reshape(-1, 3)
    v01 = V[ii + 1, jj    ].reshape(-1, 3)
    v11 = V[ii + 1, jj + 1].reshape(-1, 3)

    tri_list.append(np.stack([v00, v10, v11], axis=1))
    tri_list.append(np.stack([v00, v11, v01], axis=1))

    # Flat bottom
    B = np.stack([x, y, np.full_like(z, z_bottom)], axis=-1)
    b00 = B[ii,     jj    ].reshape(-1, 3)
    b10 = B[ii,     jj + 1].reshape(-1, 3)
    b01 = B[ii + 1, jj    ].reshape(-1, 3)
    b11 = B[ii + 1, jj + 1].reshape(-1, 3)

    tri_list.append(np.stack([b00, b11, b10], axis=1))
    tri_list.append(np.stack([b00, b01, b11], axis=1))

    # Four side walls
    def wall_strip(top_edge, bot_z, flip=False):
        tris = []
        for k in range(len(top_edge) - 1):
            t0 = top_edge[k]; t1 = top_edge[k + 1]
            b0 = t0.copy();   b0[2] = bot_z
            b1 = t1.copy();   b1[2] = bot_z
            tris.append([t0, b0, t1])
            tris.append([t1, b0, b1])
        arr = np.array(tris, dtype=np.float32)
        return arr[:, ::-1] if flip else arr

    tri_list.append(wall_strip(V[0,   :, :], z_bottom, flip=True))
    tri_list.append(wall_strip(V[-1,  :, :], z_bottom))
    tri_list.append(wall_strip(V[:,  0, :], z_bottom))
    tri_list.append(wall_strip(V[:, -1, :], z_bottom, flip=True))

    # ── assemble & compute normals ────────────────────────────────────────────
    all_tris = np.concatenate(tri_list, axis=0)  # (T, 3, 3)
    T = len(all_tris)

    e1 = all_tris[:, 1] - all_tris[:, 0]
    e2 = all_tris[:, 2] - all_tris[:, 0]
    normals = np.cross(e1, e2).astype(np.float32)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(nlen == 0, 1, nlen)

    # ── write binary STL ──────────────────────────────────────────────────────
    stl_path = out_dir / f"{stem}.stl"
    msg = f"STL height map {stem}".encode()
    header = msg + b"\x00" * (80 - len(msg))

    with open(stl_path, "wb") as f:
        f.write(header)
        f.write(np.uint32(T).tobytes())
        for k in range(T):
            f.write(normals[k].tobytes())
            for v in range(3):
                f.write(all_tris[k, v].astype(np.float32).tobytes())
            f.write(b"\x00\x00")

    size_mb = (T * 50 + 84) / 1e6
    print(f"    → {stl_path.name}  ({T:,} triangles, {size_mb:.1f} MB)")


def _resolve_targets(args):
    relevel = '--relevel' in args
    args = [a for a in args if a != '--relevel']

    if len(args) == 0:
        data_dir = DEFAULT_DATA_DIR
        return sorted(data_dir.rglob("*.vk4")) + sorted(data_dir.rglob("*.vk6")), relevel
    arg = Path(args[0])
    if arg.is_dir():
        return sorted(arg.rglob("*.vk4")) + sorted(arg.rglob("*.vk6")), relevel
    return [Path(p) for p in args], relevel


def main():
    targets, relevel = _resolve_targets(sys.argv[1:])

    if not targets:
        print("No .vk4 or .vk6 files found.")
        sys.exit(1)

    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting STL for {len(targets)} file(s) → {out_dir}/  {'[plane leveling ON]' if relevel else ''}\n")

    for p in targets:
        print(f"  {p.name} ...", flush=True)
        try:
            data = read_vk(p)
            if relevel:
                data = relevel_plane(data)
            export_stl(data, p.stem, out_dir)
        except Exception as e:
            print(f"  ERROR on {p.name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
