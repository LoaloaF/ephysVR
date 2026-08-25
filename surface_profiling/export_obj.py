"""
Export Keyence VK4/VK6 files as textured OBJ meshes.

Produces per-file:
  <stem>.obj   — mesh with UV coordinates
  <stem>.mtl   — material pointing at <stem>_texture.png
  <stem>_texture.png  — false-color height map (auto-generated)

Usage:
    python export_obj.py                      # all vk4/vk6 in Alexei_one/
    python export_obj.py /path/to/data/dir    # scan a different directory
    python export_obj.py path/to/file.vk4     # single file

Open the .obj in Blender (File → Import → Wavefront .obj) — texture loads automatically.
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, distance_transform_edt

# Shared reader from xyz script
from export_vk_to_xyz import read_vk, relevel_plane, DEFAULT_DATA_DIR, DEFAULT_OUT_DIR

SIGMA   = 0.1   # Gaussian smoothing (px)
Z_SCALE = 3.0   # height exaggeration


def export_obj(data: dict, stem: str, out_dir: Path,
               sigma: float = SIGMA, z_scale: float = Z_SCALE,
               cmap: str = "RdYlGn_r", vmin: float = None, vmax: float = None,
               optical_img: np.ndarray = None):
    """Write OBJ + MTL + texture PNG for a data dict from read_vk().

    optical_img: float32 (nx, ny) array normalised [0,1] from read_optical_from_cag().
                 When provided it is used as a grayscale texture instead of the
                 false-color height map.
    """
    z = data['z'].astype(np.float32)
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32)

    # Fill NaN pixels before smoothing
    nan_mask = np.isnan(z)
    if nan_mask.any():
        _, nn_idx = distance_transform_edt(nan_mask, return_indices=True)
        z[nan_mask] = z[tuple(nn_idx[:, nan_mask])]

    z_color = gaussian_filter(z, sigma=sigma)   # physical heights (µm) — for texture
    z = z_color * z_scale                        # exaggerated heights — for mesh geometry
    nx, ny = z.shape  # (nx rows, ny cols) in grid space

    print(f"    Grid {nx}×{ny} | Z {z_color.min():.2f}…{z_color.max():.2f} µm (×{z_scale} in mesh)")

    # ── texture PNG ───────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    texture_name = f"{stem}_texture.png"
    tex_path = out_dir / texture_name
    # Landscape texture: nx cols (x-axis) × ny rows (y-axis), matching physical aspect ratio
    fig, ax = plt.subplots(figsize=(nx / 100, ny / 100), dpi=100)

    if optical_img is not None:
        # Crop/pad optical to match z grid if sizes differ
        opt = optical_img[:nx, :ny] if optical_img.shape >= (nx, ny) else optical_img
        ax.imshow(opt.T, cmap='gray', vmin=0, vmax=1, origin="upper", aspect="auto")
        print(f"    Texture: optical reflectance (grayscale)")
    else:
        z_min = vmin if vmin is not None else float(np.nanmin(z_color))
        z_max = vmax if vmax is not None else float(np.nanmax(z_color))
        ax.imshow(z_color.T, cmap=cmap, vmin=z_min, vmax=z_max, origin="upper", aspect="auto")
        print(f"    Texture: false-color height  [{z_min:.3f}, {z_max:.3f}] µm")

    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(tex_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # ── colorbar legend PNG (height map only) ─────────────────────────────────
    if optical_img is None:
        z_min = vmin if vmin is not None else float(np.nanmin(z_color))
        z_max = vmax if vmax is not None else float(np.nanmax(z_color))
        cbar_path = out_dir / f"{stem}_colorbar.png"
        fig, ax = plt.subplots(figsize=(1.5, 4))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_min, vmax=z_max))
        cb = fig.colorbar(sm, ax=ax, fraction=1.0, pad=0)
        cb.set_label("Height (µm)", fontsize=11)
        ax.remove()
        # fig.savefig(cbar_path, dpi=100, bbox_inches="tight")
        # plt.close(fig)
        # print(f"    → {cbar_path.name}  (colorbar, z {z_min:.3f}…{z_max:.3f} µm)")

    # ── vertex arrays (use actual µm coordinates) ─────────────────────────────
    vx = x.ravel()
    vy = y.ravel()
    vz = z.ravel()

    # UV: u = ix/(nx-1), v = 1 - iy/(ny-1)  — landscape texture, x horizontal, y vertical
    col_idx = np.tile(np.arange(ny, dtype=np.float32), nx)         # iy for each vertex
    row_idx = np.repeat(np.arange(nx, dtype=np.float32), ny)       # ix for each vertex
    vu = row_idx / max(nx - 1, 1)
    vv = 1.0 - col_idx / max(ny - 1, 1)

    # ── face index arrays ─────────────────────────────────────────────────────
    i_q = np.arange(nx - 1, dtype=np.int32)
    j_q = np.arange(ny - 1, dtype=np.int32)
    ii_q, jj_q = np.meshgrid(i_q, j_q, indexing='ij')
    ii_q = ii_q.ravel(); jj_q = jj_q.ravel()

    v00 = ii_q * ny + jj_q + 1
    v10 = ii_q * ny + (jj_q + 1) + 1
    v01 = (ii_q + 1) * ny + jj_q + 1
    v11 = (ii_q + 1) * ny + (jj_q + 1) + 1

    faces = np.stack([v00, v10, v11, v00, v11, v01], axis=1).reshape(-1, 3)

    # ── write MTL ─────────────────────────────────────────────────────────────
    mtl_path = out_dir / f"{stem}.mtl"
    with open(mtl_path, "w") as f:
        f.write(f"newmtl height_material\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write(f"map_Kd {texture_name}\n")

    # ── write OBJ ─────────────────────────────────────────────────────────────
    obj_path = out_dir / f"{stem}.obj"
    t0 = time.time()
    with open(obj_path, "w") as f:
        f.write(f"# Height map mesh — {stem}\n")
        f.write(f"mtllib {stem}.mtl\n")
        f.write("usemtl height_material\n\n")

        verts = np.column_stack([vx, vy, vz])
        f.write("\n".join("v %.4f %.4f %.4f" % (r[0], r[1], r[2]) for r in verts))
        f.write("\n\n")

        uvs = np.column_stack([vu, vv])
        f.write("\n".join("vt %.6f %.6f" % (r[0], r[1]) for r in uvs))
        f.write("\n\n")

        f.write("\n".join("f %d/%d %d/%d %d/%d" % (a, a, b, b, c, c) for a, b, c in faces))
        f.write("\n")

    print(f"    → {obj_path.name}  ({len(faces):,} triangles, {time.time()-t0:.1f}s)")
    print(f"    → {mtl_path.name}")
    print(f"    → {tex_path.name}")


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
    print(f"Exporting OBJ for {len(targets)} file(s) → {out_dir}/  {'[plane leveling ON]' if relevel else ''}\n")

    for p in targets:
        print(f"  {p.name} ...", flush=True)
        try:
            data = read_vk(p)
            if relevel:
                data = relevel_plane(data)
            export_obj(data, p.stem, out_dir)
        except Exception as e:
            print(f"  ERROR on {p.name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
