"""Stage 01 -- design DXF  ->  footprint polygon (+ centroid offset).

For each design in config.DESIGNS, extracts the outer silhouette and writes it centred
on its own centroid, plus the offset needed to put it back. For each kind that groups
more than one design, also writes the union ENVELOPE those designs are nested against.

Nothing here touches the wafer. The footprint is only a collision proxy for stage 02;
the real geometry is placed from the original DXF in stage 03.

RUN WITH SYSTEM PYTHON (ezdxf + shapely + matplotlib):
    python3 WaferNesting/01_extract_footprints.py

CHECK BEFORE MOVING ON: the printed area and bbox must match the design you expect,
and the overlay PNG must show a single closed outline with no spurious lobes.
"""
import json, os
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module(os.environ.get("WAFERNEST_CONFIG", "config"))
from footprint_extract import extract

os.makedirs(C.OUT_DIR, exist_ok=True)

print("run tag   : %s" % C.RUN_TAG)
print("out dir   : %s" % C.OUT_DIR)
print()

shapes = {}
for d in C.DESIGNS:
    dxf = C.design_dxf(d)
    if not os.path.exists(dxf):
        raise SystemExit("missing DXF: %s" % dxf)
    fp, cx, cy = extract(dxf, close_gap=C.CLOSE_GAP, simplify=C.SIMPLIFY)
    mnx, mny, mxx, mxy = fp.bounds
    json.dump(list(fp.exterior.coords), open(C.footprint_path(d["key"]), "w"))
    json.dump({"cx": cx, "cy": cy}, open(C.offset_path(d["key"]), "w"))
    shapes[d["key"]] = fp
    print("%-14s kind=%-6s x%d  area=%8.3f mm2  bbox=%7.3f x %7.3f mm  "
          "centroid=(%9.3f, %9.3f)  %d verts"
          % (d["key"], d["kind"], d["count"], fp.area,
             mxx - mnx, mxy - mny, cx, cy, len(fp.exterior.coords)))

# --- per-kind envelopes -------------------------------------------------------
print()
for kind in C.kinds():
    ds = C.designs_of(kind)
    if len(ds) == 1:
        print("kind %-6s: single design %s, nested against its own footprint"
              % (kind, ds[0]["key"]))
        continue
    env = unary_union([shapes[d["key"]] for d in ds]).buffer(C.ENVELOPE_PAD).simplify(C.ENVELOPE_SIMPLIFY)
    if env.geom_type == "MultiPolygon":
        env = max(env.geoms, key=lambda g: g.area)
    env = Polygon(env.exterior)
    json.dump(list(env.exterior.coords), open(C.kind_footprint_path(kind), "w"))
    mnx, mny, mxx, mxy = env.bounds
    print("kind %-6s: envelope of %d designs  area=%8.3f mm2  bbox=%7.3f x %7.3f mm"
          % (kind, len(ds), env.area, mxx - mnx, mxy - mny))
    for d in ds:
        s = shapes[d["key"]]
        ok = env.covers(s)
        print("    covers %-14s %-5s  (outside envelope %.6f mm2)"
              % (d["key"], ok, s.difference(env).area))
        if not ok and s.difference(env).area > 1e-4:
            raise SystemExit("envelope does not cover %s -- pieces would collide" % d["key"])
    waste = env.area - min(shapes[d["key"]].area for d in ds)
    print("    worst-case wasted area per slot: %.3f mm2 -- if this is large the "
          "designs are too dissimilar to share a kind" % waste)

# --- overlay plot -------------------------------------------------------------
n = len(C.DESIGNS)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 7), squeeze=False)
for ax, d in zip(axes[0], C.DESIGNS):
    fp = shapes[d["key"]]
    ax.plot(*fp.exterior.xy, "r-", lw=1.2, label="footprint")
    env_path = C.kind_footprint_path(d["kind"])
    if len(C.designs_of(d["kind"])) > 1 and os.path.exists(env_path):
        env = Polygon(json.load(open(env_path)))
        ax.plot(*env.exterior.xy, "b--", lw=1.0, label="kind envelope")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("%s  (%.1f mm2)" % (d["key"], fp.area), fontsize=9)
plt.tight_layout()
plt.savefig(C.OVERLAY_PATH, dpi=120)
print("\nwrote footprints + %s" % os.path.basename(C.OVERLAY_PATH))
