"""Stage 04 -- independent verification of the exported wafer.

Re-reads the wafer GDS and re-derives what it SHOULD contain straight from the source
DXFs, rather than trusting anything stage 03 reported. Checks:

  1. instance counts   -- the right number of each design is present;
  2. per-layer shape counts -- for every fab layer, the wafer holds at least
     count x (shapes in that source design). This is the check that catches a layer
     silently deleted by a gap in LAYER_MAP, which is the pipeline's most likely
     and least visible failure;
  3. placement radius  -- no placed geometry reaches past R_MAX_MM.

Check 3 uses the CONVEX HULL of each cell's real shapes, never a bounding box. These
pieces are bottle-shaped: on the 2026-07 designs a bbox reads 52.4 mm, and the
axis-aligned bbox of a rotated piece 59.6 mm, where the geometry actually reaches
46.993 mm. A bbox test here fails designs that are perfectly fine.

NOT checked here: inter-piece clearance. That is enforced in stage 02 against the
footprints, and nothing downstream re-tests the true shapes against each other -- which
is why an under-covering footprint in stage 01 is a real risk, not a cosmetic one.

RUN WITH KLAYOUT:
    /Applications/klayout.app/Contents/MacOS/klayout -b -r WaferNesting/04_verify_wafer.py

Override the target with CHECK_GDS=<path> to verify some other wafer file.
"""
import pya, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
C = importlib.import_module(os.environ.get("WAFERNEST_CONFIG", "config"))

GDS = os.environ.get("CHECK_GDS", C.WAFER_OUT)
if not os.path.isabs(GDS):
    GDS = os.path.join(C.HERE, GDS)
fails = []


def count_by_ld(layout, cell):
    """{(layer, datatype): shape count} over the cell and everything beneath it."""
    out = {}
    for li in layout.layer_indexes():
        info = layout.get_info(li)
        ld = (info.layer, info.datatype)
        n = 0
        it = cell.begin_shapes_rec(li)
        while not it.at_end():
            n += 1
            it.next()
        if n:
            out[ld] = out.get(ld, 0) + n
    return out


def source_counts(dxf):
    """Per mapped (layer, datatype) shape counts for one source DXF, as stage 03 maps it."""
    ly = pya.Layout()
    opt = pya.LoadLayoutOptions(); opt.dxf_unit = 1.0; opt.dxf_polyline_mode = 2
    ly.read(dxf, opt)
    for li in list(ly.layer_indexes()):
        nm = ly.get_info(li).name.lower()
        if nm in C.LAYER_MAP:
            ly.set_info(li, pya.LayerInfo(*C.LAYER_MAP[nm]))
        else:
            ly.clear_layer(li)
    return count_by_ld(ly, ly.top_cell())


def convex_hull(pts):
    """Monotone chain. Max radius over a point set == max radius over its hull."""
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) > 0:
                    break
                out.pop()
            out.append(p)
        return out[:-1]
    return half(pts) + half(reversed(pts))


def cell_hull(layout, cell):
    """Convex hull of every real shape in the cell, in cell coordinates (dbu)."""
    pts = []
    for li in layout.layer_indexes():
        it = cell.begin_shapes_rec(li)
        while not it.at_end():
            s, t = it.shape(), it.trans()
            p = None
            if s.is_polygon():
                p = s.polygon
            elif s.is_box():
                p = pya.Polygon(s.box)
            elif s.is_path():
                p = s.path.polygon()
            if p is not None:
                for v in p.transformed(t).each_point_hull():
                    pts.append((v.x, v.y))
            it.next()
    return convex_hull(pts)


if not os.path.exists(GDS):
    print("FAIL: %s does not exist" % GDS)
    sys.exit(1)
print("verifying %s\n" % GDS)

wafer = pya.Layout(); wafer.read(GDS); wtop = wafer.top_cell()
dbu = wafer.dbu   # um per dbu

EXPECT = {"CONN_%s" % d["key"].upper(): (C.design_dxf(d), d["count"])
          for d in C.DESIGNS}
TOTAL = sum(d["count"] for d in C.DESIGNS)

# --- 1. instance counts -------------------------------------------------------
seen, insts = {}, []
for inst in wtop.each_inst():
    nm = wafer.cell(inst.cell_index).name
    seen[nm] = seen.get(nm, 0) + 1
    insts.append((nm, inst))
print("instances by cell: %s" % seen)
for nm, (_dxf, n) in EXPECT.items():
    if seen.get(nm, 0) != n:
        fails.append("expected %d instances of %s, found %d" % (n, nm, seen.get(nm, 0)))
total = sum(seen.get(nm, 0) for nm in EXPECT)
if total != TOTAL:
    fails.append("expected %d connector instances in total, found %d" % (TOTAL, total))

# --- 2. per-layer shape counts ------------------------------------------------
print()
expected_ld = {}
for nm, (dxf, n) in EXPECT.items():
    for ld, c in source_counts(dxf).items():
        expected_ld[ld] = expected_ld.get(ld, 0) + n * c
got_ld = count_by_ld(wafer, wtop)
for ld in sorted(set(expected_ld) | set(got_ld)):
    exp, got = expected_ld.get(ld, 0), got_ld.get(ld, 0)
    # the wafer template contributes its own outline/mark shapes, so got may exceed exp
    print("  layer %2d/%d: connectors expect %8d, wafer has %8d   %s"
          % (ld[0], ld[1], exp, got, "OK" if got >= exp else "MISSING"))
    if got < exp:
        fails.append("layer %d/%d: %d shapes expected, wafer has only %d" % (ld[0], ld[1], exp, got))
if not expected_ld:
    fails.append("no source shapes mapped at all -- LAYER_MAP matches none of the DXF layers")

# --- 3. placement radius ------------------------------------------------------
print()
hulls = {}
for nm in EXPECT:
    idx = wafer.cell_by_name(nm)
    if idx < 0:
        continue
    hulls[nm] = cell_hull(wafer, wafer.cell(idx))
    print("  hull of %-20s: %d points" % (nm, len(hulls[nm])))

worst, worst_nm = 0.0, None
for nm, inst in insts:
    if nm not in hulls:
        continue
    tr = inst.cplx_trans
    for x, y in hulls[nm]:
        p = tr.trans(pya.Point(x, y))
        r = ((p.x * dbu / 1000.0) ** 2 + (p.y * dbu / 1000.0) ** 2) ** 0.5
        if r > worst:
            worst, worst_nm = r, nm
print("\nmax placed-geometry radius = %.3f mm (limit %.1f mm, worst piece %s)"
      % (worst, C.R_MAX_MM, worst_nm))
if worst > C.R_MAX_MM:
    fails.append("placed geometry reaches r=%.3f mm on %s, past the %.1f mm limit"
                 % (worst, worst_nm, C.R_MAX_MM))

# --- verdict ------------------------------------------------------------------
if fails:
    for f in fails:
        print("FAIL: %s" % f)
    sys.exit(1)
print("ALL CHECKS PASSED")
