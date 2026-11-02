"""Stage 03 -- poses + design DXFs  ->  wafer GDS.

Nothing is "extracted" here. The footprints from stage 01 were only a collision proxy;
this stage goes back to the ORIGINAL DXFs, remaps their layers for fab, and places the
complete geometry at the poses stage 02 found.

Placement for each piece is   t_place * t_rot * t_center   , i.e. undo the centring
that stage 01 applied (using that design's own offset JSON), rotate to the pose angle,
then translate onto the wafer. Using an offset from a DIFFERENT footprint run than the
poses came from will shift pieces off their nested slots -- config.RUN_TAG keeps a run's
footprints, offsets and poses together in one folder precisely to prevent that.

Designs sharing a kind are assigned ALTERNATELY down that kind's poses, so each variant
is spread across the wafer rather than clustered in one region.

RUN WITH KLAYOUT:
    /Applications/klayout.app/Contents/MacOS/klayout -b -r WaferNesting/03_export_wafer.py

WARNING: any DXF layer missing from config.LAYER_MAP is silently DELETED. This script
prints every layer it maps and every layer it drops -- read that output.
"""
import pya, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
C = importlib.import_module(os.environ.get("WAFERNEST_CONFIG", "config"))

if not os.path.exists(C.POSES_PATH):
    raise SystemExit("missing %s -- run 02_nest.py first" % C.POSES_PATH)
if os.path.exists(C.WAFER_OUT):
    raise SystemExit("refusing to overwrite %s -- bump RUN_TAG in config.py or delete it"
                     % C.WAFER_OUT)

poses = json.load(open(C.POSES_PATH))
expected = sum(C.count_of(k) for k in C.kinds())
assert len(poses) == expected, "expected %d poses, got %d" % (expected, len(poses))

# --- assign a concrete design to each pose ------------------------------------
# poses carry a kind; within a kind, cycle through that kind's designs so each is
# spread across the wafer. Counts are asserted afterwards.
cursor = {k: 0 for k in C.kinds()}
assign = []
for kind, *_ in poses:
    ds = C.designs_of(kind)
    assign.append(ds[cursor[kind] % len(ds)]["key"])
    cursor[kind] += 1
for d in C.DESIGNS:
    got = assign.count(d["key"])
    assert got == d["count"], "expected %d x %s, got %d" % (d["count"], d["key"], got)

wafer = pya.Layout(); wafer.read(C.WAFER_TEMPLATE); wtop = wafer.top_cell()
print("wafer template: %s" % os.path.basename(C.WAFER_TEMPLATE))

cells = {}
for d in C.DESIGNS:
    dxf = C.design_dxf(d)
    conn = pya.Layout()
    opt = pya.LoadLayoutOptions()
    opt.dxf_unit = 1.0          # DXF coordinates are micrometres
    opt.dxf_polyline_mode = 2   # traces become filled polygons
    conn.read(dxf, opt)
    mapped, dropped = [], []
    for li in list(conn.layer_indexes()):
        nm = conn.get_info(li).name.lower()
        if nm in C.LAYER_MAP:
            conn.set_info(li, pya.LayerInfo(*C.LAYER_MAP[nm]))
            mapped.append("%s->%d/%d" % (nm, C.LAYER_MAP[nm][0], C.LAYER_MAP[nm][1]))
        else:
            conn.clear_layer(li); dropped.append(nm)
    cc = wafer.create_cell("CONN_%s" % d["key"].upper())
    cc.copy_tree(conn.top_cell())
    off = json.load(open(C.offset_path(d["key"])))
    cells[d["key"]] = (cc, int(round(off["cx"] * 1e6)), int(round(off["cy"] * 1e6)))
    print("\n%s  <- %s" % (cc.name, d["dxf"]))
    print("   mapped : %s" % ", ".join(mapped))
    print("   DROPPED: %s" % (", ".join(dropped) if dropped else "(none)"))
    if dropped:
        print("   ^^ confirm every dropped layer is one you meant to drop")

for (kind, theta, xmm, ymm), key in zip(poses, assign):
    cc, cx_dbu, cy_dbu = cells[key]
    t_center = pya.ICplxTrans(1.0, 0.0,          False, -cx_dbu, -cy_dbu)
    t_rot    = pya.ICplxTrans(1.0, float(theta), False, 0, 0)
    t_place  = pya.ICplxTrans(1.0, 0.0,          False, int(round(xmm * 1e6)), int(round(ymm * 1e6)))
    wtop.insert(pya.CellInstArray(cc.cell_index(), t_place * t_rot * t_center))

wafer.write(C.WAFER_OUT)
print("\nwrote %s" % C.WAFER_OUT)
for d in C.DESIGNS:
    print("  %-14s x %d" % (d["key"], assign.count(d["key"])))
print("\nnow verify:  klayout -b -r WaferNesting/04_verify_wafer.py")
