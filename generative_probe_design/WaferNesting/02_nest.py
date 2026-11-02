"""Stage 02 -- footprints  ->  poses (the nest).

Simulated annealing on a FEASIBILITY cost, not a density cost. The search stops the
moment cost reaches zero; it does not keep going to squeeze pieces tighter. Zero means
every piece is inside the usable area, off the alignment marks, and at least GAP from
every other piece.

Each piece carries two geometries:
    gA  the true silhouette      -> out-of-bounds area and mark overlap
    gB  gA grown by GAP/2        -> pairwise overlap, so two grown shapes touching
                                    means the real pieces are closer than GAP
cost = 4*(area outside usable) + 4*(area on marks) + 1*(pairwise overlap area)

Solver core is unchanged from pack_new_sa_refine_newdesigns_final.py. The three
near-duplicate variants in NewInterconnect/ differed ONLY in how they seeded; both
strategies live here behind config.SEED_MODE.

RUN WITH SYSTEM PYTHON (shapely + numpy + matplotlib):
    python3 WaferNesting/02_nest.py

SA is stochastic. If a run stalls: change RNG_SEED, or raise RESTARTS/ITERS. But first
read the FIRST printed line -- if the seeded polish is already far from zero AND the
restarts do not move it, the constraints are unreachable and more iterations will not
help. Check GAP/MARK_BUF/EDGE_EXCL against config.py before burning an hour.

Poses and preview are written ONLY if the run reaches feasibility.
"""
import json, os, math, random, numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.affinity import rotate, translate
from shapely import wkt as _w
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPL

import sys, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module(os.environ.get("WAFERNEST_CONFIG", "config"))

os.makedirs(C.OUT_DIR, exist_ok=True)
random.seed(C.RNG_SEED); np.random.seed(C.RNG_SEED)

R0, FLAT0 = C.WAFER_R, C.WAFER_FLAT_Y
R = R0 - C.EDGE_EXCL; FLAT = FLAT0 + C.EDGE_EXCL; HALF = C.GAP / 2
W_OOB = W_MARK = 4.0; W_OV = 1.0
SNAP = (lambda t: round(t / 90.0) * 90.0 % 360.0) if C.CARDINAL_ONLY else (lambda t: t)

usable = Point(0, 0).buffer(R, resolution=128).intersection(box(-R-1, FLAT, R+1, R+1))
MARKS  = _w.loads(open(C.MARKS_WKT).read()).buffer(C.MARK_BUF).simplify(0.05)

fpA = {}
for kind in C.kinds():
    p = C.kind_footprint_path(kind)
    if not os.path.exists(p):
        raise SystemExit("missing %s -- run 01_extract_footprints.py first" % p)
    fpA[kind] = Polygon(json.load(open(p)))
fpB = {k: v.buffer(HALF, join_style=2) for k, v in fpA.items()}

kinds = [k for k in C.kinds() for _ in range(C.count_of(k))]
N = len(kinds)
print("nesting %d pieces: %s" % (N, ", ".join("%d x %s" % (C.count_of(k), k) for k in C.kinds())))
print("clearances: edge %.2f  gap %.2f  marks %.2f mm   seed mode: %s"
      % (C.EDGE_EXCL, C.GAP, C.MARK_BUF, C.SEED_MODE))
print("total piece area %.1f mm2 of %.1f mm2 usable (%.0f%%)"
      % (sum(fpA[k].area for k in kinds), usable.area,
         100 * sum(fpA[k].area for k in kinds) / usable.area))


def gA(i, s): return translate(rotate(fpA[kinds[i]], s[2], origin=(0, 0)), s[0], s[1])
def gB(i, s): return translate(rotate(fpB[kinds[i]], s[2], origin=(0, 0)), s[0], s[1])


def full(st):
    GB = [gB(i, st[i]) for i in range(N)]
    GAo = [gA(i, st[i]).difference(usable).area for i in range(N)]
    GAm = [(gA(i, st[i]).intersection(MARKS).area if gA(i, st[i]).intersects(MARKS) else 0.0) for i in range(N)]
    oob = np.array(GAo); mk = np.array(GAm); ov = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if GB[i].intersects(GB[j]):
                a = GB[i].intersection(GB[j]).area; ov[i, j] = ov[j, i] = a
    return GB, oob, mk, ov


def cost_of(oob, mk, ov): return W_OOB*oob.sum() + W_MARK*mk.sum() + W_OV*ov.sum()/2


def solve(st, iters, T0, polish=False):
    GB, oob, mk, ov = full(st); cost = cost_of(oob, mk, ov)
    best = cost; bst = [s[:] for s in st]
    for it in range(iters):
        if cost < 1e-7:
            return 0.0, [s[:] for s in st]
        frac = it / iters; T = 0.0 if polish else T0 * (1 - frac) + 0.005
        i = random.randrange(N)
        if (not polish) and random.random() < 0.05:
            i = int(np.argmax(oob*W_OOB + mk*W_MARK + W_OV*ov.sum(1)))
            nx = random.uniform(-R, R); ny = random.uniform(FLAT, R); nt = SNAP(random.uniform(0, 360))
        else:
            sc = (3.0*(1-frac) + 0.15) if not polish else 0.12
            nx = st[i][0] + random.gauss(0, sc); ny = st[i][1] + random.gauss(0, sc)
            nt = SNAP(st[i][2] + random.gauss(0, (20*(1-frac) + 2) if not polish else 1.5))
        old = st[i][:]; st[i] = [nx, ny, nt]
        gb_ = gB(i, st[i]); ga_ = gA(i, st[i])
        no = ga_.difference(usable).area
        nm = ga_.intersection(MARKS).area if ga_.intersects(MARKS) else 0.0
        nov = np.zeros(N)
        for j in range(N):
            if j != i and gb_.intersects(GB[j]): nov[j] = gb_.intersection(GB[j]).area
        d = W_OOB*(no - oob[i]) + W_MARK*(nm - mk[i]) + W_OV*(nov.sum() - ov[i].sum())
        if d < 0 or (T > 1e-9 and random.random() < math.exp(-d / T)):
            GB[i] = gb_; oob[i] = no; mk[i] = nm; ov[i, :] = nov; ov[:, i] = nov; cost += d
            if cost < best: best = cost; bst = [s[:] for s in st]
        else:
            st[i] = old
    return best, bst


def random_layout():
    return [[random.uniform(-R, R), random.uniform(FLAT, R), SNAP(random.uniform(0, 360))]
            for _ in range(N)]


def seeded_layout():
    """Start from an existing layout, matching kinds where possible.

    A piece whose kind the seed has no spare pose for is dropped into the pose of a
    seed piece that is NOT being placed this time (the "slot" trick). That matters:
    the pieces here are similar in width and differ mostly in length, so an extra
    piece parked in a vacated slot pokes only a few mm2 outside an already-feasible
    silhouette. Dropping it at a random spot instead is what made earlier runs
    plateau around 0.35 mm2 and look like a stuck annealer.
    """
    seed = json.load(open(C.SEED_POSES))
    by_kind = {}
    for p in seed:
        by_kind.setdefault(p[0], []).append(p)
    out, borrowed, randomised = [], 0, 0
    for i in range(N):
        pool = by_kind.get(kinds[i])
        if pool:
            p = pool.pop(0)
        else:
            spare = [k for k, v in by_kind.items() if v]
            if spare:
                p = by_kind[spare[0]].pop(0); borrowed += 1
            else:
                p = [kinds[i], random.uniform(0, 360), random.uniform(-R, R), random.uniform(FLAT, R)]
                randomised += 1
        out.append([p[2], p[3], SNAP(float(p[1]))])
    print("seed: %s  (%d matched, %d in vacated slots, %d random)"
          % (os.path.basename(C.SEED_POSES), N - borrowed - randomised, borrowed, randomised))
    return out


# --- search -------------------------------------------------------------------
glob = None
if C.SEED_MODE == "seeded":
    if not C.SEED_POSES or not os.path.exists(str(C.SEED_POSES)):
        raise SystemExit("SEED_MODE='seeded' but SEED_POSES does not exist: %s" % C.SEED_POSES)
    base = seeded_layout()
    # zero-temperature polish straight off the untouched seed FIRST, before any jitter
    # has a chance to disturb pieces that were already placed well
    c0, sf0 = solve([s[:] for s in base], iters=C.ITERS, T0=0.0, polish=True)
    glob = (c0, sf0)
    print("polish from seed: best=%.4f%s" % (c0, "   ***FEASIBLE***" if c0 < 1e-7 else ""), flush=True)
    for r in range(C.RESTARTS if glob[0] >= 1e-7 else 0):
        st = [s[:] for s in glob[1]]
        for s in st:
            s[0] += random.gauss(0, 0.8); s[1] += random.gauss(0, 0.8); s[2] = SNAP(s[2] + random.gauss(0, 12))
        c, sf = solve(st, iters=C.ITERS, T0=1.5)
        if c < glob[0]: glob = (c, sf)
        print("restart %d: best=%.4f  global=%.4f%s"
              % (r, c, glob[0], "   ***FEASIBLE***" if glob[0] < 1e-7 else ""), flush=True)
        if glob[0] < 1e-7: break
elif C.SEED_MODE == "free":
    # even restarts explore from scratch, odd restarts basin-hop around the best so far
    for r in range(C.RESTARTS):
        if r % 2 == 0 or glob is None:
            st = random_layout(); how = "fresh"
        else:
            st = [s[:] for s in glob[1]]
            for s in st:
                s[0] += random.gauss(0, 0.8); s[1] += random.gauss(0, 0.8); s[2] = SNAP(s[2] + random.gauss(0, 12))
            how = "hop"
        c, sf = solve(st, iters=C.ITERS, T0=1.5)
        if glob is None or c < glob[0]: glob = (c, sf)
        print("restart %d (%-5s): best=%.4f  global=%.4f%s"
              % (r, how, c, glob[0], "   ***FEASIBLE***" if glob[0] < 1e-7 else ""), flush=True)
        if glob[0] < 1e-7: break
else:
    raise SystemExit("SEED_MODE must be 'seeded' or 'free', got %r" % C.SEED_MODE)

c, sf = solve([s[:] for s in glob[1]], iters=C.ITERS + 30000, T0=0.0, polish=True)
if c < glob[0]: glob = (c, sf)
c, sf = glob

# --- report and write ---------------------------------------------------------
GB, oob, mk, ov = full(sf)
mind = min(gA(i, sf[i]).distance(gA(j, sf[j])) for i in range(N) for j in range(i + 1, N))
edge = Point(0, 0).buffer(R0, 512).intersection(box(-R0-1, FLAT0, R0+1, R0+1)).boundary
minedge = min(gA(i, sf[i]).distance(edge) for i in range(N))
minmark = min(gA(i, sf[i]).distance(MARKS) for i in range(N))
print("\nresidual overlap=%.4f  oob=%.4f  marks=%.4f" % (ov.sum()/2, oob.sum(), mk.sum()))
print("min inter-piece gap   = %.3f mm  (target >= %.2f)" % (mind, C.GAP))
print("min edge clearance    = %.3f mm  (target >= %.2f)" % (minedge, C.EDGE_EXCL))
print("min mark clearance    = %.3f mm  (buffered target >= 0)" % minmark)

if (ov.sum()/2 + oob.sum() + mk.sum()) >= 1e-6:
    raise SystemExit("\nNO FEASIBLE LAYOUT this run. SA is stochastic -- change RNG_SEED "
                     "or raise RESTARTS/ITERS. If the cost never moves, the clearances "
                     "are unreachable for this piece count, not the search too short.")

poses = [[kinds[i], sf[i][2] % 360, sf[i][0], sf[i][1]] for i in range(N)]
json.dump(poses, open(C.POSES_PATH, "w"))

fig, ax = plt.subplots(figsize=(8, 8)); th = np.linspace(0, 2*np.pi, 400)
ax.plot(R0*np.cos(th), R0*np.sin(th), "k-", lw=1, label="wafer r=%.0f" % R0)
ax.plot(R*np.cos(th), R*np.sin(th), "r--", lw=.7, label="usable r=%.0f" % R)
ax.axhline(FLAT0, color="gray", lw=1)
for geom in (MARKS.geoms if MARKS.geom_type == "MultiPolygon" else [MARKS]):
    ax.add_patch(MPL(list(geom.exterior.coords), closed=True, fc="orange", ec="darkorange", lw=.5, alpha=.6))
palette = ["steelblue", "indianred", "seagreen", "goldenrod", "purple"]
kcol = {k: palette[i % len(palette)] for i, k in enumerate(C.kinds())}
for i, (k, t, x, y) in enumerate(poses):
    g = gA(i, sf[i])
    ax.add_patch(MPL(list(g.exterior.coords), closed=True, fc=kcol[k], ec="k", lw=.4, alpha=.85))
    ax.text(x, y, str(i+1), ha="center", va="center", fontsize=8, color="w")
ax.set_aspect("equal"); ax.set_xlim(-52, 52); ax.set_ylim(-52, 52)
ax.set_title("%s: %s  (%.0f mm edge, min gap %.2f mm)"
             % (C.RUN_TAG, " + ".join("%d %s" % (C.count_of(k), k) for k in C.kinds()),
                C.EDGE_EXCL, mind))
ax.legend(loc="upper right", fontsize=7); plt.tight_layout()
plt.savefig(C.PREVIEW_PATH, dpi=110)
print("\nFEASIBLE -> wrote %s + %s" % (os.path.basename(C.POSES_PATH), os.path.basename(C.PREVIEW_PATH)))
