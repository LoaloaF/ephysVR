# WaferNesting — handover

Places multiple copies of one or more designs onto a 100 mm wafer and emits the combined
GDS for fabrication. You give it DXFs and counts, it finds a legal arrangement and writes
the mask. **The entire run is driven from one config file** — the four stage scripts hold
no paths, clearances or layer names of their own.

The wafer template and alignment marks ship **inside** this folder, so the only external
input is the design DXFs. Everything it writes stays inside it, in `runs/<RUN_TAG>/`.

> **Before anything is fabricated, read §6.** The layer mapping for the electrode-bundle
> config is provisional and has one open decision in it.

---

## 0. Where the folders go

`WaferNesting/` expects to sit **next to `electrode_bundle/`** inside one common folder:

```
generative_probe_design/                <- any name; this is the "common folder"
  electrode_bundle/
    main.py, config.py, ...
    designs/                            <- the generator writes its DXFs here
      electrode_bundle_IONP_id11.dxf
  WaferNesting/                         <- this folder
    00_check_setup.py … 04_verify_wafer.py
    config.py, config_electrode_bundle.py
    new_wafer_actually.GDS              <- wafer template, ships with the folder
    marks_union.wkt                     <- alignment marks, ships with the folder
    runs/                               <- output, created on first run
```

Nothing is searched for. Each config states its input locations explicitly in a `PATHS`
block at the top, and the default is exactly the layout above:

```python
HERE = os.path.dirname(os.path.abspath(__file__))     # .../WaferNesting
ROOT = os.path.dirname(HERE)                          # the common folder
DESIGN_DIR = os.path.join(ROOT, "electrode_bundle", "designs")
```

**If your folders are laid out differently, set `DESIGN_DIR` to the folder holding the
DXFs** — an absolute path is fine. That is the one line that normally needs changing, and
`00_check_setup.py` names it for you when it cannot find something.

Generate a bundle to nest with, if `designs/` is empty:

```bash
cd generative_probe_design                           # the parent of electrode_bundle/
python3 -m electrode_bundle.main all --pattern-i 11
```

---

## 1. Quickstart

```bash
pip install ezdxf shapely numpy matplotlib      # stages 01-02
                                                 # stages 03-04 need KLayout

cd generative_probe_design                       # the common folder
KL=/Applications/klayout.app/Contents/MacOS/klayout
export WAFERNEST_CONFIG=config_electrode_bundle

python3 WaferNesting/00_check_setup.py           # run this FIRST
python3 WaferNesting/01_extract_footprints.py
python3 WaferNesting/02_nest.py
$KL -b -r WaferNesting/03_export_wafer.py
$KL -b -r WaferNesting/04_verify_wafer.py
```

**Start with `00_check_setup.py`.** It resolves every path the chosen config depends on,
checks the Python packages and KLayout, lists the DXFs actually present in `DESIGN_DIR`,
and on failure prints exactly which setting in which file to edit. It changes nothing.

| Stage | Does | Needs |
|---|---|---|
| `00_check_setup.py` | verifies the install and every input path | python3 |
| `01_extract_footprints.py` | design DXF → silhouette + centroid offset (+ per-kind envelopes) | python3 |
| `02_nest.py` | silhouettes → poses `[kind, angle, x, y]` | python3 |
| `03_export_wafer.py` | poses + **original DXFs** → wafer GDS | KLayout |
| `04_verify_wafer.py` | re-checks the wafer against the sources | KLayout |

The scripts anchor their paths to the folder itself, so they run from any working
directory; the commands above are the ones that have been exercised.

### Choosing a config

Every stage reads `WAFERNEST_CONFIG`, defaulting to `config`. A new design gets its own
config file rather than edits that destroy the previous one — copy the closest existing
one and change the `PATHS`, `RUN_TAG`, `DESIGNS` and `LAYER_MAP` blocks.

| Config | What | Runs out of the box? |
|---|---|---|
| `config_electrode_bundle.py` | any bundle from the generator; set `BUNDLE_DXF` + `COUNT` | **yes** |
| `config.py` (default) | the 3 connector designs, 4 + 8, seeded — reproduces the fabricated wafer | no: needs the repo's `NewInterconnect/` folder, which is **not** part of this handover |

`config.py` is kept because `reference_run_2026-07/` is the provenance for everything here
and because it is the worked example of a multi-design seeded nest. Pointed at a missing
`DESIGN_DIR` it stops with a clear message rather than doing something surprising.

### Outputs — all land in `runs/<RUN_TAG>/`

| File | What |
|---|---|
| `footprint_<key>.json` + `_offset.json` | silhouette (centred) and the centroid to undo it |
| `envelope_<kind>.json` | union envelope, only for kinds with >1 design |
| `footprints_overlay.png` | visual check of every silhouette |
| `poses.json` | `[kind, angle_deg, x_mm, y_mm]` per piece |
| `nest_preview.png` | the nest, with wafer rim, usable radius and marks |
| `wafer_<tag>.gds` | **the deliverable** |

`RUN_TAG` namespaces everything, so a new run can never overwrite the footprints or poses
an existing wafer was built from. Stage 03 refuses to overwrite an existing GDS.

---

## 2. Architecture — nest a proxy, place the original

```
  design DXFs
       │
  01 ──┴─► footprint polygon + centroid offset       runs/<tag>/footprint_*.json
       │   a COLLISION PROXY, nothing else
       │
  02 ──┴─► poses: [kind, angle, x, y] x N            runs/<tag>/poses.json
       │   simulated annealing to feasibility
       │
  03 ──┴─► FULL geometry from the original DXFs,     runs/<tag>/wafer_<tag>.gds
       │   layer-remapped, placed at those poses
       │
  04 ──┴─► independent verification                  pass / fail
```

The footprint never appears in the output. Stage 02 nests silhouettes; stage 03 goes back
to the original DXFs and places the complete design. Placement is
`t_place * t_rot * t_center` — undo stage 01's centring using that design's own offset,
rotate, translate.

| Module | Role |
|---|---|
| `config_electrode_bundle.py` | **The file you normally edit.** Paths, which bundle, how many, clearances, `LAYER_MAP` |
| `config.py` | The connector config; also the default when `WAFERNEST_CONFIG` is unset |
| `config_common.py` | Derived settings shared by both configs. Holds no paths and looks for nothing on disk |
| `footprint_extract.py` | DXF → silhouette. Shared by stage 01 |
| `00`–`04` | The stages (§1) |
| `new_wafer_actually.GDS` | Base wafer GDS — outline and alignment marks. Ships with the folder |
| `marks_union.wkt` | Alignment marks as a shapely geometry, mm |
| `reference_run_2026-07/` | The fabricated wafer's footprints and poses — the connector nest seed. **Do not edit** |
| `runs/` | **Generated output only** — safe to delete, recreated on the next run |

A design's `dxf` is a filename inside that config's `DESIGN_DIR`, or an absolute path.
`config_common.derive()` computes `OUT_DIR`, the output paths and the `DESIGNS` helpers
from the handful of real settings, so each config states only what is specific to it.

---

## 3. The three ideas worth understanding

### 3.1 The count is an input; there is no density objective

This is the most common misreading of the tool. `count` in `DESIGNS` **is** N. Stage 02
answers *"can these N pieces be placed legally?"* — never *"how many fit?"*

The cost is

```
4*(area outside usable) + 4*(area on marks) + 1*(pairwise overlap area)
```

Every term is zero for *any* legal layout. A neatly tiled arrangement and a sprawling one
with large voids both score exactly `0.0000`; the solver cannot tell them apart and stops
at the first zero it reaches. It is not packing badly — it is not packing at all, only
satisfying constraints.

**When annealing is the right tool:** N is fixed by what you need *and* feasibility is
genuinely hard. That is the connector case exactly — 70 % fill, bottle-shaped pieces with
narrow necks that no regular grid tiles well, a required 4 + 8. It converges on the first
pass.

**When it is the wrong tool:** you want a maximum count, or the piece is near-rectangular.
The bundle is 59.06 mm² inside an 83.7 mm² bbox — 71 % fill — so it tiles almost perfectly
in rows, and continuous rotation becomes actively harmful: every non-zero tilt wastes
space between neighbours. Measured for the 3.100 × 27.003 mm bundle:

| Approach | Copies |
|---|---|
| Structured column nest (28 chord-limited columns, 0.05 mm gap) | **66** |
| bbox bound | 82 |
| area bound | 116 |
| the SA run in `config_electrode_bundle.py` | 24 — *because `COUNT = 24` was requested* |

The 66 is arithmetic over the wafer chord at each column, with no search whatsoever.
Getting a maximum out of SA instead needs an outer bisection loop on N, paying a full
anneal per trial, each slower than the last, to approximate what the chord calculation
gives instantly.

**It also does not scale.** The solver keeps a full N×N overlap matrix and re-tests one
piece against all others per move, so cost per iteration grows linearly with N while the
problem gets harder. 12 and 24 are comfortable; past ~40 it becomes painful. For a real
bundle wafer, write a structured nester and use SA at most to polish the ragged rim.

### 3.2 What the footprint is, and why it must not under-cover

The silhouette is the union of **every** layer, then the *outer* shape of that union.
`extract()` in order:

1. read every LWPOLYLINE and HATCH on every layer, µm → mm;
2. `unary_union` of all of it;
3. `buffer(+0.08).buffer(-0.08)` — closes hairline splits, which is how a ring of 46
   separate polylines becomes one closed body;
4. **drop interior holes** — only exterior rings are kept, so a design with a hole
   through it nests as solid;
5. if the result is still disjoint, **keep the largest part only**;
6. `simplify(0.03)`, then centre on the centroid.

Steps 4 and 5 are the surprising ones. Stage 01 warns when 5 actually discards something.

**Why every layer.** Taking the union of all of them makes the footprint a superset of the
real body by construction, so it can only ever **over**-cover: a stray annotation or
border polyline inflates the footprint, the nester reserves too much space, and you fit
fewer pieces. Wasteful, visible in the printed area, harmless.

Under-covering is the dangerous direction, and it is worth understanding why. Each piece
carries `gA`, the true silhouette, for out-of-bounds and mark overlap, and `gB`, `gA`
grown by `GAP/2`, for pairwise overlap — two grown shapes touching means the real pieces
are closer than `GAP`. That is the *whole* clearance mechanism, and **nothing downstream
re-checks it**: stage 04 verifies counts, per-layer shape totals and max radius, not
spacing. A footprint smaller than the true body therefore yields a perfectly "feasible"
nest whose real features collide on the wafer.

So: keep the footprint conservative. Anything that could shrink it below the real geometry
has to be justified against a failure that reaches the mask, and it is the reason stage 01
asserts that a kind envelope covers each of its members — see gotcha 4.

### 3.3 The clearances are measurements, not preferences

`EDGE_EXCL 3.0` / `GAP 0.05` / `MARK_BUF 0.7` are what the **fabricated** wafer actually
honours. Measured against its own footprints: edge 2.993 mm, min inter-piece 0.000 mm
(pieces are placed flush), min mark clearance 0.701 mm.

The older scripts' `GAP 0.3` / `MARK_BUF 1.0` were never achieved by any real layout.
Nesting from an existing seed at 0.3/1.0 asks the search to open clearances the reference
never had: it plateaus around 0.35 mm² residual forever and looks **exactly** like a slow
annealer. Three seeding strategies and ~90 restarts were burned on this before the cause
was measured. If a run will not converge, check the clearances before adding iterations —
stage 02 prints its seed cost as the first line.

---

## 4. Worked examples

Both were run end to end and verified. Use whichever is closer to your case as the
starting point for a new config.

### 4.1 `config.py` — 3 connector designs, seeded

4 × `12Block_56_15` + 8 smalls (4 × `51_35`, 4 × `71_15`, sharing one nest slot), seeded
from `reference_run_2026-07/`.

| | |
|---|---|
| Piece area | 4819 mm² of 6882 mm² usable — **70 % fill** |
| Nest | feasible on the **first polish pass**, no restarts |
| Clearances achieved | gap 0.052 mm, edge 3.004 mm, marks 3.085 mm |
| Footprints | match `reference_run_2026-07/` at **0.00000000 mm²** symmetric difference; offsets bit-identical |
| Wafer | 1/0 = 14476, 3/0 = 27264, 5/0 = 8832, 6/0 = 11904 shapes; max radius **46.975 mm** — identical to the fabricated `WaferWith12_…_FINAL1.gds` |

### 4.2 `config_electrode_bundle.py` — 1 electrode bundle × 24, free nest

`electrode_bundle/designs/electrode_bundle_IONP_id11Example.dxf`, a 3.100 × 27.003 mm
strip. **Run end to end in the §0 layout**, from the common folder, with the shipped
defaults — these are the numbers you should reproduce.

| | |
|---|---|
| Footprint | 59.058 mm², 18 vertices |
| Piece area | 1417 mm² of 6882 mm² usable — **21 % fill** |
| Nest | feasible on the **first fresh restart** |
| Clearances achieved | gap 0.058 mm, edge 3.339 mm |
| Wafer | 24 instances; 1/0 = 48, 3/0 = 6192, 5/0 = 195696 shapes; max radius 46.661 mm; all checks passed |

Output lands in `runs/electrode_bundle_IONP_id11Example/`, created on the first run.

To nest a different bundle, change two lines — `BUNDLE_DXF` and `COUNT`. `RUN_TAG` follows
the DXF name automatically, so each bundle gets its own output folder and none of them
overwrite each other.

This example is the reason §3.1 and §6.1 exist: the nest is legal but loose (see
`nest_preview.png` — 24 strips at arbitrary angles with large voids), and the design's
layers needed a mapping the connector config never had.

> A note if you compare against older output: the generator's `designs/` copy of id11 has
> 8152 `pattern_ionp` polylines where an earlier copy outside that folder had 12291. The
> **outline is identical** (59.0579 mm², same bbox and centroid) so the nest is unaffected,
> but per-layer shape counts differ. Stage 04 re-derives its expectations from whichever
> DXF the config points at, so it stays self-consistent either way.

---

## 5. Adapting it to a new design

### 5.1 Inspect the DXF first

Two reader assumptions fail **silently** — they produce a plausible wrong footprint rather
than an error.

```bash
python3 -c "
import ezdxf, collections
d = ezdxf.readfile('YOUR.dxf'); m = d.modelspace()
print(collections.Counter((e.dxftype(), e.dxf.layer) for e in m))
xs = [x for e in m for x,y in e.get_points('xy')]
ys = [y for e in m for x,y in e.get_points('xy')]
print('extent %.1f x %.1f' % (max(xs)-min(xs), max(ys)-min(ys)))
"
```

- **Units must be micrometres.** A 50 × 15 mm connector must read ~50000 × 15000. Ignore
  `$INSUNITS` in the header; it is frequently nominal and wrong.
- **Only LWPOLYLINE and HATCH are read.** CIRCLE, ARC, SPLINE and block references
  (INSERT) are invisible and must be exploded first.
- **Write down every layer name.** You need them for `LAYER_MAP`.

### 5.2 Copy a config and edit it

- `RUN_TAG` — names the run and its output folder.
- `DESIGNS` — `key`, `dxf`, `kind`, `count` per design. Designs sharing a `kind` are
  nested against the union **envelope** of their footprints and become interchangeable at
  export. Only group genuinely similar shapes: every slot of that kind reserves
  envelope-sized space, and stage 01 prints the worst-case waste per slot. A design alone
  in its kind is nested against its own exact footprint.
- `LAYER_MAP` — **any layer not in this dict is silently deleted on export.** Add every
  layer of the new design or confirm you meant to drop it.
- `SEED_MODE` — `"seeded"` when a comparable layout exists, `"free"` otherwise.
- Leave `EDGE_EXCL` / `GAP` / `MARK_BUF` alone unless you have a reason (§3.3).

**Seeding dominates runtime.** On the 4 + 8 case free nesting was ~20× worse than seeding
(6.6 mm² vs 0.35 mm²). The trick that worked: going from 5 big + 7 small to 4 + 8, put the
extra small at the pose of a big you are no longer placing — a small is 8 mm shorter and
the same width, so only ~4 mm² pokes outside the vacated silhouette, and it converges in a
single polish pass. `SEED_MODE="seeded"` does this **automatically** whenever the seed has
no spare pose of the right kind, which is why the four near-duplicate
`pack_new_sa_refine_newdesigns*.py` scripts in `NewInterconnect/` collapsed into one here.

### 5.3 Check each stage before moving on

| After | Confirm |
|---|---|
| 01 | printed area and bbox match the design; overlay PNG shows one closed outline, no spurious lobes; no "disjoint parts" warning |
| 02 | reached feasible; `min inter-piece gap` and `min edge clearance` are what you asked for; preview PNG looks sane |
| 03 | the `DROPPED` layer list is empty, or only layers you meant to drop |
| 04 | `ALL CHECKS PASSED` |

---

## 6. Open items — TODO

### 6.1 Confirm the fab layer mapping for the electrode bundle — REQUIRED BEFORE FABRICATION

`config_electrode_bundle.py` maps three layers the connector `LAYER_MAP` had never seen.
Without entries all three would have been silently deleted on export, so something had to
be chosen to run the pipeline at all. Two are safe; one is a real decision.

| DXF layer | Mapped to | Status |
|---|---|---|
| `metal` | 3/0 | Reasonable — single-metal design, takes the Metal1 slot. **Confirm 3/0 is the intended metal layer** |
| `electrodes` | 3/0 | **Safe.** Geometrically a no-op — `Electrodes` lies entirely inside `Metal` (0.000000 mm² outside), so merging cannot change the mask |
| `pattern_ionp` | 5/0, with `Etching` | **OPEN — decide this.** *Not* a no-op. The 12291 IONP wells are merged onto the etch layer on the assumption they are etched features. If they need their own mask, give them a distinct layer number |

Also confirm that dropping positive `Polyimide` is right for this design, as it is for the
connectors.

### 6.2 Write a structured nester if a bundle wafer is actually wanted

The annealer returns legal layouts, not dense ones (§3.1). A column nest fits 66 bundles
where the current run places 24.

---

## 7. Gotchas

1. **Any DXF layer missing from `LAYER_MAP` is silently deleted on export.** No warning,
   no error — the geometry simply is not in the GDS. This is the pipeline's most likely
   and least visible failure, and it is exactly how `EtchingPad` was nearly lost on the
   2026-07 wafer. Stage 03 prints the mapped and dropped lists per design; **read them.**
   Stage 04's per-layer count check is the backstop.
2. **Never reuse poses because the footprint "did not change".** Regenerated designs once
   differed by 0.2–1.0 mm² — visually identical — and that was enough to drop the old poses
   to a 0.0002 mm gap. Re-run stage 02 whenever a DXF changes; `RUN_TAG` makes keeping the
   old set cheap.
3. **Never radius-check these pieces by bounding box.** They are bottle-shaped, so a
   piece's own bbox has empty corners beside the narrow neck, and the axis-aligned bbox of
   a rotated piece is wilder still: 52.4 mm and 59.6 mm respectively, where the geometry
   actually reaches 46.993 mm. Stage 04 uses the convex hull of the real shapes.
4. **The envelope simplify tolerance must stay ≤ the envelope pad.** `ENVELOPE_SIMPLIFY`
   is a separate knob from `SIMPLIFY` for this reason. Using the 0.03 silhouette tolerance
   with the 0.01 pad makes the envelope cut ~0.01 mm² *inside* its own members; stage 01's
   coverage assertion catches it, but only because the assertion is there.
5. **`extract()` is lossy in two deliberate ways.** `buffer(+0.08).buffer(-0.08)` bridges
   splits up to ~160 µm — which is how a ring of 46 separate polylines becomes one closed
   body — and if the result is still a MultiPolygon, **only the largest part is kept**, so
   a design with genuinely disconnected pieces loses all but the biggest. Stage 01 warns
   when that happens.
6. **`runs/` is disposable; `reference_run_2026-07/` is not.** The latter holds the
   fabricated wafer's footprints and the default nest seed. Nothing regenerates it.
7. **SA is stochastic.** If a run stalls, change `RNG_SEED` or raise `RESTARTS`/`ITERS` —
   but read the first printed line first (§3.3). If the cost never moves, the clearances
   are unreachable for that piece count and more iterations will not help.
8. **Stages 03 and 04 run under KLayout's Python**, which has no shapely/numpy/matplotlib.
   Keep the config modules pure standard library or you will break them.
9. **Units are µm in the DXFs, mm everywhere in the pipeline**, and dbu in KLayout
   (`mm * 1e6`).
