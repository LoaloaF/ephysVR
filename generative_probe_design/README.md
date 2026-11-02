# generative_probe_design — electrode bundles and wafer nesting

Two folders, one pipeline. `electrode_bundle/` computes a flexible electrode probe from
parameters and emits its CAD; `WaferNesting/` takes designs like that one, arranges as
many copies as you ask for on a 100 mm wafer, and writes the combined GDS for fab.

They are independent — you can nest any DXF, not just a bundle — but this is the path
they were built for:

```
electrode_bundle/            WaferNesting/
  config.py                    config_electrode_bundle.py
      │                              │
      ▼                              ▼
  designs/*.dxf  ───────────────►  wafer_<tag>.gds
   the probe geometry              N copies, nested, fab-ready
```

Each folder has its own **`HANDOVER.md`** with the full detail. Read those; this page is
only the map and the order to do things in.

> **Before anything is fabricated**, read `WaferNesting/HANDOVER.md` §6. The DXF-layer →
> fab-layer mapping for the electrode bundle is provisional and has one open decision in
> it (`pattern_ionp`).

---

## Layout

Keep both folders side by side inside this one. `WaferNesting/` finds the designs at
`../electrode_bundle/designs`, so moving either folder means editing one path
(`DESIGN_DIR`, at the top of the config).

```
generative_probe_design/      <- run everything from HERE
  electrode_bundle/
    HANDOVER.md
    main.py, config.py, ...
    designs/                  <- shipped ...Example input DXFs
    tests/
  probe_designs_output/       <- ALL generated output lands here
  WaferNesting/
    HANDOVER.md
    00_check_setup.py … 04_verify_wafer.py
    config_electrode_bundle.py, config.py
    new_wafer_actually.GDS    <- wafer template, included
    marks_union.wkt           <- alignment marks, included
    runs/                     <- output, created on first run
```

**Run every command from this folder**, not from inside either subfolder. Both tools
depend on it: `python3 -m electrode_bundle.main` needs the package's parent on the path,
and `WaferNesting`'s configs resolve `../electrode_bundle/designs` from here.

---

## Prerequisites

```bash
pip install ezdxf shapely numpy matplotlib pytest
```

Plus **KLayout** for the last two nesting stages (GDS work). On macOS the scripts look
for `/Applications/klayout.app/Contents/MacOS/klayout`.

Check both halves before starting:

```bash
python3 -m pytest electrode_bundle/tests -q                       # 27 tests
WAFERNEST_CONFIG=config_electrode_bundle \
  python3 WaferNesting/00_check_setup.py                          # resolves every path
```

`00_check_setup.py` changes nothing. If an input is missing it names the exact setting in
the exact file to edit, and lists the DXFs it can actually see.

---

## 1. Generate the designs

**The usual command.** Builds every design listed in `electrode_bundle/design_sets.py`:

```bash
python3 -m electrode_bundle.batch
```

Everything lands in **`probe_designs_output/`**: one `electrode_bundle_<name>.dxf` + `.png`
and one `electrode_to_flex_mapping_<name>.json` per design, plus
**`all_designs_side_by_side.dxf`** — every design laid out in a row in one drawing, which is
the file to open when you just want to look at the set. Add `--no-render` to skip the PNGs
(much faster) or `--no-side-by-side` to skip the combined DXF.

To change what gets built, edit `DESIGNS` in `electrode_bundle/design_sets.py`; its
docstring has a labelled diagram of every length parameter.

Rebuild the combined DXF on its own, optionally for a subset:

```bash
python3 -m electrode_bundle.side_by_side              # all of DESIGNS
python3 -m electrode_bundle.side_by_side C1 C2 H1     # name substrings
```

**One-off, single design.** The older entry point, for a single parametric probe with a
magnetic-ID barcode:

```bash
python3 -m electrode_bundle.main all --pattern-i 11
```

Writes `probe_designs_output/electrode_bundle_IONP_id11.dxf` plus its preview, the
per-stripe JSON and the channel→flex mapping. `--pattern-i` is the magnetic ID, 0–15, and
appears in the filename, so different IDs coexist. Details: `electrode_bundle/HANDOVER.md`.

> The DXFs already in `designs/` carry an **`Example`** suffix, and
> `config_electrode_bundle.py` points at those so the nesting run below works immediately.
> Files you generate yourself will *not* have that suffix — update `BUNDLE_DXF` to match.
>
> `electrode_bundle/HANDOVER.md` calls `designs/` disposable, which is true of anything
> *you* generate there. The `…Example` files are the shipped samples and are not
> reproduced by any command — keep them, or step 2 has nothing to nest until you have
> generated a design of your own.

## 2. Nest it onto a wafer

```bash
export WAFERNEST_CONFIG=config_electrode_bundle
KL=/Applications/klayout.app/Contents/MacOS/klayout

python3 WaferNesting/00_check_setup.py
python3 WaferNesting/01_extract_footprints.py
python3 WaferNesting/02_nest.py
$KL -b -r WaferNesting/03_export_wafer.py
$KL -b -r WaferNesting/04_verify_wafer.py
```

Set which design and how many at the top of `WaferNesting/config_electrode_bundle.py`
(`BUNDLE_DXF`, `COUNT`). Output lands in `WaferNesting/runs/<RUN_TAG>/`, where `RUN_TAG`
follows the DXF name, so runs never overwrite each other. The deliverable is
`wafer_<RUN_TAG>.gds`; `nest_preview.png` next to it shows the arrangement.

Stage 04 should end with `ALL CHECKS PASSED`.

---

## Two things worth knowing up front

**The piece count is an input, not a maximum.** `COUNT` says how many to place. The nester
solves for a *legal* arrangement and stops the moment it has one — it has no density
objective and will not fill the wafer for you. `WaferNesting/HANDOVER.md` §3.1 has the
numbers: for the id11 bundle a structured column nest fits 66 copies where the search
places whatever `COUNT` you asked for.

**A DXF layer missing from `LAYER_MAP` is silently deleted on export.** Stage 03 prints
what it mapped and what it dropped on every run — read that output. Stage 04's per-layer
shape-count check is the backstop.

