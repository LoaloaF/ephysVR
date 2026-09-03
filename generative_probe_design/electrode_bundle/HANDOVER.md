# electrode_bundle — handover

Parametric generator for the 64-channel flexible electrode probe (MEA1K connector project).
You give it numbers, it emits the CAD. **Nothing is drawn by hand and no input file is required**
— the entire device geometry is computed from `config.py`.

This folder is **relocatable**: copy it anywhere and it still runs, with no path to edit. All
generated output goes to **`probe_designs_output/`** — a sibling of this folder, not inside
it, so generated probes sit somewhere obvious rather than buried in the code. The only things
still inside `designs/` are the shipped `…Example` DXFs, which are inputs (WaferNesting points at
them), not output.

---

## 1. Quickstart

```bash
pip install ezdxf matplotlib numpy

cd <the folder CONTAINING electrode_bundle/>     # not into electrode_bundle itself
python3 -m electrode_bundle.main all --pattern-i 15
python3 -m pytest electrode_bundle/tests -q      # 27 tests, all green
```

> Run from the **parent** of `electrode_bundle/` — that's how `python3 -m electrode_bundle.main`
> resolves the package. There is nothing to install and no path to edit.

| Subcommand | Does | Reads a file? |
|---|---|---|
| `bundle`  | electrode bundle DXF + preview PNG | no |
| `mapping` | channel → flex-PCB pad JSON | no |
| `ionp`    | stamps IONP wells onto an **existing** DXF (in-place) | **yes** — see §6.1 |
| `all`     | all three in a single in-memory pass | no |
| `legend`  | decode reference card for all 16 magnetic IDs | no |

Flags: `--pattern-i 0-15` (magnetic ID, sets the output filenames), `--no-render` (skip the PNG).

**Use `all`.** It is the point of the design: the bundle is built **once** and its in-memory
result handed to both consumers, so the mapping never re-derives the routing order and IONP never
re-reads the DXF. `ionp` standalone exists only for stamping wells onto a DXF you already have.

### Outputs — all land in `probe_designs_output/`

Created automatically on first run.

| File | What |
|---|---|
| `electrode_bundle.dxf` + `.png` | the probe geometry |
| `electrode_bundle_IONP_id{NN}.dxf` + `.png` | the probe **plus** the magnetic-ID wells |
| `64ch_ionp_id{NN}_info.json` | per-stripe metadata, with the flex mapping embedded |
| `electrode_to_flex_mapping.json` | only written by the standalone `mapping` command |
| `ionp_id_legend.png` | decode card for IDs 0–15, written by the `legend` command |

Re-running **overwrites in place** for the same `--pattern-i`. Different IDs coexist (the ID is in
the filename), so generating 0–15 gives you 16 sets side by side.

---

## 2. Architecture — one producer, two consumers

```
config.py  ──►  bundle.build_bundle(cfg) ──►  BundleResult
                                               ├──►  mapping.py   (channel → flex pad)
                                               └──►  ionp.py      (magnetic-ID wells)
                          all wired by main.py (CLI)
```

| Module | Role |
|---|---|
| `config.py` | **The only file you normally edit.** `BundleConfig`, `IonpConfig`, `MappingConfig` |
| `geometry.py` | generic primitives (fillets, arcs) — knows nothing about electrodes |
| `shapes.py` | device shapes built on those: polyimide outline + fanout body, hooks, pad columns, routing |
| `bundle.py` | `build_bundle(cfg)` assembles the DXF; `render_bundle` draws the preview |
| `dxf_io.py` | DXF readers/writers |
| `mapping.py` | channel → flex-PCB pad assignment |
| `ionp.py` | the magnetic-ID barcode (well placement, stripe sizing, design rules) |
| `id_legend.py` | renders all 16 IDs as a decode reference card |
| `main.py` | CLI entry point |
| `tests/` | pytest suite (§5) |
| `designs/` | the shipped `…Example` input DXFs — **not** where output goes |

`BundleResult` (`bundle.py`) is the hand-off object: the DXF doc, the contact locations, the
channel→pad assignment, `wire_top`, and the channel count.

To change the design, edit a default in `config.py` **or** pass overrides without touching the
file: `BundleConfig(wide_channel=None, hook_drop=300)`.

---

## 3. The two ideas worth understanding

### 3.1 Electrode depths are a cumulative-gap staircase

The gap *before* channel `i` defaults to `delta_y` (= `8000/63` ≈ 126.98 µm), overridable per
channel via `delta_y_overrides = {i: µm}`. Positions are the **cumulative sum** of gaps, so
editing one gap shifts every electrode above it and leaves everything below untouched.

The shipped default is a dense band: `{33..48: 30.0}` — 16 tightened gaps.

**Anchor — which end stays put.** The staircase is built bottom-up, so the **deep tip (channel 0,
at `bottom_elec`) is pinned** and the shallow end absorbs every change. Tightening gaps pulls the
electrodes *above* the band **downward**, so their wires get *longer*. This is not a small effect:

| | uniform | with the shipped `{33..48: 30.0}` band |
|---|---|---|
| array span | 8000.00 µm | **6448.25 µm** |
| shallow end | — | sits **1551.75 µm lower** |
| deep tip | pinned | pinned (unchanged) |

There is deliberately **no `spacing_anchor` option**, because pinning the shallow end instead
differs only by a *rigid vertical offset* — spacings are identical — and is already reachable
with an existing knob:

```
shift = sum(delta_y - new_gap)   over every overridden gap   # = 1551.75 µm today
to pin the shallow end instead:  bottom_elec -= shift
```

**Invariant either way:** the fanout, the bond pads and the common `wire_top`
(= `bottom_elec - l/2` = 11993.5 µm) do **not** depend on `delta_y` or the gaps, and never move.
Only contact positions and their own wire lengths change. All restated in `config.py` next to
`delta_y_overrides`.

*If* you end up routinely retuning spacing while holding the **shallow** end fixed (e.g. a fixed
cortical-surface reference across probe variants), that manual `bottom_elec` arithmetic gets
error-prone and a `spacing_anchor="top"|"bottom"` flag starts to pay for itself. Until then it
would be a second code path to reach a state one existing knob already reaches.

### 3.2 The IONP barcode is error-correcting

Each shank carries an MRI-readable stripe pattern of iron-oxide wells encoding an **ID 0–15**:

- a **tip-pinned bottom anchor** + a **top anchor** (`ID_TOP_ANCHOR_CENTER = 0.96`) bracketing
  **7 data slots** (`ID_SLOT_CENTERS`), in coordinates normalized **along the electrode array,
  not the whole shank**: `0` = the tip (where the deepest electrode ch0 sits) and `1` =
  `id_top_margin` (220 µm) above the **shallowest** electrode ch63. With today's geometry that
  is y = 0 → 6541 µm, i.e. the barcode spans the recording sites and stops ~5.5 mm short of the
  fanout at `wire_top` = 11993.5 µm. "Top electrode" means the topmost of the 64, not the top of
  the probe;
- the 4-bit ID is expanded across those 7 slots by a **Hamming[7,4,3]** code (`id_codeword` in
  `config.py`). Verified: 16 distinct codewords, **minimum pairwise distance 3** — any two IDs
  differ in ≥3 stripes, so a single misread stripe still decodes;
- stripe **heights are not fixed** — they are grown at build time (`_grow_band`,
  `_grow_tip_anchor`) from the real electrode positions until each stripe clears the hard MRI
  floor of `min_n_wells = 700`. The bottom anchor is grown upward from the tip and ends up the
  fattest stripe (~840 µm), which doubles as an **orientation cue**;
- wells within `keepout_radius = 30 µm` of a contact are dropped.

Encoding is **presence/position only** — no ordering assumptions.

---

## 4. Design rules

`ionp.check_design_rules` runs on every IONP build and prints per-stripe well counts plus a
pass/fail line. It enforces `min_n_wells = 700` (the MRI floor), `max_n_wells = 10_000`, and
`min_strip_distance = 300 µm` between stripes. **Read that output** — it is the build's own
sanity check, and a failure means the barcode may not decode.

---

## 5. Tests

```bash
python3 -m pytest electrode_bundle/tests -q     # 27 tests
```

- **`test_electrode_spacing.py`** — the staircase. Builds the real bundle and reads back
  `electrode_locs[:, 1]`; because pad shape is identical between configs, y-differences isolate
  the electrode shift exactly. Pins **locality** (nothing below an edited gap moves) and
  **accumulation** (edits sum going up), plus out-of-range gap keys raising.
  Baselines come from `_base_cfg()`, **not** a bare `BundleConfig()` — the shipped default carries
  the dense band and would contaminate every comparison. Keep it that way.
- **`test_ionp_id.py`** — the magnetic-ID barcode: codeword properties and stripe construction.

---

## 6. Gotchas

1. **`ionp` standalone needs its input DXF to already exist** at
   `probe_designs_output/electrode_bundle_IONP_id{NN}.dxf` — it stamps wells onto that file in
   place. `all`
   sidesteps this entirely; prefer it unless you specifically want to re-stamp an existing design.
2. **Run from the parent of `electrode_bundle/`**, not from inside it, and prefer the **module**
   form `python3 -m electrode_bundle.<name>`. Every module here uses relative imports
   (`from .bundle import ...`), so running one **by file path** normally fails with
   `ImportError: attempted relative import with no known parent package` — a path run has no
   parent package for the leading `.` to resolve against. `id_legend.py` carries a small
   `__package__` bootstrap so it works either way (it is the one module people reach for
   standalone); the others do not, and don't need to — `main.py` is the entry point.
3. **`probe_designs_output/` is disposable.** Everything in it is regenerated from `config.py` +
   `design_sets.py`. If you want to keep a design, copy it out — the next run with the same name
   overwrites it. (`designs/` is the opposite: shipped inputs, don't delete.)
4. **`ref_dxf` is optional and off.** Setting `BundleConfig.ref_dxf` drives contact centres from
   real electrode centroids in a reference DXF instead of the parametric grid. Default `None`
   means no file is read anywhere in the `bundle`/`all` path — and note that any such reference
   DXF is **not** part of this folder; you would have to supply it.
5. `n_pads_per_column = 33` → two columns at `x = ±975`, **66 pads total**: each column is
   1 REF/GND + 32 connected, so 64 channels are connected and 2 pads are REF/GND.
6. **Units are µm** internally throughout.

