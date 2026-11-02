# Device context: MEA1K ↔ polyimide interconnect

Facts only. Every line is tagged with where it came from. **Nothing in this file is my own
inference** — predictions, design analysis and open problems live in
`device_open_questions.md`, and must not be treated as established.

Provenance tags:
`[mea1k-paper]` `[flex2chip]` `[thesis]` `[ufte-paper]` `[mt-specs]` `[mt-leaflet]`
`[maxwell]` (correspondence w/ F. Cardes) · `[code]` (our repo) · `[simon]` · `[derived]`
(arithmetic from tagged facts — checkable, but check it).
`[conflict]` = sources disagree, unresolved.

> **The CMOS side is fixed hardware; the polyimide side is not.** Many PI devices are in
> circulation and they differ in pad shape, pad count, routing and shank count. §3 gives
> examples of specific past designs to show the ranges — none of them is a default. Where a
> conclusion turns on PI-side geometry, check which device is meant rather than assuming.

## 0. Terminology (not interchangeable)

| Term | Means |
|---|---|
| **MEA1K electrode** | one of the 26,400 Pt sites on the CMOS array |
| **pad** | a contact site on the polyimide interconnect, chip-facing side |
| **PEDOT electrode** | a brain-facing recording site on the PI device |

Sources violate this: `[mea1k-paper]` says "pixel", `[thesis]` says "MEA1k pads" for chip
electrodes and "contact pads" for brain-facing sites. Translate when quoting. `[simon]`

---

## 1. CMOS array geometry

- Active array **3.85 mm (x) × 2.10 mm (y)**, **26,400** electrodes, **17.5 µm pitch** both
  axes, 3265 el/mm². `[mea1k-paper]`
- **220 columns × 120 rows**; **`el = y·220 + x`, origin top-left**. `[derived]` `[simon]`
- Electrode opening **9.3 × 5.45 µm² platinum** — not square. `[mea1k-paper]`
- Gaps between openings: **8.2 µm along x**, **12.05 µm along y**. `[derived]`
- Electrodes recessed below an SiO₂/Si₃N₄ passivation stack: **1.6 µm** `[thesis]` vs
  **1.2 µm** `[flex2chip]`. `[conflict]` — never measured on our chips. `[simon]`
- Die 10.1 × 7.6 mm². Readout channels above and below the array; **32 stimulation units on
  the left and right**. `[mea1k-paper]`
- Wirebonds are encapsulated in epoxy forming a raised wall around the array. We shave it
  back; current clearance from the array edge **~2–3 mm**. `[thesis]` `[simon]`

## 2. CMOS circuits

### 2.1 Switch matrix
- 2 SRAM cells + 2 switches under each electrode; **86,000 switches / 59,000 SRAM cells**
  array-wide; 6 horizontal + 6 vertical wires per row/column; routing wires average 420 µm.
  `[mea1k-paper]`
- **1024 of 26,400** electrodes readable at once, and *not any arbitrary 1024* — routing is
  solved as an ILP max-flow/min-cost problem and requested electrodes can silently fail.
  Always compare `array.get_config().mappings` against what you asked for.
  `[mea1k-paper]` `[code: try_routing/setup_array]`
- Largest contiguous block: 23 × 23 electrodes (≈402 × 402 µm). `[mea1k-paper]`

### 2.2 Readout
- 3 amplification stages, programmable total gain to **78 dB**; on-chip **10-bit ADC at
  20 kHz**; input-referred noise 2.4 µV_rms (300 Hz–10 kHz). `[mea1k-paper]`
- **`set_gain()` affects the recording path only** (V/V). It is unrelated to the `*_LSB_nA`
  values in stim characterisation, which are stimulation current LSBs. `[simon]`
- **Recording LSB = (3.3 V / 1024) / gain = 3.22 mV / gain**, input-referred.
  At `gain=7` → ≈460 µV/LSB (±235 mV full scale). `[simon]` `[derived]`
- DAC codes are 10-bit; **512 = mid-supply ≈ 1.65 V**. Codes are not volts at the electrode.
  `[code]`

### 2.3 Stimulation units
- **32 units**; each selectable **voltage / current** mode, current mode has **small** and
  **large** range. `[code: turn_on_stimulation_units]`
- **A stim unit is not freely assignable.** `connect_electrode_to_stimulation()` returns
  whichever unit the matrix can reach, one per electrode, and can fail. Configs are
  pre-generated per unit under
  `mea1k_configs/single_el2stimunit_configs2/StimUnit{NN}/`. `[code]`
- **Units differ from each other and must be calibrated.** Measured (1 MΩ or 100 kΩ load at
  PI, 1 kHz sine): small_current **5.1–6.1 nA/LSB**, large_current **42–51 nA/LSB**,
  zero-current DAC code **412–798** (not 512). Units 0–15 and 16–31 form two distinct
  populations. Per-unit LSB and zero-current code are applied so delivered current is equal
  across units. **Never use a nominal LSB.** `[code]` `[simon]`
- Architecture note: the successor chip's stim buffer is a CCII+ current conveyor with a
  current-mirror gain switch (~1 → ~11 A/A) for the range change. **Different chip** — do
  not assume it transfers. `[cmos_phd_thesis]`

### 2.4 Nodes not in the public API docs
- **Ring node**: `array.connect_amplifier_to_ringnode(ch)`, `query_amplifier_at_ringnode()`,
  `disconnect_amplifier_from_ringnode()`. `[maxwell]`
- **External port = "PI" pin**: `maxlab.chip.Core().use_external_port(True)`. Amplifier →
  ring node → PI puts an external load in the stim current path. Touching the PI pin must
  visibly raise noise on that amplifier trace. **PI is broken out on the in-vivo headstage
  too**, so recalibration on an assembled headstage is possible. `[maxwell]` `[simon]`
- **"MA"** is the return for a PI–MA load, and is **AC ground at mid-supply (~1.65 V)**, not
  chip GND. `[simon — from memory, not verified]`
- Ring/reference electrodes are held at VRefVari (mid-supply), off the electrode bus;
  `maxlab.util.initialize()` does this. `[code]`
- Low-level only (no MaxLab API): `cmdVRefVari`, `cmdSw1/Sw2`, `cmdProbePads`, raw HEX load.
  `ptblack_full.hex.nrk` shorts **all 26,400 electrodes to one bus** — in that mode you are
  never touching a few electrodes, you are touching all of them in parallel. `[code]`

---

## 3. Polyimide devices (variable)

### 3.1 What varies between devices
`[simon]` Pad shape, pad count and routing all differ between PI devices. There are tethered
devices integrating 6 or 8 shanks, debug devices that short all pads to one contact side,
and a newer generation using **plain rectangular pads** rather than spring pads.

The device-dependent quantities:
- pad shape — spring-type suspended microstructure vs plain rectangle — and its dimensions
- pad count, pitch, and footprint relative to the 3.85 × 2.10 mm array
- routing: metal layers, trace width/spacing, pads individually routed or shorted together
- shanks: count, PEDOT electrodes per shank and per fiber, spacing between them
- PEDOT electrode size, coating and impedance

### 3.2 Common across devices
- Ti/Au (typ. 10 nm / 150 nm) between polyimide (PI2610) layers, ~1.2 µm per layer; one
  metal layer, or two where routing density demands it (wire pitch ~5 µm → ~2.5 µm).
  `[ufte-paper]` `[thesis]`
- Fiber cross-section 7 µm × **2.4 µm** (1 metal) or 7 µm × **3.6 µm** (2 metal).
  `[ufte-paper]` `[thesis]`
- **PEDOT electrodes** ~13 × 13 µm², PEDOT:PSS 450 nm in 3 spin/bake cycles. |Z| @1 kHz
  **54 ± 16 kΩ** (Intan-era UFTE) vs **284 ± 75 kΩ** (thesis, MEA1K-era) — different
  measurement methods, not comparable. `[ufte-paper]` `[thesis]` `[conflict]`
- Released by peeling under water, which self-assembles fibers into bundles. One fiber per
  bundle ends in a ~25 µm ID loop hooking a 50 µm tungsten shuttle (tip thinned to 20 µm);
  others overrun by ~500 µm. PEG4000 + silk fibroin hold the bundle during insertion.
  `[ufte-paper]`

### 3.3 Example past designs (**not** current)
- **Flex2Chip** `[flex2chip]`: 35 µm ⌀ pad on three beams 2 µm × 10 µm, 1 µm thick, 1 µm
  above the electrode; 50 µm pad pitch; 2200 channels, 272 ch/mm².
- **Thesis Gen-1** `[thesis]`: 50 µm ⌀ pads, ≥22 µm gap, **1278 pads** over 2.1 × 3.85 mm
  (158 ch/mm²), ~15 % over the 1024 budget as failure margin; 4 shanks, 1160 PEDOT
  electrodes / 580 pairs, dual pads per fiber 20/30/40 µm apart. Connector block 4342 µm
  then 5640 µm wide — both wider than the 3850 µm array.
- **Design rule that does generalise:** pad spacing must exceed the 17.5 µm MEA1K electrode
  pitch — not only against shorts, but because connectivity clusters must stay separable.
  Cluster separability *is* the mapping method (§5). `[thesis]`

---

## 4. Bonding

### 4.1 Gold plating the MEA1K electrodes
- Cyanide-based 24K solution, 62–71 °C (IR closed loop), Pt counter electrodes centred and
  distant, MEA1K electrodes at **−1.15 V** vs reference (to avoid plating the ring
  electrodes), ~20 µA over 1.3381 mm² (vendor recommends 31 µA/mm²), ~10 min per µm,
  ~30 min total. `[thesis]`
- **Pillar height: 2–3.5 µm with real variability.** The thesis' 1.83 ± 0.18 µm
  profilometry is not reliable — treat as a wide margin, not a number. `[simon]`

### 4.2 Spring-pad collapse (spring-pad devices only)
- Assembly is self-assembly, not bonding: thin IPA film, drag into rough alignment with
  paintbrushes, let IPA evaporate. Capillary force pulls each pad down; van der Waals holds
  it. `[flex2chip]` `[thesis]`
- Measured: pad displaces 1 µm, deformation shared across the three beams. Contact is
  **ohmic, 66.5 ± 12.9 Ω, phase 0.00° @10 Hz and 0.01° @1 kHz** — so measured phase shift
  in vivo is electrode–electrolyte, not the bond. `[flex2chip]`
- Yield, Gen-1 in vitro: **843 / 1278 pads = 72.4 %** at 0.8 connectivity, no cross-cluster
  shorts, stable ≥18 days with slight improvement as residual IPA evaporates. Losses
  attributed to misalignment and the epoxy ridge lifting the periphery. `[thesis]`
- Held by a custom **titanium holder with an Ecoflex pad** between holder and device.
  `[thesis]`

### 4.3 Metal1 vs metal2 pad response to pressure (spring-pad devices)
Two-metal-layer spring-pad devices (§3.2) show a consistent, reproducible difference between
metal1 and metal2 pad populations across pressure sweeps, observed on MEA1K23 and MEA1K24.
`[simon]`

- **metal1** (fabricated first, ~1 µm farther from the CMOS surface after flip, thinner ~1 µm
  flap): forms an initial connection more readily — rarely connects at zero pressure, but
  engages under light pressure. Connectivity is bimodal: **~0.8** (tens-of-MΩ contact,
  relative to the 112 MΩ amplifier input impedance) or **~0.95** (few Ω to 1–5 MΩ contact).
  Pads that land at 0.8 tend to *stay* at 0.8 as pressure increases rather than moving to
  0.95 — roughly half the pads plateau this way, and the fraction varies strongly by chip
  region (e.g. MEA1K23 bottom-left corner good, opposite side poor).
- **metal2** (closer to CMOS, thicker flap): does not connect at low pressure — needs more
  pressure to engage at all — but once pressure is sufficient, a larger fraction of engaged
  pads reach the >0.8 band than for metal1.
- **Shared pressure-response pattern, both metals**: a pad already at 0.95 connectivity but
  still high impedance (1–5 MΩ) can be driven to a genuinely low-impedance bond by more
  pressure (a "pressure limit" effect, distinct from raising connectivity itself). More
  pressure also grows already-connected clusters (more electrodes per pad) before it engages
  new pads.
- On one MEA1K24 device, metal1 pads reached ~0.95 on both halves of the chip, while metal2
  pads reached only ~0.8 on one half and ~0.95 on the other — a spatial asymmetry specific to
  metal2 on that device.

**Root cause not established** — of both the metal1/metal2 difference and the spatial
(per-chip-half) variation within a metal. Two candidate explanations, not disambiguated
without systematic profilometry: (a) etch quality — metal1's shallower etch pit may free the
spring flap more reliably, so the difference is about which pads can *mechanically move*; (b)
gold pillar height/cleanliness variation across the chip, independent of the PI side. Do not
treat either as settled. See `device_open_questions.md` §D. `[simon]`

### 4.4 Shin-Etsu MT anisotropic connector (next generation)
Gold-plated metal wires in silicone rubber on a **regular matrix lattice**, sold as sheet.
Wires run at an angle, so the contact on one face is displaced from the other by
**Off-set = T/2**. Two lattice pitches Pi × Ps. Wire tips protrude from both faces.
`[mt-specs]` `[mt-leaflet]`

| Variant | Wire ⌀ | Pi × Ps | Sheet L×W | T | Off-set | Hardness | Protrusion |
|---|---|---|---|---|---|---|---|
| MT-T-4X(-E) | 23 µm BeCu | 50 × 50 µm | 5–50 mm | 0.15 / 0.25 mm | 75 / 125 µm | 70 | 15 µm both sides |
| MT-T-6X-E | 17 µm BeCu | 32 × 50 µm | 5–50 mm | 0.15 / 0.25 mm | 75 / 125 µm | 70 | 15 µm both sides |
| MT-T-9X *(TBD)* | 12 µm brass | 30 × 30 µm | **10 × 10 mm** | 0.15 / 0.25 mm | 75 / 125 µm | 70 | 15 µm both sides |
| MT-P-4X | 23 µm + Au tip | 50 × 50 µm | 5–50 mm | 0.15 / 0.25 mm | 75 / 125 µm | 70 | 20 µm both sides |
| MT-T-LR-4X | low-resistivity | 50 × 50 µm | 5–20 mm | 0.15 / 0.25 mm | 75 / 125 µm | 70 | 15 µm both sides |

Thick `MT` families exist at T = 0.5/1.0/2.0 mm, hardness 30 — offsets of 250–1000 µm.

- Wire tilt = arctan(0.5) = **26.6° from normal**. `[derived]`
- Connection behaviour (MT-T-4X-E, ⌀0.5 mm Au pin vs 10 × 20 mm Au electrode, 4-terminal):
  resistance ~140 mΩ uncompressed → **<20 mΩ by ~20–30 µm compression** → 5–10 mΩ asymptote.
  Load rises roughly linearly to ~2.5–3 N per ⌀0.5 mm contact at 80 µm compression.
  `[mt-leaflet]`
- Durability: daisy-chain resistance ~0.1–1 Ω over 90,000 compressions at 20 N load.
  **Repeated cycling only — sustained-hold compression set is not characterised.**
  `[mt-leaflet]`

---

## 5. Measurement loop

- **Connectivity map** (the mapping primitive): shanks in saline, 5 mV / 1 kHz sine on an
  external electrode, every one of the 26,400 MEA1K electrodes records ~1 s, power at
  995–1005 Hz normalised to the injected signal → **connectivity ∈ [0,1]**. Phase separates
  ohmic (0°) from capacitive (90°). Cluster pattern-matching against the known pad layout
  assigns MEA1K electrode → pad → fiber → shank → depth → region. `[thesis]` `[code]`
- **Impedance**: per stim unit, set the calibrated zero-current DAC code, build a 1 kHz sine
  around it, connect electrode → stim unit, record.
  `Z = amplitude_uV / (LSB_nA · dac_sine_amplitude / 1000)`. Both factors per-unit
  calibrated. `[code: record_imp.py]`
- **Extraction**: 50 Hz notch → lock-in I/Q demodulation at the reference → causal low-pass
  at f_ref/10, discarding 2 time constants of settling. `[code: signal_helpers.py]`
- Working bands in the maps: <20 kΩ suspiciously low, 40–140 kΩ good, 140–340 kΩ acceptable,
  >340 kΩ bad, >800 kΩ dead. `[code]`
- Protocol cautions: allow ~0.5 s for stim units to wake (internal capacitances);
  characterise at the exact frequency and settings used later; a scope on PI changes the PI
  node impedance. `[maxwell]`
- **Lateral shorts are established by measurement, not geometry**: stimulate one electrode,
  record its neighbours, call a short when neighbours see the stimulus (tile-shorts /
  union-find). `[code]` `[simon]`
- Data tree: `devices/headstage_devices/{name}` (chip + PCB) vs
  `devices/implant_devices/{YYMMDD}_{headstage}_{design}` (a specific bond). Ground truth
  per bond in `.../bonding/bonding_mapping_<implant>.csv` (`mea1k_el`,
  `mea1k_connectivity`, `pad_id`, ranks, later `impedance_Ohm`). `[code]`
