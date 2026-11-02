# Open questions and untested hypotheses

**Everything here is unverified.** These are predictions, arithmetic consequences and design
arguments — mine unless tagged otherwise. None of it has been measured on our hardware.
Do not cite anything in this file as a fact; the facts are in `device_context.md`.

Each item says what would settle it.

---

## A. Chip side

**A1. Overplating should bridge along x before y.** The x-gap between MEA1K electrodes is
8.2 µm and the y-gap 12.05 µm, so gold bridging should be anisotropic.
*Settle by:* checking whether observed shorts in the tile-shorts data are directional.
*Stakes:* low — shorts are detected empirically either way. Only useful as a plating
diagnostic.

**A2. Stim units 0–15 / 16–31 are the left and right banks.** The two LSB populations match
the physical split of 32 units across the two sides of the array.
*Settle by:* asking MaxWell, or checking whether unit ID correlates with the x-position of
its reachable electrodes.
*Stakes:* moderate — if true, a left–right gradient in an impedance map could be
instrumental rather than biological, on top of the per-unit LSB correction.

**A3. `gain=7` is chosen because stim transients would clip at higher gain.** 460 µV/LSB
gives ±235 mV full scale, and the observed transients run 25–200 mV.
*Settle by:* Simon confirming the intent. The arithmetic is solid; the motive is inferred.
*Stakes:* low.

---

## B. MT anisotropic connector

This is the live design decision, so these matter most.

**B1. Gold plating is still required — the wires cannot reach into the recess.**
A 23 µm (or even the 9X's 12 µm) wire tip cannot enter a 9.3 × 5.45 µm electrode opening; it
should bridge the well and rest on the passivation. The 15 µm protrusion would not help,
because the obstacle is lateral, not vertical. If so, the plating requirement is unchanged
or tighter: bring Pt to or above the passivation plane so a compliant tip has a surface to
press on.
*Settle by:* a bench test on a plated and an unplated chip with the same film and load.
*Stakes:* high — it decides whether the gold-plating step survives.

**B2. The T/2 offset shifts every connectivity cluster by 75 µm (T = 0.15) or 125 µm
(T = 0.25).** That is 4.3 or 7.1 MEA1K electrode pitches, uniform and in the wire-tilt
direction. Consequences if true: the PI device must be placed deliberately biased against
that direction; the sheet's tilt axis must be tracked when cutting; ~75–125 µm of usable
overlap is lost at one edge.
*Settle by:* first bonding trial — the shift is directly visible in the connectivity map.
*Stakes:* moderate. Harmless if anticipated, confusing if not.

**B3. One wire will contact 2–4 adjacent MEA1K electrodes.** A 23 µm tip is wider than the
17.5 µm pitch. Two consequences: connectivity clusters become small and wire-shaped rather
than pad-shaped, and **the tile-shorts test will flag those electrodes as shorted —
correctly, by design.** Only inter-*pad* shorts would be defects.
*Settle by:* first bonding trial.
*Stakes:* moderate — mainly a trap for the analysis loop if unanticipated.

**B4. Pad sizing rule.** The lattice is regular, so coverage can be guaranteed rather than
left to chance: an axis-aligned rectangle with sides ≥ Pi and ≥ Ps always contains a lattice
point. Requiring an inter-pad gap ≥ the wire diameter so one wire cannot bridge two pads:

| Film | min pad | min gap | pitch | pads on 3850 × 2100 µm |
|---|---|---|---|---|
| MT-T-4X-E (50 × 50, ⌀23) | 50 × 50 µm | 23 µm | 73 µm | ~52 × 28 = **1456** |
| MT-T-6X-E (32 × 50, ⌀17) | 32 × 50 µm | 17 µm | 49 × 67 µm | ~78 × 31 = **2418** |
| MT-T-9X (30 × 30, ⌀12) | 30 × 30 µm | 12 µm | 42 µm | ~91 × 50 = **4550** |

All clear the 1024-channel budget. Rotation tolerance: a pad of side *s* still guarantees a
wire while `s ≥ p(cos θ + sin θ)`, so a 60 µm pad on the 50 µm 4X lattice tolerates ~10°.
Under *uncontrolled* rotation you need `p√2 ≈ 71 µm` pads at 94 µm pitch → ~880 pads, below
budget — so registering the sheet's rotation is worth roughly 600 channels on the 4X film.
*Settle by:* this is arithmetic on datasheet numbers, so it is as good as those numbers; the
assumptions to check are (a) the lattice really is regular and untapered across a sheet,
(b) lattice pitch tolerance, (c) that a gap equal to the wire diameter is actually enough.
*Stakes:* high — it sets pad geometry for the next mask.

**B5. Clamping force is the real problem, and does not scale from the datasheet.**
Contact resistance is milliohms, negligible against a 50–300 kΩ PEDOT electrode — the
interface is mechanical, not electrical. But the quoted load is per ⌀0.5 mm pin pressed into
a large sheet, where lateral constraint and shear contribute differently than they would
over our 8.09 mm² array. Naive area scaling lands at order 10–40 N. The die tolerates that;
the titanium holder, epoxy ring and wirebonds may not.
*Settle by:* bench measurement — compress a film against a dummy die and measure force vs
compression vs contact resistance directly. Do not extrapolate the datasheet curve.
*Stakes:* high, and blocking.

**B6. Compression set over months is uncharacterised.** The datasheet tests repeated
cycling (90,000 compressions at 20 N), which is the opposite duty cycle from a chronic
implant holding one static load.
*Settle by:* asking Shin-Etsu directly.
*Stakes:* high — this is the main chronic-viability unknown.

**B7. Variant choice is open.** MT-T-9X (30 × 30 µm, 12 µm wire) fits this array best but is
marked TBD and only comes in 10 × 10 mm sheets. MT-T-6X-E is available and sufficient.
*Settle by:* asking Shin-Etsu about 9X availability and lead time.

**B8. Biocompatibility and saline stability** of the silicone and the BeCu / brass wire at
the implant site — not addressed in the leaflet at all.

---

## C. Unresolved facts (would move to `device_context.md` once answered)

- **Passivation recess: 1.2 µm or 1.6 µm?** Sources disagree, never measured on our chips.
  Sets the minimum gold-pillar height. Low urgency given the 2–3.5 µm pillar margin.
- **What MA is tied to.** Recorded as AC ground at mid-supply on Simon's recollection;
  unverified. Changes the assumed stim return path.
- **Typical connectivity-cluster size per pad** on current spring-pad devices — needed as
  the baseline against which the (predicted smaller, wire-shaped) MT signature is judged.
- **Whether a stim unit's reachable electrode set has a spatial footprint** — see A2.

---

## D. Spring-pad bonding mechanism

**D1. Metal1/metal2 pressure-response difference: etching or gold pillars?** Metal1 pads
engage at lower pressure than metal2, and roughly half of engaged pads plateau at ~0.8
connectivity instead of reaching ~0.95, with strong spatial variation across the chip (see
`device_context.md` §4.3). Two explanations, not disambiguated: (a) etch depth/quality
controls whether a pad's spring flap can move at all — metal1's shallower etch pit etches
more reliably; (b) gold pillar height/cleanliness varies across the chip, independent of the
PI side. Simon currently leans toward (b) for the spatial variation but has not ruled out
(a), and the two are conflated in the current data.
*Settle by:* systematic profilometry — of both the PI flap deflection/etch depth and the gold
pillar heights — correlated against the connectivity map for the same region. Neither has
been measured systematically so far.
*Stakes:* moderate–high for the current spring-pad devices, and it also determines whether
the finding generalizes to the next design: metal2's "badly etched" pads structurally
resemble the no-spring design being planned next, and that configuration has only worked
well once so far (MEA1K23 4+4 device, which still had the etching issue).
