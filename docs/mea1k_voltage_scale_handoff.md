# MEA1K voltage scale: what we know, what we inferred, what is still open

**Status as of 2026-10-23.** Handoff for an agent picking up this investigation.
Everything below is tagged: **[DATA]** = measured and reproduced, **[INFERENCE]** =
a reading of the data that could be wrong, **[OPEN]** = not established.

---

## 1. The question

Driving a MEA1K stim unit in current mode through a known calibration resistor, the
voltage reported by the MEA1K recording amplifier divided by the current measured
externally comes out ~15% below the resistor value. The repo assumes
`ephys_constants.MEA_OVERRIDE_GAIN = 7`. Either that gain is wrong, or something
else in the chain is.

**Do not change `MEA_OVERRIDE_GAIN` on the strength of this document.** It scales
every MEA1K voltage in the codebase by ~16% and moves every impedance ever measured.
The cause is not yet identified. See §6.

---

## 2. The circuit

```
         MEA1K CHIP                              EXTERNAL
  ┌───────────────────────────────┐
  │   ┌─────────┐                 │
  │   │StimUnit │  I_stim         │
  │   │ current ├──────┐          │
  │   │  mode   │      │          │
  │   └─────────┘      ● ringnode │
  │              ┌─────┴─────┐    │
  │              │  routing  │    │
  │              └─────┬─────┘    │
  │              ┌─────┴─────┐    │
  │              │ rec. amp  │── gain 7 ──> V_mea1k   ③
  │              │ (1 Hz HP) │    │
  │              └─────┬─────┘    │
  │               V_ref ●         │   ← reference node. In the DRY experiments
  └─────────────────────┼─────────┘     the REF site is exposed to AIR.
                        │
     Chip MA pin ●──────┴──────●──┬──────────────┐
                                  │              │
                             ┌────┴────┐    ┌────┴────┐
                             │ R       │    │ pico chB│  ②
                             │ 1.024 M │    │  1 MΩ   │
                             └────┬────┘    │  ‖14 pF │
     Chip PA pin ●──────────────────●──┬────┴─────────┘
                                       │
                                 ┌─────┴─────┐
                                 │ R_sense   │──> ×100 ──> pico chA   ①
                                 │   10 kΩ   │
                                 └─────┬─────┘
                                      ═══ GND (earth / BNC shell)
```

Three independent measurements of the same node:
① current through `R_sense` (×100 amplifier), ② node voltage MA→GND directly,
③ the MEA1K amplifier. ② spans `R + R_sense` = 1.034 MΩ nominal.

**[DATA]** Pico ch B's 1 MΩ input sits in parallel with the 1.034 MΩ load and halves
the current. Measured fraction vs an unloaded run: 0.491; predicted
`1.0/(1.0+1.034)` = 0.492. **Absolute currents and LSBs from any run with ch B
attached are not usable.** Ratios are unaffected — both sides see the same loaded
circuit.

---

## 3. What is in the data

### 3.1 The current-sense chain is correct **[DATA]**

`V_pico / I_pico` across every run and every frequency: **1.021 – 1.038 MΩ** against
a nominal 1.034 MΩ. Within ~1%, which is the tolerance of `R_sense` and the ×100 gain.

This is the control that makes everything else meaningful: the two pico chains agree
with each other and with the resistor.

### 3.2 The MEA1K reads ~14% low, reproducibly **[DATA]**

`V_mea1k / V_pico`, four independent hookups (including a full disconnect/rewire and
one deliberately-shorted run in between):

| run | n | ratio |
|---|---|---|
| `2026-10-09_16.09_..._ChainProbeTest` | 8 | 0.8645 ± 0.0051 |
| `2026-10-09_16.18_..._ChainProbe2` | 8 | 0.8603 ± 0.0117 |
| `2026-10-09_21.45_..._ChainProbe` | 59/64 | 0.8615 (median) |
| `2026-10-10_13.25_..._ChainProbeFreqSweep` | 48 | 0.8486 @100 Hz → 0.8762 @2 kHz |

All under `/mnt/SpatialSequenceLearning/devices/well_devices/4983/recordings/`.

Cross-checked with a completely independent estimator (4th-order Butterworth
bandpass 940–1060 Hz, amplitude = √2·std, no lock-in, no FFT): **0.8610 ± 0.0112**
against the pipeline's 0.8615. The effect is not an artifact of the amplitude
estimator.

### 3.3 It is frequency-independent **[DATA]** — this killed the leading hypothesis

Frequency sweep, `2026-10-10_13.25_..._ChainProbeFreqSweep`, 4 stim units × 2
amplifiers × 6 frequencies:

| freq | ratio | sd | Z_pico (MΩ) | drive |
|---|---|---|---|---|
| 100 Hz | 0.8486 | 0.0117 | 1.0208 | 100% |
| 200 Hz | 0.8488 | 0.0105 | 1.0211 | 100% |
| 500 Hz | 0.8505 | 0.0101 | 1.0222 | 92% |
| 1 kHz | 0.8569 | 0.0117 | 1.0259 | 79% |
| 2 kHz | 0.8762 | 0.0232 | 1.0383 | 58% |
| ~~3 kHz~~ | ~~0.9408~~ | ~~0.60~~ | — | 22% |

An RC pole at 1.7 kHz (the value that would explain a 14% loss at 1 kHz) predicts
**0.998 at 100 Hz** — a 15% swing across that decade. Measured swing: **1%, in the
opposite direction.** Plot: `processed/frequency_response.png` in that folder.

Ignore the 3 kHz row: the stimulus itself collapses to 22% there because a 3 kHz sine
at a 20 kS/s DAC update rate is only 6.7 samples/cycle with ±3 DAC codes. That is the
DAC, not the measurement — `Z_pico` stays correct throughout, which is the control.

### 3.4 Saline does not fix it **[DATA, user-reported run]**

`2026-10-05_08.57_R=1000000_CharStim_ShowCase_large_current` was **overwritten** on
2026-10-10 18:19 by a stim-unit run with saline on the chip (which connects the GND
and REF contact sites to the bath). The folder name is therefore misleading — its
`processed/large_current_lsb_characterization.csv` is the saline run, R = 1 024 000,
192 rows, 32 stim units.

`V_mea1k / I_pico` = **0.8556 MΩ** (median), i.e. 0.828–0.836 of nominal. The dry
equivalent was ~0.868 MΩ. **The shortfall survives grounding the reference.**

Caveat: that run has no pico ch B, so it compares `V_mea1k` against the *current*
chain, not the direct voltage chain. Not the same quantity as §3.2, but the same size.

### 3.5 The chip's own metadata says gain 7 **[DATA]**

In `data_store/data0000/settings/` (and `settings/` in the newer layout):
`gain = 7.0`, `lsb = 0.00046038 V`, `hpf = 1.0 Hz`, `sampling = 20000 Hz`.

`lsb` is exactly `(3300 mV / 1024) / 7`, so `MAX_AMPL_mV`, `ADC_RESOLUTION` and
`MEA_OVERRIDE_GAIN` in `ephys_constants.py` match MaxWell's convention exactly.
**[INFERENCE]** This does *not* independently confirm the physical volts — `lsb` is
almost certainly computed from the configured gain, not measured. It proves the repo
is consistent with MaxWell, not that either is right.

### 3.6 External sine in saline: linear and uniform, but no absolute gain **[DATA]**

Files: `~/Data/recordings/Trace_20261010_18_42_06_50mV_Gain7.raw.h5` and
`..._10mV_Gain7.raw.h5.raw.h5` (note the doubled extension on the second).
1012 routed channels, 1 kHz, gain 7, chip in saline, drive from a Pt electrode.

| generator setting | measured amplitude | measured/setting |
|---|---|---|
| 50 mV | 34.226 mV | 0.6845 |
| 10 mV | 6.833 mV | 0.6833 |

Three results that do **not** depend on knowing the applied voltage:

- **Linearity: 5.0090 measured vs 5.0000 applied (+0.18%)** over a 5× range. No
  compression, no saturation.
- **Channel-to-channel sd 5.3%** (IQR-based 3.35%) over 1012 amplifiers. Small,
  and incapable of producing a 15% systematic.
- **The bath is equipotential over the array.** A planar fit of amplitude vs
  electrode (x, y) gives a gradient of 0.279 mV/mm — a 3.34% swing across the full
  3.8 mm — and explains only **R² = 0.028** of the variance. So the 5.3% spread is
  amplifier gain variation, not field geometry, and all 1012 sites genuinely see the
  same voltage.

**[INFERENCE]** Because electrode impedances vary by tens of percent site to site and
the measured spread is only 5.3%, the amplifier input impedance must be ≫ the
electrode impedance (consistent with the ~113 MΩ figure the user quoted). The
electrode–electrolyte interface therefore attenuates by well under 1% and can be
dropped from the analysis.

### 3.7 The bank split is in the stim units, not the readout **[DATA]**

Stim units 0–15 deliver more current than 16–31 (median `I_pico` 0.0837 vs 0.0730 µA,
−13%), but `V_mea1k / V_pico` is identical across the two banks (0.8631 vs 0.8607).
The bank difference is a property of the current sources and is unrelated to the
voltage-scale question.

---

## 4. What is inference, not measurement

### 4.1 "The effective gain is 5.94"

`7 × 0.849 = 5.94`. This follows **only if** the pico chains are the truth and the
MEA1K is the thing that is wrong. The data supports the pico side (§3.1), but the
inference assumes there is nothing between the ch B tap at the Chip MA pin and the
amplifier that could cause this. **[OPEN]**

A note on the sign, because it constrains the possibilities: a *series* impedance
between the MA pin and the amplifier cannot produce this. The ringnode is upstream of
the pin, so a series element makes the amplifier read **higher**, not lower. Only two
topologies give the right sign:

1. the amplifier's actual voltage gain is ~5.94 rather than 7, or
2. the reference node moves *with* the signal: `V_ref ≈ 0.15 · V_node`, which needs a
   resistive coupling from the node to the reference.

§3.4 weakens (2) but does not kill it — the saline run has the caveat noted there.

### 4.2 The external-injection number cannot be converted to a gain

The saline recordings give `measured = (G_true/7) · α · V_generator`, where α is the
bath divider: Pt electrode interface, spreading resistance, GND site impedance, REF
position. **α is not measured by anything in those files.** 0 < α ≤ 1.

- α = 0.685 → gain 7
- α = 0.807 → gain 5.94 (agreeing with the stim result)
- α = 1.00 → gain 4.79

All three are physically plausible. **The saline recordings cannot choose between
them.** Any statement of the form "the external test says the gain is X" is wrong.

**[DATA, logical constraint]** One branch *is* excluded. A function generator's
amplitude setting assumes a 50 Ω load; into the effectively open circuit of a saline
bath it delivers 2× the setting. The user reports using the Vpp setting. If "50 mVpp"
meant 25 mV amplitude at the electrode, then since α ≤ 1 the measured 34.226 mV
exceeds the applied — impossible for gain 7, forcing gain ≥ 9.58. So either the
generator was in HighZ mode (output ≈ 50 mV amplitude), or the gain is above 7.
**[OPEN]** Nobody has checked the generator's HighZ/50 Ω setting.

### 4.3 Phase tells us nothing — do not use it

`read_stim_DAC(..., adjust_delay=True)` (the **default**) shifts the DAC trace 2
samples later. At 20 kS/s that is exactly **36° at 1 kHz**. The empirical correction
was found on a *different PCB* with the same chip.

The pipeline reported 28.17° ± 0.08° at 1 kHz, which looked like a beautiful match to
the 30.6° an RC pole would need. Backing out the fudge gives a **raw lag of −7.83°**,
and raw lags per frequency of −2.2° (100 Hz), −1.3° (200), −2.2° (500), −7.9° (1000),
−32° (2000), −66° (3000).

The raw phase still contains unknown chip and ADC pipeline latency, and a pure delay
of 1.56 samples produces 28° at 1 kHz with no attenuation at all. **Absolute phase is
not evidence for or against any hypothesis here.** `probe_measurement_chains.py` now
records `phase_shift_raw_deg` alongside the fudged value.

---

## 5. Not explained

- **[OPEN] The cause of the 0.85.** Frequency-independent, consistent across 64
  amplifier/stim-unit combinations (sd ~0.011), linear, survives saline. Not a filter,
  not a noise artifact, not a per-amplifier gain spread, not the current chain.
- **[OPEN] Why the ratio drifts upward with frequency** (0.8486 → 0.8762 from 100 Hz
  to 2 kHz). Small, monotonic, opposite to any lowpass. Could be a capacitance
  shunting a series element; could be the drive collapse biasing the 2 kHz point.
- **[OPEN] 5 of 64 recordings in the 21.45 run are outliers** (stim units 4, 8, 11,
  18) where *both* pico channels drop to ~30% while the MEA1K stays normal and
  `Z_pico` stays at 1.02. Pattern suggests a mistimed pico window, not a chain error.
  They poison the mean — **use medians when summarising that run.** Not diagnosed.
- **[OPEN] The bath-probe measurement (the experiment that would settle it) has not
  produced usable data yet.** See §7.

---

## 6. Recommendation on `MEA_OVERRIDE_GAIN`

**Leave it at 7.** Reasons:

- The cause is unidentified, and the two candidate topologies (§4.1) imply different
  corrections — a gain error scales everything, a reference-coupling error may not
  apply when the reference is wired differently (e.g. in a real saline experiment).
- 5.94 is suspiciously close to 6, which smells like a mis-set or mis-documented
  register rather than an arbitrary calibration offset. Worth checking what gain the
  headstage is actually configured for before encoding a fudge factor.
- Every impedance ever measured with this codebase moves by 16% if it changes.

If you need currents *now*: `I = V_mea1k / (0.849 · R)` is validated flat from 100 Hz
to 2 kHz. Treat 0.849 as an empirical constant of unknown origin, not a gain.

---

## 7. The open experiment

**Measure the bath potential at the array directly, simultaneously with the MEA1K.**
This removes α entirely: generator amplitude, HighZ setting and bath geometry all
cancel, and you get `implied gain = 7 · V_mea1k / V_bath` with no assumption.

Script: `probe_saline_gain.py` (`record` / `eval` modes). Wiring and procedure are in
its docstring. Key points:

- ch B tip on a wire in the saline just above the array; **ground clip on the chip's
  circuit ground**, not the bath and not REF — the BNC shell is earth ground and
  clipping it elsewhere shorts that node.
- the generator must run continuously and untouched across both recordings
- no synchronisation needed: it is steady state, so the two recordings only have to
  see the same conditions, not the same samples

### Status of the first attempt — inconclusive

`~/Data/recordings/bath_probe_2026-10-12_16.53.01.npz`: a clean 1 kHz line at
**0.966 mV** (31.9% of post-settle power; 50.6% of the power is sub-5 Hz drift). Real
signal, but far too small to be the array potential if the generator was anywhere near
its earlier setting.

Three diagnostics, in order, each a 5 s recording:

1. **Lift the wire out of the saline.** If 1 kHz survives, it is stray capacitive
   pickup and the probe never measured the bath. Fastest and most decisive.
2. **Immerse much more wire.** Ch B's input is 1 MΩ in series with the wire–saline
   interface; a nearly-dry or insulated tip loads it badly. If the amplitude moves,
   that was the cause.
3. **Tip on the generator lead.** Gives the drive amplitude, hence α directly.

Arithmetic worth knowing for #3: if the bath really is ~1 mV and α ≈ 0.02, then for
the Oct-10 trace to have read 34 mV the generator would have had to output ~3.4 V.
That is not a "50 mV" setting, so either the generator is set much lower now or the
probe is not coupled.

Then: **record a fresh GUI trace while the probe script runs.** Nothing from Oct 10
can substitute — see §8.

---

## 8. Analysis traps already hit (do not repeat these)

Each of these produced a confident wrong answer during this investigation.

1. **FFT scalloping.** The pico and the MEA1K have independent clocks (~1400 ppm
   apart), so their peaks land on different FFT bins. A single-bin Hann amplitude is
   biased up to 15% low when the peak falls between bins — exactly the size of the
   effect under investigation. Use a bandpass + √2·std, or integrate power over a band.
2. **n-dependent normalisation.** The two devices give different window lengths
   (7999 vs 7209 samples). A band-power estimator with a √n bug shifted the ratio from
   0.856 to 0.903 while leaving the pico/pico ratio untouched — which is the tell.
3. **The 2-sample DAC shift.** See §4.3.
4. **Hard-coded filters in the eval.** Before the fix, `_eval_probe_file` used
   `highpass=100`, `lowpass=3000` and `max_gap=20` regardless of frequency. On a
   sweep that reports a ratio of **1.71 at 100 Hz and 1.41 at 3 kHz** — a fabricated
   frequency dependence. Now scaled per frequency; verified against synthetic sines at
   every frequency (0.998–1.000 flat).
5. **Pairing recordings across sessions.** `probe_saline_gain.py eval` originally
   picked the newest npz and silently matched it to a 46-hour-old h5, printing an
   implied gain of 248. It now refuses pairs more than `MAX_PAIR_AGE_S` apart.
6. **npz/h5 pairing within a session.** MaxWell appends `_0`/`_1` to h5 names on
   clash while the npz is overwritten, so the newest npz can pair with the oldest h5.
   Handled by a regex suffix strip plus a 10 s mtime window.
7. **Reading the channel from a constant.** An npz records which channel it used
   (`channels`); do not trust a module constant that may have been edited since.
8. **Disk.** MaxWell keeps writing (~40 MB/s in legacy mode) until `stop_saving`. An
   exception between `start_saving` and `stop_saving` once left a 17.9 GB orphan at
   `2026-10-09_15.57_R=1024000_ChainProbe_large_current/`. That folder is one invalid
   recording and can be deleted. `_record_sine` is now wrapped in try/finally.

---

## 9. Code map

| file | role |
|---|---|
| `probe_measurement_chains.py` | the three-chain probe: record + eval + frequency sweep. `main()` holds the sweep config. |
| `probe_saline_gain.py` | bath-probe gain measurement (§7). `record` / `eval`. |
| `characterize_mea1k_stimulaters.py` | stim-unit characterisation; `_record_sine`, `_sine_onset_offset` are imported by the probe. |
| `mea1k_connectivity_scripts/signal_helpers.py` | `extract_amplitude` — fixed-reference lock-in, `lp_cutoff = f/10`. |
| `mea1k_modules/mea1k_raw_preproc.py` | `read_raw_data`, `read_stim_DAC` (**`adjust_delay=True` by default — see §4.3**). |
| `PicoRecorder.py` | multi-channel streaming. 16BIT is 1 channel, 15BIT is 2 (A/B only). |
| `bonding_analysis/check_DAC_calibr.ipynb` | figures; last section is the external-sine gain analysis (§3.6). |

### Environment

- Runs need MaxLab's bundled Python: `~/MaxLab/python/bin/python3` with
  `PYTHONPATH=/home/houmanjava/MaxLab/share/maxlab_lib`. It is the only interpreter on
  the machine with both `maxlab` and `picosdk`.
- MaxLab GUI traces need `HDF5_PLUGIN_PATH=/usr/local/hdf5/lib/plugin` (MaxWell's
  compression filter) or h5py fails with `can't open directory .../hdf5/plugin`.
- Offline analysis works in the base conda python with `maxlab`/`picosdk` stubbed via
  `unittest.mock.MagicMock` — but `picosdk.functions` needs a real module object with
  `adc2mV` and `assert_pico_ok` attributes, not a bare MagicMock.
- Two h5 layouts coexist: `sig` (newer) and `data_store/data0000/groups/routed/raw`
  (legacy / GUI). Some files carry `settings/*`, some do not.
