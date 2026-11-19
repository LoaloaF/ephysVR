# %%
"""Is the chip-vs-external impedance mismatch on MEA1K27 a current-source error?

Row 1  The external (picoscope) current in all_impedance.csv was fitted in a fixed sample
       window, but the pico is not synchronized to the MaxWell recording: the sine often
       started late and was partly cut off. Here the external current is refitted in a window
       detected on the pico trace itself (signal_helpers.detect_sine_window, now also used by
       record_imp). Cached to processed/ext_current_detected_window.csv.
Row 2  With the refit, where does the spread come from? Within-stim-unit deviations of the
       LSB calibration (1 MOhm, voltage only), the current ratio, chip and external impedance,
       and their run-to-run reproducibility (small- vs large-current run).
Row 3  Per stim unit: residual LSB error, LSB calibration stability, global offset.
"""
import os
import sys
import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..'))        # ephysVR
sys.path.insert(0, os.path.join(_here, '..', '..'))  # baseVR
from baseVR.base_functionality import device_paths
from mea1k_connectivity_scripts.signal_helpers import extract_amplitude, detect_sine_window

HEADSTAGE = "MEA1K27"
REC_ROOT = os.path.join(device_paths()[0], "devices/headstage_devices", HEADSTAGE, "recordings")
CALIB = {
    'small_current': "2026-11-05_17.59_R=1000000_CharStim_small_current",
    'large_current': "2026-11-05_17.34_R=1000000_CharStim_large_current",
}
# older independent calibration (R = 1.024 MOhm, 6 amplifiers per stim unit, large current).
# It and 2026-11-02_14.35 have picoscope files, but the pico trace is flat (sense path not in
# the circuit), so no calibration run has a usable external current.
CALIB_OCT = "2026-10-15_14.26_R=1024000_CharStim_large_current"
IMP = {
    'small_current': "2026-11-05_22.21_test12_100k_36umP_56umPE_PT3_ImpNew",
    'large_current': "2026-11-05_23.17_test12_100k_36umP_56umPE_PT3_ImpNew",
}
LSB_FREQ = 1000          # impedance runs stimulate at 1 kHz, LSB transferred at 1 kHz
CLIP_IMP_OHM = 1e6       # above: amplifier saturated / open, excluded
OLD_WINDOW = (3600, 11000)  # fixed pico window used so far in record_imp
FS = 20_000
# pico current-sense chain, as in PicoRecorder (not imported: needs the picosdk)
SENSE_GAIN, SENSE_R_OHM = 100, 10_000
MODE_COLORS = {'small_current': 'tab:blue', 'large_current': 'tab:orange'}


def pico_uA(mv):
    return np.asarray(mv, dtype=float) / SENSE_GAIN * 1000 / SENSE_R_OHM


def pct(log10_dev):
    """log10 deviation -> % deviation"""
    return 100 * (10**np.asarray(log10_dev) - 1)


def su_demean(df, col, by='stim_unit'):
    return df[col] - df.groupby(by)[col].transform('mean')


def pico_fname(rec_dir, el, su):
    g = glob.glob(os.path.join(REC_ROOT, rec_dir, f"el_config_El{el:05d}_StimUnit{su:02d}_Ampl*.npz"))
    return g[0] if g else None


# ======== refit of the external current in the detected sine window ========
def _refit_one(args):
    rec_dir, el, su, freq, ncycles = args
    fname = pico_fname(rec_dir, el, su)
    if fname is None:
        return el, np.nan, np.nan, np.nan
    cur = pico_uA(np.load(fname)['mv'])
    rng = max(1, freq // 100)
    win = detect_sine_window(cur, FS, freq, int(ncycles / freq * FS))
    if win is None:
        return el, np.nan, np.nan, np.nan
    ampl, *_ = extract_amplitude(cur, FS, freq - rng, freq + rng, lowpass=3000.0,
                                 estimate_amplitude_from=win[0], estimate_amplitude_to=win[1])
    return el, ampl, *win


def refit_ext_current(rec_dir, imp_df):
    cache = os.path.join(REC_ROOT, rec_dir, "processed", "ext_current_detected_window.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    jobs = [(rec_dir, int(r.electrode), int(r.stim_unit), int(r.stim_sequence_frequency_Hz),
             int(r.stim_sequence_ncycles)) for r in imp_df.itertuples()]
    print(f"refitting {len(jobs)} pico traces of {rec_dir} ...")
    with Pool(16) as p:
        res = p.map(_refit_one, jobs, chunksize=20)
    out = pd.DataFrame(res, columns=['electrode', 'ext_current_uA_detwin', 'win_start', 'win_end'])
    out.to_csv(cache, index=False)
    return out


def load_imp(mode):
    x = pd.read_csv(os.path.join(REC_ROOT, IMP[mode], "processed", "all_impedance.csv"))
    x = x.merge(refit_ext_current(IMP[mode], x), on='electrode', how='left')
    n_all = len(x)
    x = x[(x.impedance_Ohm <= CLIP_IMP_OHM) & (x.ext_current_uA_detwin > 0)].copy()
    print(f"{mode}: {len(x)}/{n_all} electrodes below {CLIP_IMP_OHM/1e6:g} MOhm")
    x['frac_in_old_win'] = ((np.minimum(x.win_end, OLD_WINDOW[1]) - np.maximum(x.win_start, OLD_WINDOW[0]))
                            .clip(lower=0) / (OLD_WINDOW[1] - OLD_WINDOW[0])).clip(upper=1)
    x['l_ratio_old'] = np.log10(x.external_current_uA / x.current_uA)
    x['l_ratio'] = np.log10(x.ext_current_uA_detwin / x.current_uA)
    x['l_chip'] = np.log10(x.impedance_Ohm)
    x['l_ext'] = np.log10(x.amplitude_uV / x.ext_current_uA_detwin)
    for c in ['l_chip', 'l_ext', 'l_ratio', 'l_ratio_old']:
        x[f'{c}_dev'] = su_demean(x, c)
    return x


def load_calib(mode):
    s = pd.read_csv(os.path.join(REC_ROOT, CALIB[mode], "LSB_sweep", "processed",
                                 f"{mode}_lsb_sweep.csv"))
    s = s[s.freq == LSB_FREQ].copy()
    s['l_lsb'] = np.log10(s[f'{mode}_LSB_nA'])
    s['dev'] = su_demean(s, 'l_lsb', by='stimunit_id')
    return s


def sd_pct(v):
    return pct(np.std(v, ddof=1))


def short(m):
    return m.split('_')[0]


# %%
if __name__ == '__main__':
    calib = {m: load_calib(m) for m in CALIB}
    imp = {m: load_imp(m) for m in IMP}
    both = imp['small_current'].merge(imp['large_current'], on='electrode', suffixes=('_s', '_l'))

    fig, axes = plt.subplots(3, 3, figsize=(18, 16), constrained_layout=True)

    # ---- A: old ratio is set by how much of the sine fell into the fixed window ----
    ax = axes[0, 0]
    for m in IMP:
        x = imp[m]
        r = np.corrcoef(x.frac_in_old_win, x.l_ratio_old)[0, 1]
        ax.scatter(100 * x.frac_in_old_win, pct(x.l_ratio_old), s=6, alpha=.4,
                   color=MODE_COLORS[m], edgecolors='none', label=f'{short(m)}: r={r:.2f}')
    ax.axhline(0, color='k', lw=.8)
    ax.set_xlabel('% of the fixed pico window (3600-11000) that contains the sine')
    ax.set_ylabel('(external - set) / set current, OLD fit (%)')
    ax.legend(fontsize=8, loc='upper left', markerscale=3)
    ax.set_title('A  Old external current = how much sine the fixed window caught\n'
                 '(pico not synchronized: sine starts anywhere in 3500-9700)',
                 fontsize=10, loc='left')

    # ---- B: example trace, sine starting late ----
    ax = axes[0, 1]
    x = imp['small_current']
    ex = x.loc[(x.frac_in_old_win - .6).abs().idxmin()]
    cur = pico_uA(np.load(pico_fname(IMP['small_current'], int(ex.electrode),
                                     int(ex.stim_unit)))['mv'])
    t = np.arange(len(cur)) / FS * 1000
    ax.plot(t, cur, lw=.4, color='k')
    ax.axvspan(*(np.array(OLD_WINDOW) / FS * 1000), color='tab:red', alpha=.15,
               label='old fixed window')
    ax.axvspan(ex.win_start / FS * 1000, ex.win_end / FS * 1000, color='tab:green', alpha=.15,
               label='detected window')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('pico current (uA)')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title(f'B  Example: El{int(ex.electrode)}, StimUnit{int(ex.stim_unit)}, '
                 f'{100*ex.frac_in_old_win:.0f}% of window filled\n'
                 f'old {pct(ex.l_ratio_old):+.0f}%  ->  refit {pct(ex.l_ratio):+.0f}% vs set',
                 fontsize=10, loc='left')

    # ---- C: old vs refit distribution ----
    ax = axes[0, 2]
    bins = np.linspace(-80, 40, 121)
    for m in IMP:
        x = imp[m]
        ax.hist(pct(x.l_ratio_old), bins=bins, histtype='step', color=MODE_COLORS[m], lw=1,
                ls='--', label=f'{short(m)} old: SD {sd_pct(x.l_ratio_old):.1f}%')
        ax.hist(pct(x.l_ratio), bins=bins, histtype='stepfilled', color=MODE_COLORS[m],
                alpha=.5, label=f'{short(m)} refit: SD {sd_pct(x.l_ratio):.1f}%, '
                                f'median {np.median(pct(x.l_ratio)):+.1f}%')
    ax.axvline(0, color='k', lw=.8)
    ax.set_yscale('log')
    ax.set_xlabel('(external - set) / set current (%)')
    ax.set_ylabel('# electrodes')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('C  Refit: the current source is consistent to ~2%,\n'
                 'but ~20% higher than the LSB says', fontsize=10, loc='left')

    # ---- D: within-stim-unit spread, refit values ----
    ax = axes[1, 0]
    groups, labels, colors = [], [], []
    for m in IMP:
        for data, lab in [(calib[m]['dev'], 'LSB calib\namp-to-amp'),
                          (imp[m]['l_ratio_dev'], 'ext / set\ncurrent'),
                          (imp[m]['l_chip_dev'], 'chip\nimp'),
                          (imp[m]['l_ext_dev'], 'ext imp\n(refit)')]:
            groups.append(pct(data))
            labels.append(f'{lab}\n{short(m)}')
            colors.append(MODE_COLORS[m])
    pos = np.arange(len(groups))
    vp = ax.violinplot(groups, positions=pos, showextrema=False, widths=.8)
    for body, c in zip(vp['bodies'], colors):
        body.set_facecolor(c)
        body.set_alpha(.35)
    rng_ = np.random.default_rng(0)
    for p, g, c in zip(pos, groups, colors):
        ax.scatter(p + rng_.uniform(-.12, .12, len(g)), g, s=3, color=c, alpha=.4)
        ax.text(p, 1.0, f'SD\n{sd_pct(np.log10(1 + np.asarray(g) / 100)):.1f}%',
                transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8,
                weight='bold')
    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(-25, 30)
    ax.set_ylabel('deviation from stim-unit mean (%)')
    ax.set_title('D  Within a stim unit: current error ~0.4%,\n'
                 'chip = external imp spread (5-7%) -> real impedance spread',
                 fontsize=10, loc='left')

    # ---- E: chip impedance per electrode reproducible ----
    ax = axes[1, 1]
    xs, ys = pct(both.l_chip_dev_s), pct(both.l_chip_dev_l)
    r = np.corrcoef(both.l_chip_dev_s, both.l_chip_dev_l)[0, 1]
    ax.scatter(xs, ys, s=6, alpha=.4, color='k', edgecolors='none')
    lim = np.percentile(np.abs(np.r_[xs, ys]), 99.5) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1, label='y=x')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('small-current run: chip imp, dev. from SU mean (%)')
    ax.set_ylabel('large-current run: chip imp, dev. from SU mean (%)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title(f'E  Same electrode, two runs/modes: chip impedance\n'
                 f'r = {r:.2f} (n={len(both)}): the spread is the electrodes',
                 fontsize=10, loc='left')

    # ---- F: per-SU current ratio reproducible across runs ----
    ax = axes[1, 2]
    gs = imp['small_current'].groupby('stim_unit').l_ratio.mean()
    gl = imp['large_current'].groupby('stim_unit').l_ratio.mean()
    gs, gl = gs - gs.mean(), gl - gl.mean()
    r = np.corrcoef(gs, gl)[0, 1]
    ax.scatter(pct(gs), pct(gl), s=30, color='gray', edgecolors='k', lw=.4)
    for su in gs.index:
        ax.annotate(str(su), (pct(gs[su]), pct(gl[su])), fontsize=6, xytext=(3, 2),
                    textcoords='offset points')
    lim = 1.15 * max(np.abs(pct(gs)).max(), np.abs(pct(gl)).max())
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='y=x')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('small current: stim-unit mean ext/set, dev. from mean (%)')
    ax.set_ylabel('large current: stim-unit mean ext/set, dev. from mean (%)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title(f'F  Residual per-stim-unit LSB error: '
                 f'SD {sd_pct(gs):.1f}% / {sd_pct(gl):.1f}%\n'
                 f'systematic, same in both modes (r = {r:.2f})', fontsize=10, loc='left')

    # ---- G: residual vs LSB ----
    ax = axes[2, 0]
    for m in IMP:
        g = imp[m].groupby('stim_unit').agg(l_lsb=('LSB_nA', lambda v: np.log10(v.iloc[0])),
                                            l_ratio=('l_ratio', 'mean'))
        g = g - g.mean()
        slope, icpt = np.polyfit(g.l_lsb, g.l_ratio, 1)
        r = np.corrcoef(g.l_lsb, g.l_ratio)[0, 1]
        ax.scatter(pct(g.l_lsb), pct(g.l_ratio), s=30, color=MODE_COLORS[m], edgecolors='k',
                   lw=.4, label=f'{short(m)}: slope {slope:.2f}, r={r:.2f}')
        xx = np.linspace(g.l_lsb.min(), g.l_lsb.max(), 10)
        ax.plot(pct(xx), pct(slope * xx + icpt), color=MODE_COLORS[m], lw=1)
    ax.axhline(0, color='k', ls='--', lw=1, label='slope 0: LSB fully corrects')
    ax.set_xlabel('calibrated LSB of stim unit, deviation from mean (%)')
    ax.set_ylabel('mean ext / set current of stim unit,\ndeviation from mean (%)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('G  Units with a high LSB deliver even more than calibrated:\n'
                 'the 1 MOhm calibration compresses the unit-to-unit differences',
                 fontsize=10, loc='left')

    # ---- H: LSB calibration reproducible over 3 weeks ----
    ax = axes[2, 1]
    oct_ = pd.read_csv(os.path.join(REC_ROOT, CALIB_OCT, "processed",
                                    "large_current_lsb_characterization.csv"))
    lsb_oct = oct_.groupby('stimunit_id').large_current_LSB_nA.apply(lambda v: np.log10(v).mean())
    lsb_nov = calib['large_current'].groupby('stimunit_id').l_lsb.mean()
    lsb_oct, lsb_nov = lsb_oct - lsb_oct.mean(), lsb_nov - lsb_nov.mean()
    r = np.corrcoef(lsb_oct, lsb_nov)[0, 1]
    ax.scatter(pct(lsb_oct), pct(lsb_nov), s=30, color=MODE_COLORS['large_current'],
               edgecolors='k', lw=.4)
    for su in lsb_oct.index:
        ax.annotate(str(su), (pct(lsb_oct[su]), pct(lsb_nov[su])), fontsize=6,
                    xytext=(3, 2), textcoords='offset points')
    ax.plot([-12, 12], [-12, 12], 'k--', lw=1, label='y=x')
    ax.set_aspect('equal')
    ax.set_xlabel('LSB 2026-10-15 (R=1.024 MOhm, 6 amps/SU)\ndeviation from mean (%)')
    ax.set_ylabel('LSB 2026-11-05 (R=1 MOhm, 3 amps/SU)\ndeviation from mean (%)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title(f'H  The 1 MOhm calibration itself is stable over 3 weeks\n'
                 f'r = {r:.2f}, scatter {sd_pct(lsb_oct - lsb_nov):.1f}% (large current)',
                 fontsize=10, loc='left')

    # ---- I: global offset per stim unit ----
    ax = axes[2, 2]
    for i, m in enumerate(IMP):
        x = imp[m]
        su = np.sort(x.stim_unit.unique())
        data = [pct(x.loc[x.stim_unit == s, 'l_ratio']) for s in su]
        bp = ax.boxplot(data, positions=np.arange(len(su)) + (i - .5) * .38, widths=.33,
                        patch_artist=True, showfliers=False, manage_ticks=False)
        for b in bp['boxes']:
            b.set_facecolor(MODE_COLORS[m])
            b.set_alpha(.6)
        med = np.median(pct(x.l_ratio))
        ax.axhline(med, color=MODE_COLORS[m], ls='--', lw=1,
                   label=f'{short(m)}: median {med:+.1f}%')
    ax.set_xticks(np.arange(len(su)))
    ax.set_xticklabels(su, fontsize=7)
    ax.set_xlabel('stim unit')
    ax.set_ylabel('(external - set) / set current, refit (%)')
    ax.legend(fontsize=8, loc='lower left')
    # MEA1K voltage reads ~0.857x true at 1 kHz (unbonded chip 4983, pico-verified, see
    # docs/mea1k_voltage_scale_handoff.md) -> the voltage-based LSB is too small by that
    # factor -> expected ext/set = 1/0.857
    expected = 100 * (1 / 0.857 - 1)
    ax.axhline(expected, color='k', ls=':', lw=1.2,
               label=f'expected from MEA1K voltage scale 0.857: {expected:+.1f}%')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title('I  ~20% more current than the LSB says: the LSB is calibrated with the\n'
                 'MEA1K voltage, which reads ~14% low (handoff doc) -> LSB too small',
                 fontsize=10, loc='left')

    fig.suptitle(f'{HEADSTAGE}: chip vs external current sense (PT3, small + large current), '
                 f'compared with the 1 MOhm LSB calibration ({LSB_FREQ} Hz, gain 7)',
                 fontweight='bold')
    # plt.show()
    plt.savefig('./live_figures/test.png', dpi=300)
