"""Figures of characterize_mea1k_stimulaters.py: the DAC sweeps for the zero-current DAC code
and the current LSB. Each function takes data and returns the figure, the script saves it."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

TARGET_RANGE_uV = 25_000    # band around 0 the transient peak should land in at the zero-current code

# ======== zero-current DAC code ========

def dac_transients(traces, title, baseline_n, connect_search_to):
    """Debug view of one DAC sweep set: DAC (top), amplifier transient (middle), pico current
    (bottom), one color per recording (from _extract_DAC_transient_set). Middle panel: grey =
    baseline samples, blue band = where the connect is searched, ▲ = detected connect,
    ● = overshoot peak (within connect .. +350), ■ = plateau at connect + 225."""
    fig, ax = plt.subplots(figsize=(20, 10), nrows=3, sharex=True)
    ax[1].axvspan(0, baseline_n, color='0.8', alpha=0.4, zorder=0)
    ax[1].axvspan(baseline_n, connect_search_to, color='tab:blue', alpha=0.06, zorder=0)
    for t in traces:
        ax[1].plot(t["amplifier_trace"], linewidth=1, alpha=.7)
        col = ax[1].lines[-1].get_color()
        ax[1].plot([t["connect_at"]], [t["amplifier_trace"][t["connect_at"]]], marker='^', ms=9,
                   color=col, linestyle='none')
        ax[1].vlines(t["peak_at"], np.min(t["amplifier_trace"]), np.max(t["amplifier_trace"]),
                     alpha=0.4, linestyle='--', color=col)
        ax[1].scatter([t["peak_at"]], [t["peak"]], color=col, s=50, edgecolor=col, zorder=10,
                      marker='o' if t["is_overshoot"] else 's',
                      label=f'DAC {t["DAC_val"]}, Peak {t["peak"]/1000:.0f} mV')
        if t["dac_trace"] is not None:
            ax[0].plot(t["dac_trace"].astype(float), linewidth=1, color=col, alpha=0.4,
                       label=f'DAC {t["DAC_val"]}')
        if not np.isnan(t["current_uA"]).all():
            ax[2].plot(t["current_uA"][:3000], linewidth=0.6, alpha=0.25, color=col)
            ax[2].plot(t["current_smooth_uA"][:3000], linewidth=1.5, alpha=0.9, color=col,
                       linestyle='--', label='Pico current [uA]')
            ax[2].scatter([t["peak_at"]], [t["current_at_peak"]], color=col, s=50, edgecolor=col,
                          marker='o', zorder=10,
                          label=f'DAC {t["DAC_val"]}, Current {t["current_at_peak"]:.2f} uA')
    ax[0].set_title(title)
    ax[0].set_ylabel("DAC value")
    ax[1].set_ylabel("Amplifier [uV], baseline subtracted")
    ax[1].set_xlabel("Sample index")
    ax[2].set_ylabel("Current [uA]")
    ax[2].set_xlabel("Sample index")
    for a in (ax[0], ax[2]):
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=6, ncol=2, loc='upper right')
    if not ax[2].lines:
        ax[2].set_title("no picoscope recording", fontsize=9)
    # how the peak is read out, as a second legend
    how = [Patch(color='0.8', alpha=0.4, label=f'baseline + noise (first {baseline_n} samples)'),
           Patch(color='tab:blue', alpha=0.15, label=f'connect searched in {baseline_n}..{connect_search_to}'),
           Line2D([], [], marker='^', color='k', linestyle='none', label='connect: first edge > 50% of the largest there'),
           Line2D([], [], marker='o', color='k', linestyle='none', label='peak: overshoot > 5x noise above plateau (connect .. +350)'),
           Line2D([], [], marker='s', color='k', linestyle='none', label='peak: no overshoot, plateau at connect + 225')]
    ax[1].add_artist(ax[1].legend(handles=how, fontsize=7, loc='lower left'))
    ax[1].legend(fontsize=6, ncol=2, loc='upper right')
    fig.tight_layout()
    return fig

def _draw_fit(ax, fit, fit_label):
    """Sigmoid fit, the measured point closest to 0 (red) and the fit's zero crossing (green)."""
    ax.plot(*fit["curve"], color="green", linestyle="--",
            label=f"{fit_label} (R² = {fit['r_squared']:.3f})")
    ax.axvline(fit["emp_dac"], color='red', linestyle='--', alpha=0.7,
               label=f"Empirical Min (DAC: {fit['emp_dac']:.0f}, |Peak|: {abs(fit['emp_peak']):,.1f} uV)")
    ax.scatter([fit["emp_dac"]], [fit["emp_peak"]], facecolors='none', edgecolors='red', s=200,
               linewidths=2, zorder=6)
    if fit["zero_current_dac"] is not None:
        dac, offset = fit["zero_current_dac"], fit["zero_current_peak"]
        ax.axvline(dac, color='green', linestyle='-.', alpha=0.7,
                   label=f"Fit 0-Cross (DAC: {dac}, Offset: {offset:,.1f} uV, "
                         f"Δ: {abs(fit['emp_dac'] - dac):.0f})")
        ax.scatter([dac], [offset], facecolors='none', edgecolors='green', s=200, linewidths=2, zorder=6)

def stim_unit_characterization(stim_unit_results, fit_all, fit_last, stimunit_id):
    """Transient peak vs DAC code of one stim unit: all sets so far (top), the latest set
    (bottom), each with its sigmoid fit (fit_* from _fit_zero_current_dac, or None)."""
    fig, ax = plt.subplots(nrows=2, figsize=(14, 10))
    fig.suptitle(f"Characterization: Stimulation Unit {stimunit_id:02d}", fontsize=16, fontweight="bold")
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ax[0].axhspan(-TARGET_RANGE_uV, TARGET_RANGE_uV, color='gray', alpha=0.15, zorder=0,
                  label=f"±{TARGET_RANGE_uV/1000:.0f}k uV Target Range")
    for i, res_df in enumerate(stim_unit_results):
        if not res_df.empty:
            ax[0].scatter(res_df["DAC_val"], res_df["peak_uV"], s=30, alpha=0.3,
                          color=colors[i % len(colors)],
                          label=f"Ampl {res_df['ampl_id'].iloc[0]} | Set {res_df['set_id'].iloc[0]}")
    if fit_all is not None:
        _draw_fit(ax[0], fit_all, "Fit All")
    ax[0].set_title("All Measured Points")

    current_set = stim_unit_results[-1]
    ax[1].axhspan(-TARGET_RANGE_uV, TARGET_RANGE_uV, color='gray', alpha=0.15, zorder=0)
    ax[1].scatter(current_set["DAC_val"], current_set["peak_uV"], s=50, alpha=0.9,
                  color=colors[(len(stim_unit_results) - 1) % len(colors)], label="Current Set")
    if fit_last is not None:
        _draw_fit(ax[1], fit_last, "Fit Current")
    ax[1].set_title("Current Set Only")
    ax[1].set_xlabel("DAC value")
    ax[1].set_xticks([])

    for a in ax:
        a.axhline(0, color='gray', linestyle='--', alpha=0.5)
        a.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        a.set_ylabel("Peak [uV]")
        a.grid(False, axis='x')
        a.grid(True, axis='y', alpha=0.3)
        a.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
    # leave room for the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def all_dac_sweeps(results, stim_units, old_codes=None, new_codes=None):
    """All DAC sweep points per stim unit (4x8 grid), with the old (red) and new (green)
    zero-current DAC codes. results: concatenated results_StimUnit*.csv"""
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(4*8, 16), sharey=True, sharex=True)
    axes = axes.flatten()
    for stim_unit in stim_units:
        d = results[results.stimunit_id == stim_unit]
        if d.empty:
            continue
        ax = axes[stim_unit]
        ax.scatter(d.DAC_val, d.peak_uV, alpha=0.4, s=20, label=f"StimUnit {stim_unit:02d} samples")
        if old_codes is not None and stim_unit in old_codes.index:
            ax.axvline(old_codes.loc[stim_unit], color='red', linestyle='--',
                       label=f"OLD Zero-Current DAC: {old_codes.loc[stim_unit]}")
        if new_codes is not None and stim_unit in new_codes.index:
            ax.axvline(new_codes.loc[stim_unit], color='green', linestyle=':', linewidth=2,
                       label=f"NEW Zero-Current DAC: {new_codes.loc[stim_unit]}")
        ax.axhspan(-TARGET_RANGE_uV, TARGET_RANGE_uV, color='gray', alpha=0.15, zorder=0,
                   label=f"±{TARGET_RANGE_uV/1000:.0f}k uV Target Range")
        ax.set_title(f"Stim Unit {stim_unit:02d}", y=.95)
        ax.set_xlabel("DAC value", fontsize=10)
        ax.set_ylabel("Peak uV")
        ax.legend(fontsize=6)
        [ax.spines[side].set_visible(False) for side in ['top', 'right', 'left', 'bottom']]
    return fig

# ======== current LSB ========

def lsb_summary(aggr, R, stim_units, stim_mode):
    """One LSB condition: zero-current DAC, amplitude, phase and LSB per stim unit."""
    fig = plt.figure(figsize=(14, 8))
    ax = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2),
          fig.add_subplot(2, 2, 3, projection='polar'), fig.add_subplot(2, 2, 4)]
    unique_med = aggr.drop_duplicates('stimunit_id')
    freqs = sorted(aggr['freq'].unique())
    markers = ['o', 's', '^']

    ax[0].set_title("1.) Zero current DAC code")
    ax[0].set_ylabel("Zero current DAC code")
    for i, freq in enumerate(freqs):
        d = aggr[aggr['freq'] == freq]
        ax[0].scatter(d['stimunit_id'], d[f'{stim_mode}_zero_current_DAC'], s=50,
                      c=d['dac_code_offset'], alpha=0.6, marker=markers[i % len(markers)])

    ax[1].set_title("2.) Mean Amplitude [uV]")
    ax[1].set_ylabel("Mean Amplitude [uV]")
    for i, freq in enumerate(freqs):
        d = aggr[aggr['freq'] == freq]
        ax[1].scatter(d['stimunit_id'], d['mean_amplitude_uV'], alpha=0.8, c=d['dac_code_offset'],
                      edgecolor='none', s=30, marker=markers[i % len(markers)], label=f'{freq}Hz')
    ax[1].scatter(unique_med['stimunit_id'], unique_med['median_amplitude_uV'], color='blue',
                  s=200, marker='_', linewidths=2, zorder=5, label='Median')

    ax[2].set_title("Phase Shift (Polar)", pad=20)
    for i, freq in enumerate(freqs):
        d = aggr[aggr['freq'] == freq]
        ax[2].scatter(np.mod(np.deg2rad(d['phase_shift_deg']), 2 * np.pi), 1 + d['stimunit_id'],
                      s=50, color='green', alpha=0.6, marker=markers[i % len(markers)], label=f'{freq}Hz')
    ax[2].set_theta_zero_location('N')
    ax[2].set_theta_direction(-1)
    ax[2].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.1, 1.0))

    ax[3].set_title(f"3.) LSB, {stim_mode} mode [nA]\n(Linear scaling by resistance R={R/1000:g}kΩ)")
    ax[3].set_ylabel("LSB [nA]")
    for i, freq in enumerate(freqs):
        d = aggr[aggr['freq'] == freq]
        ax[3].scatter(d['stimunit_id'], d[f"{stim_mode}_LSB_nA"], alpha=0.3, color='blue',
                      edgecolor='none', s=30, marker=markers[i % len(markers)], label=f'{freq}Hz')
    ax[3].scatter(unique_med['stimunit_id'], unique_med[f"{stim_mode}_median_LSB_nA"], color='blue',
                  s=200, marker='_', linewidths=2, zorder=5, label='Median')

    for a in (ax[0], ax[1], ax[3]):
        a.set_xlabel("Stim Unit ID")
        a.set_xticks(stim_units)
        a.tick_params(axis='x', which='both', rotation=90)
        a.grid(True)
    ax[1].legend(fontsize=9)
    ax[3].legend(fontsize=9)
    fig.tight_layout()
    return fig

def lsb_sweep_summary(aggr, R, stim_mode):
    """Several LSB conditions (recording gain x frequency): LSB per stim unit and vs frequency,
    phase, and the picoscope / amplifier LSB ratio."""
    gains, freqs = sorted(aggr['gain'].unique()), sorted(aggr['freq'].unique())
    colors = {g: plt.cm.tab10(i) for i, g in enumerate(gains)}
    markers = {f: m for f, m in zip(freqs, ['o', 's', '^', 'D', 'v', 'P'])}
    lsb, ext_lsb = f"{stim_mode}_LSB_nA", f"{stim_mode}_external_LSB_nA"

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"LSB sweep, {stim_mode} (R={R/1000:g}kΩ)")
    for g in gains:
        for f in freqs:
            d = aggr[(aggr['gain'] == g) & (aggr['freq'] == f)]
            if d.empty:
                continue
            ax[0, 0].scatter(d['stimunit_id'], d[lsb], s=25, alpha=.5, color=colors[g], marker=markers[f],
                             label=f"gain {g:g}, {f} Hz")
            ax[0, 1].scatter([f]*len(d), d[lsb], s=20, alpha=.3, color=colors[g])
            ph = np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(d['phase_shift_deg'])))))
            ax[1, 0].scatter([f], [ph], s=60, color=colors[g], marker=markers[f])
            ax[1, 1].scatter([f]*len(d), d[ext_lsb] / d[lsb], s=20, alpha=.4, color=colors[g])
    for g in gains:
        d = aggr[aggr['gain'] == g]
        med = d.groupby('freq')[lsb].median()
        ax[0, 1].plot(med.index, med.values, color=colors[g], marker='_', ms=14, label=f"gain {g:g}")
        ratio = d.assign(r=d[ext_lsb] / d[lsb]).groupby('freq')['r'].median()
        ax[1, 1].plot(ratio.index, ratio.values, color=colors[g], marker='_', ms=14, label=f"gain {g:g}")
    ax[0, 0].set(title="Amplifier LSB per stim unit", xlabel="Stim Unit ID", ylabel="LSB [nA]")
    ax[0, 0].legend(fontsize=6, ncol=2)
    ax[0, 1].set(title="Amplifier LSB vs frequency (line: median)", xlabel="Frequency [Hz]",
                 ylabel="LSB [nA]", xscale='log')
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].set(title="Phase shift vs DAC (circular mean)", xlabel="Frequency [Hz]",
                 ylabel="Phase [deg]", xscale='log')
    # without picoscope recordings the ratios are all NaN, a log axis can't place ticks then
    has_ext = aggr[ext_lsb].notna().any()
    ax[1, 1].set(title="External (pico) LSB / amplifier LSB, 1 = amplifier gain correct"
                       + ("" if has_ext else "\n(no picoscope recordings)"),
                 xlabel="Frequency [Hz]", ylabel="ratio", xscale='log' if has_ext else 'linear')
    ax[1, 1].axhline(1, color='gray', linestyle='--', alpha=.5)
    ax[1, 1].legend(fontsize=8)
    for a in ax.flat:
        a.grid(True, alpha=.3)
    fig.tight_layout()
    return fig
