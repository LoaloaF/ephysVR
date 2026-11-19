"""Figures of characterize_mea1k_stimulaters.py: the DAC sweeps for the zero-current DAC code
and the current LSB. Each function takes data and returns the figure, the script saves it."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

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

def stim_unit_characterization(stim_unit_results, fit_all, fit_last, stimunit_id, old_code=None):
    """Transient peak vs DAC code of one stim unit: all sets so far (top), the latest set
    (bottom), each with its sigmoid fit (fit_* from _fit_zero_current_dac, or None).
    old_code: previous zero-current code (recalibration), drawn in purple."""
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
        if old_code is not None:
            a.axvline(old_code, color='purple', linewidth=2, alpha=0.7, label=f"Previous code: {old_code}")
        a.axhline(0, color='gray', linestyle='--', alpha=0.5)
        a.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        a.set_ylabel("Peak [uV]")
        a.grid(False, axis='x')
        a.grid(True, axis='y', alpha=0.3)
        a.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
    # leave room for the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def stitch_pngs(fnames, ncols=8, scale=1/3):
    """Grid of the saved per stim unit figures (same size), scaled down. Returns a PIL image."""
    imgs = [Image.open(f) for f in fnames]
    w, h = int(imgs[0].width*scale), int(imgs[0].height*scale)
    nrows = -(-len(imgs) // ncols)
    grid = Image.new("RGB", (w*min(ncols, len(imgs)), h*nrows), "white")
    for i, img in enumerate(imgs):
        grid.paste(img.convert("RGB").resize((w, h), Image.LANCZOS), ((i % ncols)*w, (i//ncols)*h))
    return grid

# ======== current LSB ========

def lsb_summary(aggr, R, stim_units, stim_mode):
    """Zero-current DAC, amplitude, phase and LSB per stim unit. Several conditions (gain,
    frequency, sine amplitude) get their own marker, shifted a bit next to each other."""
    fig = plt.figure(figsize=(14, 8))
    ax = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2),
          fig.add_subplot(2, 2, 3, projection='polar'), fig.add_subplot(2, 2, 4)]
    cond_cols = ['gain', 'freq', 'sine_amplitude_DAC_units']
    conditions = list(aggr.groupby(cond_cols).groups)
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    width = .8 / len(conditions)

    for i, (gain, freq, sine) in enumerate(conditions):
        d = aggr[(aggr.gain == gain) & (aggr.freq == freq) & (aggr.sine_amplitude_DAC_units == sine)]
        med = d.drop_duplicates('stimunit_id')
        x = d['stimunit_id'] + (i - (len(conditions)-1)/2) * width
        x_med = med['stimunit_id'] + (i - (len(conditions)-1)/2) * width
        m = markers[i % len(markers)]
        label = f"gain {gain:g}, {freq}Hz, ±{sine}"

        ax[0].scatter(x, d[f'{stim_mode}_zero_current_DAC'], s=50, c=d['dac_code_offset'],
                      alpha=0.6, marker=m)
        ax[1].scatter(x, d['mean_amplitude_uV'], alpha=0.8, c=d['dac_code_offset'],
                      edgecolor='none', s=30, marker=m, label=label)
        ax[1].scatter(x_med, med['median_amplitude_uV'], color='blue', s=200*width, marker='_',
                      linewidths=2, zorder=5, label='Median' if i == 0 else None)
        ax[2].scatter(np.mod(np.deg2rad(d['phase_shift_deg']), 2 * np.pi), 1 + d['stimunit_id'],
                      s=50, color='green', alpha=0.6, marker=m, label=label)
        ax[3].scatter(x, d[f"{stim_mode}_LSB_nA"], alpha=0.3, color='blue', edgecolor='none',
                      s=30, marker=m, label=label)
        ax[3].scatter(x_med, med[f"{stim_mode}_median_LSB_nA"], color='blue', s=200*width,
                      marker='_', linewidths=2, zorder=5, label='Median' if i == 0 else None)

    ax[0].set_title("1.) Zero current DAC code")
    ax[0].set_ylabel("Zero current DAC code")
    ax[1].set_title("2.) Mean Amplitude [uV]")
    ax[1].set_ylabel("Mean Amplitude [uV]")
    ax[2].set_title("Phase Shift (Polar)", pad=20)
    ax[2].set_theta_zero_location('N')
    ax[2].set_theta_direction(-1)
    ax[2].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.1, 1.0))
    ax[3].set_title(f"3.) LSB, {stim_mode} mode [nA]\n(Linear scaling by resistance R={R/1000:g}kΩ)")
    ax[3].set_ylabel("LSB [nA]")
    for a in (ax[0], ax[1], ax[3]):
        a.set_xlabel("Stim Unit ID")
        a.set_xticks(stim_units)
        a.tick_params(axis='x', which='both', rotation=90)
        a.grid(True)
    ax[1].legend(fontsize=9)
    ax[3].legend(fontsize=9)
    fig.tight_layout()
    return fig
