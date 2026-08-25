import os
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, tf2sos
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


# ── Filters ──────────────────────────────────────────────────────────────────

def _causal_sos(sos, signal):
    """Apply an SOS filter causally, initialised to the first sample (no DC step)."""
    zi = sosfilt_zi(sos) * signal[0]
    out, _ = sosfilt(sos, signal, zi=zi)
    return out


def notch_filter(signal, sampling_rate, freq=50.0, Q=30.0):
    """Causal notch. Q=30 → ~1.7 Hz bandwidth at 50 Hz."""
    b, a = iirnotch(freq / (0.5 * sampling_rate), Q)
    return _causal_sos(tf2sos(b, a), signal)


def _causal_lowpass(signal, sampling_rate, highcut, order=4):
    nyquist = 0.5 * sampling_rate
    sos = butter(order, highcut / nyquist, btype='low', output='sos')
    return _causal_sos(sos, signal)


# ── Lock-in demodulation ─────────────────────────────────────────────────────

def lock_in_extract(signal, ref_freq, sampling_rate, lp_cutoff):
    """
    Demodulate signal at ref_freq using a lock-in amplifier.

    Multiplies by cos/sin references then causal-lowpass filters.
    Returns amplitude envelope and instantaneous phase (radians).

    lp_cutoff controls the noise/speed tradeoff:
      lower  → smoother, slower to settle, better SNR
      higher → faster onset, noisier
    Settling time ≈ 2 / lp_cutoff  seconds  (2 time-constants).
    """
    t = np.arange(len(signal)) / sampling_rate
    I = signal * np.cos(2 * np.pi * ref_freq * t)
    Q = signal * np.sin(2 * np.pi * ref_freq * t)
    I_lp = _causal_lowpass(I, sampling_rate, lp_cutoff)
    Q_lp = _causal_lowpass(Q, sampling_rate, lp_cutoff)
    amplitude = 2.0 * np.sqrt(I_lp**2 + Q_lp**2)
    phase     = np.arctan2(Q_lp, I_lp)
    return amplitude, phase


# ── Onset detection ───────────────────────────────────────────────────────────

def _detect_onset_offset(dac):
    d  = np.diff(dac.astype(float))
    mx, mn = d.max(), d.min()
    rise = np.where(d >= mx * 0.5)[0]
    fall = np.where(d <= mn * 0.5)[0]
    start = int(rise[0]) if len(rise) else 0
    end   = int(fall[-1]) if len(fall) else len(dac) - 1
    return (start, end) if start < end else (0, len(dac) - 1)


def _set_smart_yticks(ax):
    """Set y-axis ticks only for values: 0, ±3000, ±50000, ±100000 that fall within current y-range."""
    candidate_ticks = [0, 3000, -3000, 50000, -50000, 100000, -100000]
    ymin, ymax = ax.get_ylim()
    
    # Filter ticks that fall within the current y-range
    valid_ticks = [t for t in candidate_ticks if ymin <= t <= ymax]
    valid_ticks = sorted(valid_ticks)
    
    if valid_ticks:
        ax.set_yticks(valid_ticks)
        ax.set_yticklabels([f'{int(t):,}' if t != 0 else '0' for t in valid_ticks])


# ── Main estimation ───────────────────────────────────────────────────────────

def estimate_frequency_power(signal, sampling_rate, min_band, max_band, name=None,
                             dac=None, lp_cutoff=None, debug=False, dac2=None, dac3=None,
                            #  plot_n_after_onset=200, plot_n_before_onset=20):
                             plot_n_after_onset=None, plot_n_before_onset=None,
                             return_baseline=False):
    """
    Estimate amplitude and phase shift at the stimulus frequency.

    Everything goes through the lock-in:
      - amplitude  = mean of lock-in envelope over the settled window
      - phase shift = circular mean of (signal_phase − dac_phase)
                      over the settled window

    Parameters
    ----------
    min_band, max_band  Band edges [Hz].  Centre = lock-in reference frequency.
                        Set min_band=0 for a lowpass scenario.
    lp_cutoff           Lock-in lowpass cutoff [Hz].
                        Default: max(ref_freq / 10, 2).
    plot_n_after_onset  Only plot the first N samples after DAC onset.
                        If None, plot entire range. Used for debug plots only.
    plot_n_before_onset Only plot N samples before DAC onset.
                        If None, plot from start. Used for debug plots only.
    return_baseline     If True, also demodulate the pre-onset segment
                        (signal[:start]) and return its settled lock-in
                        amplitude alongside the during-stimulation amplitude,
                        as (mean_ampl, mean_phase_shift, baseline_ampl).
                        Requires dac (to detect onset); otherwise baseline_ampl
                        is None.
    """
    m      = signal.mean()
    signal = signal.copy() - m

    # 50 Hz notch before anything else
    signal = notch_filter(signal, sampling_rate)
    
    # # filter out 3khz+
    # cutoff_hz = 3000.0
    # nyquist = 0.5 * sampling_rate
    # if sampling_rate > 0 and cutoff_hz < nyquist:
    #     sos = butter(4, cutoff_hz, btype='low', fs=sampling_rate, output='sos')
    #     signal = sosfiltfilt(sos, signal)

    # Onset / offset
    if dac is not None:
        start, end = _detect_onset_offset(dac)
    else:
        start, end = 0, len(signal) - 1
        
    # start = end-2000
    # start, end = 0, 2000
    # start, end = 1700, 2000
    # fig, ax = plt.subplots(3, 1, figsize=(12, 7))
    # ax[0].plot(signal[start:end])
    # ax[0].plot(signal[start:end])
    # ax0b = ax[0].twinx()
    # # ax0b.plot(dac[start:end], color='green', alpha=0.6, label='DAC')
    # # ax0b.set_ylabel('DAC units', color='green')
    # # ax0b.tick_params(axis='y', labelcolor='green')
    # ax[0].legend()
    # print("Debug: Detected onset at sample {}, offset at sample {}".format(start, end))
    # plt.savefig('./live_figures/debug_signal_onset.png')
    # plt.close('all')
    # exit()

    # Lock-in reference and settling
    ref_freq = (min_band + max_band) / 2.0 if min_band > 0 else max_band
    if lp_cutoff is None:
        lp_cutoff = max(ref_freq / 10.0, 2.0)
    settle_n = min(int(2.0 * sampling_rate / lp_cutoff), (end - start) // 3)

    # Demodulate signal
    seg_sig           = signal[start:end]
    sig_amp, sig_phase = lock_in_extract(seg_sig, ref_freq, sampling_rate, lp_cutoff)
    mean_ampl          = float(np.mean(sig_amp[settle_n:]))

    # Demodulate DAC and compute phase shift
    mean_phase_shift = None
    dac_amp = dac_phase = None
    if dac is not None:
        seg_dac = dac[start:end].astype(float)
        # invert signal (flip)
        seg_dac -= seg_dac.mean()
        seg_dac *= -1
        dac_amp, dac_phase = lock_in_extract(seg_dac, ref_freq, sampling_rate, lp_cutoff)
        phi_diff         = sig_phase[settle_n:] - dac_phase[settle_n:]
        mean_phase_shift = float(np.degrees(np.angle(np.mean(np.exp(1j * phi_diff)))))

    # Baseline: lock-in amplitude of the pre-onset segment (before the sine sequence starts)
    baseline_ampl = None
    base_amp      = None
    if return_baseline:
        if dac is None:
            start = 0
            
        pre_seg = signal[:start]
        if len(pre_seg) > 0:
            base_amp, _  = lock_in_extract(pre_seg, ref_freq, sampling_rate, lp_cutoff)
            base_settle_n = min(int(2.0 * sampling_rate / lp_cutoff), len(pre_seg) // 3)
            baseline_ampl = float(np.mean(base_amp[base_settle_n:]))

    if debug:
        fig, ax = plt.subplots(3, 1, figsize=(12, 7))
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle("Amplifier voltage trace")

        t     = np.arange(len(signal)) / sampling_rate * 1000   # ms
        
        # Limit plotting range if plot_n_before/after_onset are specified.
        # When a baseline was computed, default to showing the full pre-onset
        # window too, so the baseline region is visible.
        if plot_n_before_onset is not None:
            plot_start = max(start - plot_n_before_onset, 0)
        elif return_baseline and base_amp is not None:
            plot_start = 0
        else:
            plot_start = start
        plot_end = start + plot_n_after_onset if plot_n_after_onset is not None else end
        plot_end = min(plot_end, end)  # Don't exceed actual signal end
        
        t_seg = t[plot_start:plot_end]

        # ── Calculate valid segment range within plot window ───────────────
        # Segment spans from index start to end in full signal coordinates
        # We want to plot from plot_start to plot_end
        # In segment coordinates (relative to start), that's from (plot_start - start) to (plot_end - start)
        seg_start_in_window = max(0, plot_start - start)
        seg_end_in_window = min(len(sig_amp), plot_end - start)
        
        # Position in t_seg where segment data begins (for NaN padding before onset)
        onset_idx_in_seg = start - plot_start if plot_start < start else 0

        # ── ax[0]: raw signal + DAC ───────────────────────────────────────
        plot_signal = signal[plot_start:plot_end]
        lbl = 'Raw signal (notch + demeaned)'
        if not(min_band < 30 < max_band):
        # high pass filter over 30 Hz to remove slow drifts (not part of lock-in, just for cleaner debug plots)
            plot_signal = plot_signal - _causal_lowpass(plot_signal, sampling_rate, highcut=30.0)
            lbl += ' + highpass(30 Hz)'
        ax[0].plot(t_seg, plot_signal, color='blue', alpha=0.8, label=lbl)
        if return_baseline and base_amp is not None and plot_start < start:
            ax[0].axvspan(t[plot_start], t[start], color='teal', alpha=0.1, label='Baseline window')
        ax[0].axvline(x=t[start], color='gray', linestyle='--', alpha=0.6, label='Onset')
        ax[0].axvline(x=t[plot_end], color='gray', linestyle='-.',  alpha=0.6, label='Offset' if plot_n_after_onset is None and plot_n_before_onset is None else f'Plot end')
        ax[0].set_xlabel('Time [ms]')
        ax[0].set_ylabel(f'Δ Potential\nfrom {m:.0f} uV')
        ax[0].yaxis.set_major_locator(plt.MultipleLocator(1000))
        ax[0].yaxis.grid(True, which='major', alpha=0.7, linestyle=':')
        _set_smart_yticks(ax[0])
        [ax[0].spines[s].set_visible(False) for s in ['top', 'right', 'left', 'bottom']]
        if dac is not None:
            ax0b = ax[0].twinx()
            dac_center = dac[plot_start:plot_end].mean()
            inverted_dac = (dac[plot_start:plot_end]-dac_center) * -1
            ax0b.plot(t_seg, inverted_dac+dac_center, color='green', alpha=0.6, label='DAC')
            ax0b.set_ylabel('DAC units', color='green')
            ax0b.tick_params(axis='y', labelcolor='green')
            [ax0b.spines[s].set_visible(False) for s in ['top', 'right', 'left', 'bottom']]
            lines, labels = ax[0].get_legend_handles_labels()
            lb, lb2 = ax0b.get_legend_handles_labels()
            ax[0].legend(lines + lb, labels + lb2, loc='upper right')
        else:
            ax[0].legend()

        # ── ax[1]: lock-in amplitude envelope ────────────────────────────
        # Align lock-in data with t_seg (pad with NaN if showing pre-onset data)
        sig_amp_plot = np.full(len(t_seg), np.nan)
        if seg_end_in_window > seg_start_in_window:
            sig_amp_plot[onset_idx_in_seg:onset_idx_in_seg + (seg_end_in_window - seg_start_in_window)] = sig_amp[seg_start_in_window:seg_end_in_window]

        # Splice in the pre-onset baseline lock-in amplitude, if computed
        if return_baseline and base_amp is not None and plot_start < start:
            base_start_in_window = plot_start
            base_end_in_window   = min(start, plot_end)
            sig_amp_plot[0:base_end_in_window - base_start_in_window] = base_amp[base_start_in_window:base_end_in_window]

        seg_sig_plot = np.full(len(t_seg), np.nan)
        if seg_end_in_window > seg_start_in_window:
            seg_sig_plot[onset_idx_in_seg:onset_idx_in_seg + (seg_end_in_window - seg_start_in_window)] = seg_sig[seg_start_in_window:seg_end_in_window]
        if return_baseline and base_amp is not None and plot_start < start:
            seg_sig_plot[0:base_end_in_window - base_start_in_window] = signal[base_start_in_window:base_end_in_window]

        if return_baseline and base_amp is not None and plot_start < start:
            ax[1].axvspan(t[plot_start], t[start], color='teal', alpha=0.1, label='Baseline window')
        ax[1].plot(t_seg, sig_amp_plot, color='red', alpha=0.9,
                   label=f'Lock-in amplitude  (LP={lp_cutoff:.1f} Hz, {min_band:.1f}-{max_band:.1f} Hz)')
        ax[1].axhline(y=mean_ampl, color='k', linestyle='dashed',
                      label=f'Mean (settled): {mean_ampl:.3f} uV')
        if return_baseline and baseline_ampl is not None:
            ax[1].axhline(y=baseline_ampl, color='teal', linestyle='dashed',
                          label=f'Baseline (pre-onset): {baseline_ampl:.3f} uV')
        # draw signal as well
        ax[1].plot(t_seg, seg_sig_plot, color='blue', alpha=0.5, label='Signal (seg)')
        if settle_n < len(t_seg) - onset_idx_in_seg:
            ax[1].axvline(x=t_seg[settle_n + onset_idx_in_seg], color='orange', linestyle=':',
                          label='Settle end')
        ax[1].set_xlabel('Time [ms]')
        ax[1].set_ylabel('Amplitude [uV]')
        ax[1].sharex(ax[0])
        ax[1].yaxis.set_major_locator(plt.MultipleLocator(1000))
        ax[1].yaxis.grid(True, which='major', alpha=0.7, linestyle=':')
        _set_smart_yticks(ax[1])
        [ax[1].spines[s].set_visible(False) for s in ['top', 'right', 'left', 'bottom']]
        ax[1].legend(loc='upper right')

        # ── ax[2]: instantaneous phase difference ─────────────────────────
        if dac is not None:
            dac_phase_plot = np.full(len(t_seg), np.nan)
            if seg_end_in_window > seg_start_in_window:
                dac_phase_plot[onset_idx_in_seg:onset_idx_in_seg + (seg_end_in_window - seg_start_in_window)] = dac_phase[seg_start_in_window:seg_end_in_window]
            
            sig_phase_plot = np.full(len(t_seg), np.nan)
            if seg_end_in_window > seg_start_in_window:
                sig_phase_plot[onset_idx_in_seg:onset_idx_in_seg + (seg_end_in_window - seg_start_in_window)] = sig_phase[seg_start_in_window:seg_end_in_window]
            
            phi_diff_full = np.degrees(np.angle(np.exp(1j * (sig_phase_plot - dac_phase_plot))))
            ax[2].plot(t_seg, phi_diff_full, color='purple', alpha=0.7,
                       label='Instantaneous phase diff')
            ax[2].axhline(y=mean_phase_shift, color='k', linestyle='dashed',
                          label=f'Circular mean: {mean_phase_shift:.1f}°')
            if settle_n + onset_idx_in_seg < len(t_seg):
                ax[2].axvline(x=t_seg[settle_n + onset_idx_in_seg], color='orange', linestyle=':',
                              label='Settle end (excluded from mean)')
            ax[2].set_ylim(-180, 180)
            ax[2].set_ylabel('Phase diff [°]')
            ax[2].set_title(f'Phase shift: {mean_phase_shift:.1f}°')
        else:
            ax[2].plot(t_seg, sig_amp_plot, color='red', alpha=0.8, label='Lock-in amplitude')
            ax[2].set_ylabel('Amplitude [uV]')

        ax[2].set_xlabel('Time [ms]')
        ax[2].sharex(ax[0])
        ax[2].yaxis.set_major_locator(plt.MultipleLocator(1000))
        ax[2].yaxis.grid(True, which='major', alpha=0.7, linestyle=':')
        _set_smart_yticks(ax[2])
        [ax[2].spines[s].set_visible(False) for s in ['top', 'right', 'left', 'bottom']]
        ax[2].legend()
        
        if name is not None:
            fig.suptitle(f"Config: {name}", fontsize=16)

        plt.savefig('./live_figures/debug_signal.png')
        # plt.show()

    if return_baseline:
        return mean_ampl, (mean_phase_shift if dac is not None else None), baseline_ampl
    return mean_ampl, mean_phase_shift if dac is not None else None


# ── Amplitude extraction (clean rewrite) ───────────────────────────────────────

def preprocess_signal(signal, sampling_rate, notch=True, demean=True,
                      highpass=None, lowpass=None):
    """
    Apply optional preprocessing steps to a signal.

    All steps are opt-in via args so the caller controls (and the plot shows)
    exactly what was applied.

    Returns (processed_signal, applied_labels).
    """
    out    = np.asarray(signal, dtype=float).copy()
    labels = []
    if demean:
        out -= out.mean()
        labels.append('demean')
    if notch:
        out = notch_filter(out, sampling_rate)
        labels.append('notch 50 Hz')
    if highpass is not None:
        out = out - _causal_lowpass(out, sampling_rate, highpass)
        labels.append(f'highpass {highpass:g} Hz')
    if lowpass is not None:
        out = _causal_lowpass(out, sampling_rate, lowpass)
        labels.append(f'lowpass {lowpass:g} Hz')
    return out, labels


def _lock_in_over(signal, start, end, ref_freq, sampling_rate, lp_cutoff):
    """Lock-in over [start:end]. Returns (mean_amp, amp_envelope, phase, settle_n)."""
    amp, phase = lock_in_extract(signal[start:end], ref_freq, sampling_rate, lp_cutoff)
    settle_n   = min(int(2.0 * sampling_rate / lp_cutoff), (end - start) // 3)
    return float(np.mean(amp[settle_n:])), amp, phase, settle_n


def extract_amplitude(signal, sampling_rate, min_band, max_band,
                      estimate_amplitude_from=None, estimate_amplitude_to=None,
                      dac=None, use_dac_interval=False,
                      notch=True, demean=True, highpass=None, lowpass=None,
                      lp_cutoff=None, name=None, debug=False,
                      plot_fname='debug_signal', zoom_from=None, zoom_to=None,
                      ylim=None, save_to_dir=None, dac_ylim_range=32):
    """
    Lock-in amplitude at the stimulus frequency over a chosen sample interval.

    Analysis interval (samples)
        [estimate_amplitude_from : estimate_amplitude_to].
        With use_dac_interval=True the DAC onset/offset is used instead.

    Preprocessing (opt-in, indicated in the plot)
        demean, notch (50 Hz), highpass, lowpass.

    lp_cutoff   Lock-in lowpass cutoff [Hz]. Default max(ref_freq/10, 2).
    zoom_from, zoom_to
        Delta (in samples) from the sine start for the right-hand zoom panel,
        e.g. zoom_from=-200, zoom_to=400 shows 200 samples before to 400 after
        onset. Defaults to the analysis interval.
    ylim
        (ymin, ymax) to fix the y-limits of the amplitude panels
        (processed + zoom). Default None -> responsive/auto-scaled.
    save_to_dir
        Directory to save the debug figure to. Default None -> ./live_figures.
    dac_ylim_range
        Half-range of the DAC y-axes, always centred on 512
        (i.e. limits are 512 ± dac_ylim_range). Default 32.

    DC levels (from the raw signal, before demeaning)
        baseline  mean level before onset  (signal[:start]); None if start==0.
        sine_dc   mean level during the sine (signal[start:end]) — its DC centre.
        The DC shift induced by the sine is (sine_dc - baseline).

    Returns (mean_amp, phase_shift, baseline, sine_dc).
    phase_shift is None without a dac; baseline is None without a pre-onset region.
    """
    proc, applied = preprocess_signal(signal, sampling_rate, notch=notch,
                                      demean=demean, highpass=highpass, lowpass=lowpass)

    # Analysis interval
    if use_dac_interval and dac is not None:
        start, end = _detect_onset_offset(dac)
    elif estimate_amplitude_from is not None and estimate_amplitude_to is not None:
        start, end = int(estimate_amplitude_from), int(estimate_amplitude_to)
    else:
        raise ValueError("Pass estimate_amplitude_from/to, or use_dac_interval=True with a dac.")
    start, end = max(0, start), min(len(proc), end)

    # Lock-in
    ref_freq = (min_band + max_band) / 2.0 if min_band > 0 else max_band
    if lp_cutoff is None:
        lp_cutoff = max(ref_freq / 10.0, 2.0)
    mean_amp, amp, sig_phase, settle_n = _lock_in_over(proc, start, end, ref_freq,
                                                       sampling_rate, lp_cutoff)

    # Phase shift vs DAC (inverted DAC reference)
    phase_shift = None
    if dac is not None:
        seg_dac = dac[start:end].astype(float)
        seg_dac = -(seg_dac - seg_dac.mean())
        _, dac_phase = lock_in_extract(seg_dac, ref_freq, sampling_rate, lp_cutoff)
        phi          = sig_phase[settle_n:] - dac_phase[settle_n:]
        phase_shift  = float(np.degrees(np.angle(np.mean(np.exp(1j * phi)))))

    # DC levels from the raw signal (preprocessing demeans, so use the original)
    raw      = np.asarray(signal, dtype=float)
    baseline = float(np.median(raw[:start])) if start > 0 else None
    sine_dc  = float(np.mean(raw[start:end]))

    if debug:
        _plot_extract_amplitude(signal, proc, applied, sampling_rate, start, end,
                                amp, settle_n, mean_amp, min_band, max_band,
                                lp_cutoff, dac, zoom_from, zoom_to, name,
                                plot_fname, baseline, sine_dc, ylim, save_to_dir,
                                dac_ylim_range)
    return mean_amp, phase_shift, baseline, sine_dc


def _plot_extract_amplitude(raw, proc, applied, sampling_rate, start, end,
                            amp, settle_n, mean_amp, min_band, max_band,
                            lp_cutoff, dac, zoom_from, zoom_to, name,
                            plot_fname, baseline=None, sine_dc=None, ylim=None,
                            save_to_dir=None, dac_ylim_range=32):
    """Debug figure: full raw (top-left), processed + lock-in (bottom-left), zoom (right)."""
    t = np.arange(len(raw)) / sampling_rate * 1000.0   # ms

    def _draw_dac(ax, sl):
        """Draw the (inverted) DAC on a twin y-axis, fixed to 512 ± dac_ylim_range."""
        axd = ax.twinx()
        c   = dac[sl].astype(float)
        c   = -(c - c.mean()) + c.mean()          # invert around centre
        axd.plot(t[sl], c, color='green', alpha=0.6, label='DAC')
        axd.set_ylabel('DAC units', color='green')
        axd.tick_params(axis='y', labelcolor='green')
        axd.set_ylim(512 - dac_ylim_range, 512 + dac_ylim_range)
        return axd

    # Lock-in envelope placed on the full time axis (NaN outside the interval)
    amp_full            = np.full(len(raw), np.nan)
    amp_full[start:end] = amp[:end - start]

    # Zoom window: zoom_from/zoom_to are deltas (in samples) relative to the
    # sine start, e.g. -200 -> 200 samples before, +400 -> 400 after.
    # Default to the analysis interval.
    z0 = start + int(zoom_from) if zoom_from is not None else start
    z1 = start + int(zoom_to)   if zoom_to   is not None else end
    z0, z1 = max(0, z0), min(len(raw), z1)

    fig = plt.figure(figsize=(15, 7))
    gs  = fig.add_gridspec(2, 2, width_ratios=[3, 1.2], hspace=0.35, wspace=0.22)
    ax_raw  = fig.add_subplot(gs[0, 0])
    ax_proc = fig.add_subplot(gs[1, 0], sharex=ax_raw)
    ax_zoom = fig.add_subplot(gs[:, 1])

    def _mark_interval(ax):
        ax.axvspan(t[start], t[min(end, len(t) - 1)], color='gray', alpha=0.12,
                   label='Analysis interval')

    # top-left: full raw signal, untouched + DC baseline / sine centre
    ax_raw.plot(t, raw, color='blue', alpha=0.8, lw=0.7)
    _mark_interval(ax_raw)
    if baseline is not None:
        ax_raw.hlines(baseline, t[0], t[start], color='orange', ls='dashed', linewidth=2,
                      label=f'Pre-sine baseline: {baseline:.1f} uV')
    if sine_dc is not None:
        sine_lbl = f'Sine DC centre: {sine_dc:.1f} uV'
        if baseline is not None:
            sine_lbl += f'  (shift {sine_dc - baseline:+.1f})'
        ax_raw.hlines(sine_dc, t[start], t[min(end, len(t) - 1)], color='magenta', ls='dashed',
                      linewidth=2, label=sine_lbl)
    ax_raw.set_title('Raw signal (full, no processing)')
    ax_raw.set_ylabel('Potential [uV]')
    if dac is not None:
        _draw_dac(ax_raw, slice(0, len(raw)))
    ax_raw.legend(loc='upper right', fontsize=8)

    # Centre the displayed signal on the sine's own DC baseline so the amplitude
    # line lines up with the sine peaks (the sine oscillates within the window
    # around its own DC level, not the global mean used for demeaning).
    sine_base = float(np.mean(proc[start:end]))
    proc_disp = proc - sine_base

    # bottom-left: processed signal (centred on sine baseline) + lock-in amplitude
    proc_lbl = 'Processed: ' + (', '.join(applied) if applied else 'none') + ' − sine baseline'
    ax_proc.plot(t, proc_disp, color='blue', alpha=0.55, lw=0.7, label=proc_lbl)
    ax_proc.plot(t, amp_full, color='red', alpha=0.9,
                 label=f'Lock-in amp (LP={lp_cutoff:.1f} Hz, {min_band:.0f}-{max_band:.0f} Hz)')
    ax_proc.axhline(mean_amp, color='k', ls='dashed',
                    label=f'Mean (settled): {mean_amp:,.2f} uV')
    ax_proc.axhline(-mean_amp, color='k', ls='dashed', alpha=0.4)
    _mark_interval(ax_proc)
    ax_proc.set_xlabel('Time [ms]')
    ax_proc.set_ylabel('Amplitude [uV]')
    if ylim is not None:
        ax_proc.set_ylim(ylim)
    ax_proc.legend(loc='upper right', fontsize=8)

    # right: zoom (default = analysis interval); keeps DAC overlay
    tz = t[z0:z1]
    ax_zoom.plot(tz, proc_disp[z0:z1], color='blue', alpha=0.55, lw=0.8, label='Processed')
    ax_zoom.plot(tz, amp_full[z0:z1], color='red', alpha=0.9, label='Lock-in amp')
    ax_zoom.axhline(mean_amp, color='k', ls='dashed', lw=0.8)
    ax_zoom.axhline(-mean_amp, color='k', ls='dashed', lw=0.8, alpha=0.4)
    settle_idx = start + settle_n
    if z0 <= settle_idx < min(z1, len(t)):
        ax_zoom.axvline(t[settle_idx], color='orange', ls=':', label='Settle end')
    if z0 <= start < min(z1, len(t)):
        ax_zoom.axvline(t[start], color='gray', ls='--', alpha=0.6, label='Interval start')
    ax_zoom.set_title('Zoom')
    ax_zoom.set_xlabel('Time [ms]')
    if ylim is not None:
        ax_zoom.set_ylim(ylim)
    if dac is not None:
        _draw_dac(ax_zoom, slice(z0, z1))
    ax_zoom.legend(loc='upper right', fontsize=8)

    title = f'Amplitude: {mean_amp:_.2f} uV'
    if name:
        title = f'{name}   —   {title}'
    fig.suptitle(title, fontsize=21, fontweight='bold')
    
    plt.savefig(f'./live_figures/{plot_fname}.png')
    if save_to_dir is not None:
        os.makedirs(save_to_dir, exist_ok=True)
        plt.savefig(os.path.join(save_to_dir, f'{plot_fname}.png'))
    plt.close(fig)