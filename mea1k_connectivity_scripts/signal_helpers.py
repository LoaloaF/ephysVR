from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, tf2sos
import numpy as np
import matplotlib.pyplot as plt


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


# ── Main estimation ───────────────────────────────────────────────────────────

def estimate_frequency_power(signal, sampling_rate, min_band, max_band, name=None,
                             dac=None, lp_cutoff=None, debug=False, dac2=None, dac3=None):
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
    """
    m      = signal.mean()
    signal = signal.copy() - m

    # 50 Hz notch before anything else
    signal = notch_filter(signal, sampling_rate)
    
    if not(min_band < 30 < max_band):
    # high pass filter over 30 Hz to remove slow drifts (not part of lock-in, just for cleaner debug plots)
        signal = signal - _causal_lowpass(signal, sampling_rate, highcut=30.0)

    # Onset / offset
    if dac is not None:
        start, end = _detect_onset_offset(dac)
    else:
        start, end = 0, len(signal) - 1
    
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
        seg_dac -= seg_dac.mean()
        dac_amp, dac_phase = lock_in_extract(seg_dac, ref_freq, sampling_rate, lp_cutoff)
        phi_diff         = sig_phase[settle_n:] - dac_phase[settle_n:]
        mean_phase_shift = float(np.degrees(np.angle(np.mean(np.exp(1j * phi_diff)))))

    if debug:
        fig, ax = plt.subplots(3, 1, figsize=(12, 7))
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle("Amplifier voltage trace")

        t     = np.arange(len(signal)) / sampling_rate * 1000   # ms
        t_seg = t[start:end]

        # ── ax[0]: raw signal + DAC ───────────────────────────────────────
        ax[0].plot(t, signal, color='blue', alpha=0.8, label='Signal (notched)')
        ax[0].axvline(x=t[start], color='gray', linestyle='--', alpha=0.6, label='Onset')
        ax[0].axvline(x=t[end],   color='gray', linestyle='-.',  alpha=0.6, label='Offset')
        ax[0].set_xlabel('Time [ms]')
        ax[0].set_ylabel(f'Δ Potential\nfrom {m:.0f} uV')
        ax[0].grid(True)
        [ax[0].spines[s].set_visible(False) for s in ['top', 'right', 'left', 'bottom']]
        if dac is not None:
            ax0b = ax[0].twinx()
            ax0b.plot(t, dac, color='green', alpha=0.6, label='DAC')
            ax0b.set_ylabel('DAC units', color='green')
            ax0b.tick_params(axis='y', labelcolor='green')
            [ax0b.spines[s].set_visible(False) for s in ['top', 'right', 'left', 'bottom']]
            lines, labels = ax[0].get_legend_handles_labels()
            lb, lb2 = ax0b.get_legend_handles_labels()
            ax[0].legend(lines + lb, labels + lb2, loc='upper right')
        else:
            ax[0].legend()

        # ── ax[1]: lock-in amplitude envelope ────────────────────────────
        ax[1].plot(t_seg, sig_amp, color='red', alpha=0.9,
                   label=f'Lock-in amplitude  (LP={lp_cutoff:.1f} Hz)')
        ax[1].axhline(y=mean_ampl, color='k', linestyle='dashed',
                      label=f'Mean (settled): {mean_ampl:.3f} uV')
        # draw signal as well
        ax[1].plot(t_seg, seg_sig, color='blue', alpha=0.5, label='Signal (seg)')
        if settle_n < len(t_seg):
            ax[1].axvline(x=t_seg[settle_n], color='orange', linestyle=':',
                          label='Settle end')
        ax[1].set_xlabel('Time [ms]')
        ax[1].set_ylabel('Amplitude [uV]')
        ax[1].sharex(ax[0])
        ax[1].grid(True)
        [ax[1].spines[s].set_visible(False) for s in ['top', 'right', 'left', 'bottom']]
        ax[1].legend(loc='upper right')

        # ── ax[2]: instantaneous phase difference ─────────────────────────
        if dac is not None:
            phi_diff_full = np.degrees(np.angle(np.exp(1j * (sig_phase - dac_phase))))
            ax[2].plot(t_seg, phi_diff_full, color='purple', alpha=0.7,
                       label='Instantaneous phase diff')
            ax[2].axhline(y=mean_phase_shift, color='k', linestyle='dashed',
                          label=f'Circular mean: {mean_phase_shift:.1f}°')
            if settle_n < len(t_seg):
                ax[2].axvline(x=t_seg[settle_n], color='orange', linestyle=':',
                              label='Settle end (excluded from mean)')
            ax[2].set_ylim(-180, 180)
            ax[2].set_ylabel('Phase diff [°]')
            ax[2].set_title(f'Phase shift: {mean_phase_shift:.1f}°')
        else:
            ax[2].plot(t_seg, sig_amp, color='red', alpha=0.8, label='Lock-in amplitude')
            ax[2].set_ylabel('Amplitude [uV]')

        ax[2].set_xlabel('Time [ms]')
        ax[2].sharex(ax[0])
        ax[2].grid(True)
        [ax[2].spines[s].set_visible(False) for s in ['top', 'right', 'left', 'bottom']]
        ax[2].legend()
        
        if name is not None:
            fig.suptitle(f"Config: {name}", fontsize=16)

        plt.savefig('./live_figures/debug_signal.png')
        plt.show()

    return mean_ampl, mean_phase_shift if dac is not None else None