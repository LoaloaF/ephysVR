import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert

from scipy.fft import fft


def bandpass_filter(signal, sampling_rate, lowcut, highcut, order=4):
    nyquist = 0.5 * sampling_rate  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Create a Butterworth band-pass filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter to the signal using filtfilt for zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def lowpass_filter(signal, sampling_rate, highcut, order=4):
    nyquist = 0.5 * sampling_rate  # Nyquist frequency
    high = highcut / nyquist
    
    # Create a Butterworth low-pass filter
    b, a = butter(order, high, btype='low')
    
    # Apply the filter to the signal using filtfilt for zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal

def extract_average_amplitude(signal):
    # Compute the analytic signal using the Hilbert transform
    analytic_signal = hilbert(signal)
    # Compute the envelope (magnitude of the analytic signal)
    amplitude_envelope = np.abs(analytic_signal)
    # Compute the average amplitude of the envelope
    average_amplitude = np.mean(amplitude_envelope)
    return average_amplitude, amplitude_envelope

def estimate_frequency_power(signal, sampling_rate, min_band, max_band, debug=False):
    m = signal.mean()
    signal -= m
    
    # Compute the FFT of the signal
    fft_result = np.fft.fft(signal)
    # Compute the power spectrum
    power_spectrum = np.abs(fft_result) ** 2
    # Compute the corresponding frequencies
    freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    # Only keep the positive frequencies
    positive_freqs = freqs[freqs >= 1]
    positive_power_spectrum = power_spectrum[freqs >= 1]
    # print(positive_power_spectrum)
    # power_1KHz = positive_power_spectrum[(positive_freqs > min_band) & 
    #                                      (positive_freqs < max_band)].max()
    # power_300_3000KHz, mean not max
    power_1KHz = positive_power_spectrum[(positive_freqs > min_band) & 
                                         (positive_freqs < max_band)]
    power_1KHz = np.median(power_1KHz)
    
    if min_band == 0:
        signal_1khz = lowpass_filter(signal, sampling_rate, max_band)
    else:
        signal_1khz = bandpass_filter(signal, sampling_rate, min_band, max_band)
    
    mean_ampl, _ = extract_average_amplitude(signal_1khz)
    if debug:
        fig, ax = plt.subplots(3, 1, figsize=(12, 6))
        fig.subplots_adjust( hspace=.5)
        fig.suptitle("Voltage with external 2uA 1KHz sine signal")
        
        t = np.arange(len(signal))/sampling_rate *1000
        ax[0].plot(t, signal, color='blue', alpha=.8, label='Signal')
        ax[0].set_xlabel('Time [ms]')
        ax[0].set_yticks([-10000, -1000, 0, 1000, 10000])
        ax[0].set_ylabel(f'Δ Potential\nfrom {m:.0f} uV')
        ax[0].grid(True)
        [ax[0].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
        ax[0].legend()
        
        ax[1].plot(positive_freqs, positive_power_spectrum, color='orange',
                   label='Power Spectrum')
        ax[1].scatter([1000], power_1KHz, edgecolor='red', facecolor='none', 
                      label=f'1KHz Power: {power_1KHz:.1e}', s=100)
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Power')
        ax[1].set_xlim(0, 1500)
        # ax[1].set_ylim(0, 1e5//2)
        ax[1].grid(True)
        [ax[1].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
        ax[1].legend()
        
        
        ax[2].plot(t, signal_1khz, color='blue', alpha=.5,
                   label='1KHz Bandpass Filtered Signal')
        ax[2].plot([t[0]-20,t[-1]+20], [mean_ampl,mean_ampl], color='k', 
                   linestyle='dashed', label=f'Average Amplitude: {mean_ampl:.3f} uV')
        ax[2].set_xlabel('Time [ms]')
        ax[2].set_ylabel('Amplitude')
        ax[2].set_ylabel(f'Δ Potential\nfrom {m:.3f} uV')
        ax[2].set_yticks(ax[0].get_yticks())
        ax[2].grid(True)
        ax[2].sharex(ax[0])
        [ax[2].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
        ax[2].legend()
        plt.show()
        debug = False
        
    # print((np.abs(signal[:20] +1.37237234)))
    # print((np.abs(signal[:20] +1.37237234) < .001).all())
    # print()
    # print((np.abs(signal[-20:])))
    # print((np.abs(signal[-20:] -0.66959086)))
    # print((np.abs(signal[-20:] -0.66959086) < .001).all())
    
    if (np.abs(signal[:20] +1.37237234) < .001).all() or (np.abs(signal[-20:] -0.66959086) < .001).all():
        print("Signal is clipped")
        return np.nan, np.nan
        
    return power_1KHz, mean_ampl


def calculate_phase_shift(data, dac, sampling_rate=20000, freq=1000, debug=False):
    """
    Calculate the phase shift of each row in `data` with respect to the DAC signal.

    Parameters:
        data (ndarray): 2D array where each row is a signal.
        dac (ndarray): 1D array representing the DAC signal.
        sampling_rate (int): Sampling rate of the signals in Hz.
        freq (float): Frequency of the sinusoid in Hz (e.g., 1 kHz).
        debug (bool): If True, plot the calculation process for debugging.

    Returns:
        phase_shifts (ndarray): 1D array of phase shifts for each row in `data`.
    """
    n_samples = data.shape[1]
    t = np.arange(n_samples) / sampling_rate  # Time vector

    # Generate a reference sinusoid at the target frequency
    reference_sinusoid = np.sin(2 * np.pi * freq * t)

    # FFT of the DAC signal
    dac_fft = fft(dac)
    dac_freq_idx = int(freq * n_samples / sampling_rate)
    dac_phase = np.angle(dac_fft[dac_freq_idx])

    if debug:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.title("DAC Signal and Reference Sinusoid")
        plt.plot(t, dac, label="DAC Signal")
        plt.plot(t, reference_sinusoid, label="Reference Sinusoid", linestyle="dashed")
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.title("FFT of DAC Signal")
        plt.plot(np.fft.fftfreq(n_samples, 1 / sampling_rate), np.abs(dac_fft))
        plt.axvline(freq, color="red", linestyle="dashed", label="1 kHz Component")
        plt.legend()
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.tight_layout()
        plt.show()

    # Calculate phase shifts for each row in data
    phase_shifts = []
    for i, row in enumerate(data):
        row_fft = fft(row)
        row_phase = np.angle(row_fft[dac_freq_idx])
        phase_shift = row_phase - dac_phase
        # Normalize phase shift to [-π, π]
        phase_shift = (phase_shift + np.pi) % (2 * np.pi) - np.pi
        print(phase_shift)
        phase_shifts.append(phase_shift)

        if debug :  # Plot the first 5 rows for debugging
            plt.figure(figsize=(12, 4))
            plt.title(f"Row {i} Signal and Phase Shift")
            plt.plot(t, row, label="Row Signal")
            plt.plot(t, dac, label="DAC Signal", linestyle="dashed")
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.show()

    return np.array(phase_shifts)
