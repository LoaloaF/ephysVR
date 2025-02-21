import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert

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
        ax[0].set_yticks(np.array((-10,0,10))/50)
        ax[0].set_ylabel(f'Δ Potential\nfrom {m:.0f} mV')
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
        ax[1].set_ylim(0, 1e5//2)
        ax[1].grid(True)
        [ax[1].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
        ax[1].legend()
        
        
        ax[2].plot(t, signal_1khz, color='blue', alpha=.5,
                   label='1KHz Bandpass Filtered Signal')
        ax[2].plot([t[0]-20,t[-1]+20], [mean_ampl,mean_ampl], color='k', 
                   linestyle='dashed', label=f'Average Amplitude: {mean_ampl:.3f} mV')
        ax[2].set_xlabel('Time [ms]')
        ax[2].set_ylabel('Amplitude')
        ax[2].set_ylabel(f'Δ Potential\nfrom {m:.3f} mV')
        ax[2].set_yticks(np.array((-10,0,10))/50)
        ax[2].grid(True)
        ax[2].sharex(ax[0])
        [ax[2].spines[spine].set_visible(False) for spine in ['top', 'right', 'left', 'bottom']]
        ax[2].legend()
        plt.show()
        
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