"""
PicoScope 5000A Streaming Recorder
===================================
Simple start/stop API for continuous recording.

Usage:
    from picoscope_recorder import PicoRecorder

    rec = PicoRecorder(voltage_range="2V", resolution="12BIT", sample_interval_us=100)
    rec.start("my_capture.npz")
    # ... do other stuff, wait, etc ...
    rec.stop()  # saves NPZ and closes cleanly
"""

import ctypes
import csv
import glob
import os
import threading
import time
import numpy as np

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok


# Valid options for user-facing params
RANGES = {
    "10MV":  ps.PS5000A_RANGE['PS5000A_10MV'],
    "20MV":  ps.PS5000A_RANGE['PS5000A_20MV'],
    "50MV":  ps.PS5000A_RANGE['PS5000A_50MV'],
    "100MV": ps.PS5000A_RANGE['PS5000A_100MV'],
    "200MV": ps.PS5000A_RANGE['PS5000A_200MV'],
    "500MV": ps.PS5000A_RANGE['PS5000A_500MV'],
    "1V":    ps.PS5000A_RANGE['PS5000A_1V'],
    "2V":    ps.PS5000A_RANGE['PS5000A_2V'],
    "5V":    ps.PS5000A_RANGE['PS5000A_5V'],
    "10V":   ps.PS5000A_RANGE['PS5000A_10V'],
    "20V":   ps.PS5000A_RANGE['PS5000A_20V'],
}

RESOLUTIONS = {
    "8BIT":  ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_8BIT"],
    "12BIT": ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"],
    "14BIT": ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_14BIT"],
    "15BIT": ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_15BIT"],
    "16BIT": ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_16BIT"],
}


class PicoRecorder:
    def __init__(
        self,
        voltage_range: str = "2V",
        resolution: str = "12BIT",
        sample_interval_us: int = 100,
        coupling: str = "DC",
        channel: str = "A",
        buffer_size: int = 500,
    ):
        """
        Args:
            voltage_range:      "10MV" to "20V"
            resolution:         "8BIT", "12BIT", "14BIT", "15BIT", "16BIT"
            sample_interval_us: sample period in microseconds (100 = 10 kS/s)
            coupling:           "DC" or "AC"
            channel:            "A" or "B"
            buffer_size:        driver buffer chunk size (500 is fine)
        """
        if voltage_range not in RANGES:
            raise ValueError(f"voltage_range must be one of {list(RANGES)}")
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {list(RESOLUTIONS)}")

        self._range = RANGES[voltage_range]
        self._range_label = voltage_range
        self._resolution = RESOLUTIONS[resolution]
        self._interval_us = sample_interval_us
        self._coupling = ps.PS5000A_COUPLING[f'PS5000A_{coupling.upper()}']
        self._channel = ps.PS5000A_CHANNEL[f'PS5000A_CHANNEL_{channel.upper()}']
        self._buf_size = buffer_size

        self._handle = ctypes.c_int16()
        self._opened = False
        self._recording = False
        self._thread = None
        self._samples = []
        self._lock = threading.Lock()
        self._filename = None
        
        self._open_device()
        self._configure_channel()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def close(self):                  # call once when fully done
        """Close the device. Idempotent and safe to call after an error."""
        if not self._opened:
            return
        # make sure streaming is stopped before releasing the handle
        if self._recording:
            try:
                self.stop()
            except Exception:
                self._recording = False
        try:
            ps.ps5000aCloseUnit(self._handle)
        except Exception:
            pass
        self._opened = False

    def stop(self):
        if not self._recording:
            return
        self._recording = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        ps.ps5000aStop(self._handle)
        self._save_npz(len(self._samples))

    def start(self, filename):
        if not self._opened:
            raise RuntimeError("PicoRecorder device is not open — cannot start.")
        if self._recording:            # never leave a previous stream running
            self.stop()
        self._filename = filename
        self._samples.clear()
        self._buf = np.zeros(self._buf_size, dtype=np.int16)
        self._register_buffer()
        self._start_streaming()       # fast — no USB enumeration
        self._recording = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def __del__(self):
        # safety net: release the USB device even if close() was never called
        # (e.g. the run crashed), so the next process can open it.
        try:
            self.close()
        except Exception:
            pass
        
    # def start(self, filename: str = "capture.npz"):
    #     """Open the scope, begin streaming, and collect samples in the background."""
    #     if self._recording:
    #         raise RuntimeError("Already recording — call stop() first.")

    #     self._filename = filename
    #     self._samples.clear()

    #     self._open_device()
    #     self._configure_channel()
    #     self._buf = np.zeros(self._buf_size, dtype=np.int16)
    #     self._register_buffer()
    #     self._start_streaming()

    #     self._recording = True
    #     self._thread = threading.Thread(target=self._poll_loop, daemon=True)
    #     self._thread.start()
    #     print(f"Recording to '{filename}' at {1e6 / self._interval_us:.0f} S/s ...")

    # def stop(self):
    #     """Stop streaming, save NPZ, and close the device."""
    #     if not self._recording:
    #         print("Not recording.")
    #         return

    #     self._recording = False
    #     self._thread.join(timeout=5)

    #     ps.ps5000aStop(self._handle)

    #     count = len(self._samples)
    #     if count > 0:
    #         self._save_npz(count)
    #     else:
    #         print("No samples collected.")

    #     ps.ps5000aCloseUnit(self._handle)
    #     self._opened = False
    #     print(f"Stopped — {count} samples saved to '{self._filename}'.")

    @property
    def sample_count(self) -> int:
        """Number of samples collected so far."""
        with self._lock:
            return len(self._samples)

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ----------------------------------------------------------
    # Internals
    # ----------------------------------------------------------

    def _open_device(self):
        status = ps.ps5000aOpenUnit(ctypes.byref(self._handle), None, self._resolution)
        try:
            assert_pico_ok(status)
        except Exception:
            if status in (282, 286):
                assert_pico_ok(ps.ps5000aChangePowerSource(self._handle, status))
            else:
                raise
        self._opened = True

    def _configure_channel(self):
        assert_pico_ok(ps.ps5000aSetChannel(
            self._handle, self._channel, 1, self._coupling, self._range, 0.0
        ))

    def _register_buffer(self):
        assert_pico_ok(ps.ps5000aSetDataBuffers(
            self._handle,
            self._channel,
            self._buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            None,
            self._buf_size,
            0,
            ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'],
        ))

    def _start_streaming(self):
        interval = ctypes.c_int32(self._interval_us)
        assert_pico_ok(ps.ps5000aRunStreaming(
            self._handle,
            ctypes.byref(interval),
            ps.PS5000A_TIME_UNITS['PS5000A_US'],
            0,              # no pre-trigger
            0,              # totalSamples (0 = unlimited)
            0,              # autoStop off
            1,              # downsample ratio
            ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'],
            self._buf_size,
        ))
        self._actual_interval_us = interval.value

    def _poll_loop(self):
        auto_stop = [False]

        def _callback(handle, n, start, overflow, trigAt, trig, autoStop, param):
            with self._lock:
                self._samples.extend(self._buf[start:start + n].tolist())
            if autoStop:
                auto_stop[0] = True

        c_callback = ps.StreamingReadyType(_callback)

        while self._recording and not auto_stop[0]:
            ps.ps5000aGetStreamingLatestValues(self._handle, c_callback, None)
            time.sleep(0.005)

    def _save_npz(self, count):
        maxADC = ctypes.c_int16()
        assert_pico_ok(ps.ps5000aMaximumValue(self._handle, ctypes.byref(maxADC)))

        raw = np.array(self._samples[:count], dtype=np.int16)
        mv = adc2mV(raw, self._range, maxADC)
        time_s = np.arange(count) * (self._actual_interval_us * 1e-6)

        np.savez_compressed(self._filename, time_s=time_s, mv=mv)


def plot_capture(filename: str):
    """Plot the frequency distribution (FFT) of the captured data from NPZ."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. install it with `pip install matplotlib`.")
        return

    print(f"Plotting FFT from '{filename}'...")
    
    try:
        data = np.load(filename)
        time_s = data['time_s']
        mv = data['mv']
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    if len(time_s) < 2:
        print("Not enough data to plot.")
        return

    
    # Calculate frequency spectrum
    dt = time_s[1] - time_s[0]
    if dt <= 0:
        print("Invalid time step in data.")
        return
        
    n = len(mv)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.fft.rfft(mv)
    magnitude = np.abs(fft_vals) * (2.0 / n)  # Normalize magnitude
    
    plt.figure(figsize=(10, 4))
    # Skip DC offset (index 0) for clearer visualization
    plt.plot(freqs[1:], magnitude[1:], color='r')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (mV)")
    plt.title(f"Frequency Distribution: {filename}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # save to live_figures
    plt.savefig('./live_figures/frequency_distribution.png')
    
    plt.savefig('./live_figures/20khz_power_timeline.png')


def plot_dc_offsets(
    rec_dir: str,
    pattern: str = "*.npz*",
    plot: bool = True,
    interval_s: tuple[float, float] | None = None,
    units: str = "voltage",
):
    """Plot DC offset estimates for NPZ files in a directory.

    Returns:
        List of mean DC levels per file (mV or uA).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if plot:
            print("matplotlib is required for plotting. install it with `pip install matplotlib`.")
        return []

    files = sorted(glob.glob(os.path.join(rec_dir, pattern)))
    if not files:
        print(f"No files matched: {os.path.join(rec_dir, pattern)}")
        return []

    def _moving_average(x, window_s, fs):
        window = max(3, int(window_s * fs))
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=np.float64) / window
        return np.convolve(x, kernel, mode="same")

    if units not in ("voltage", "current"):
        raise ValueError("units must be 'voltage' or 'current'")

    dc_means = []

    for filename in files:
        try:
            data = np.load(filename)
            time_s = data["time_s"]
            mv = data["mv"]
        except Exception as exc:
            print(f"Failed to load {filename}: {exc}")
            continue

        if len(time_s) < 2:
            print(f"Not enough data in {filename}")
            continue

        dt = time_s[1] - time_s[0]
        if dt <= 0:
            print(f"Invalid time step in {filename}")
            continue

        fs = 1.0 / dt

        # 2 ms smoothing window to isolate slow DC trends from 1 kHz/20 kHz components
        dc_smoothed = _moving_average(mv, 0.002, fs)   # ~2 ms

        if units == "current":
            # 10K sense resistor, 1000 amplification, pico voltage in mV -> current in uA
            signal = mv / 1000 * 1000 / 10_000
            dc_smoothed = dc_smoothed / 1000 * 1000 / 10_000
            y_label = "uA"
        else:
            signal = mv
            y_label = "mV"

        dc_means.append(float(np.mean(dc_smoothed)))

        if plot:
            if interval_s is None:
                t_start = time_s[0] + 0.4
                t_end = min(time_s[-1], t_start + 0.1)
            else:
                t_start, t_end = interval_s
                if t_end <= t_start:
                    print(f"Invalid interval in {filename}: {interval_s}")
                    continue

            interval_mask = (time_s >= t_start) & (time_s <= t_end)

            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
            axes[0].plot(time_s, signal, color="0.2", linewidth=0.6, label="Raw")
            axes[0].plot(time_s, dc_smoothed, color="tab:green", linewidth=1.0, label="DC (2 ms avg)")
            axes[0].axvspan(t_start, t_end, color="0.7", alpha=0.8)
            axes[0].set_ylabel(y_label)
            axes[0].set_title(f"Raw Signal: {os.path.basename(filename)}")
            axes[0].legend(loc="upper right")

            if interval_mask.any():
                axes[1].plot(time_s[interval_mask], signal[interval_mask], color="tab:blue", linewidth=0.9)
            else:
                axes[1].text(0.5, 0.5, "Interval out of range", ha="center", va="center")
            axes[1].set_facecolor("0.7")
            axes[1].set_ylabel(y_label)
            axes[1].set_title(f"Zoomed Interval: {t_start:.3f}s to {t_end:.3f}s")
            axes[1].set_xlabel("Time (s)")

            plt.tight_layout()
            out_path = "./live_figures/DC_offset.png"
            # plt.show()
            plt.savefig(out_path)
            plt.close(fig)
            print(f"Saved DC offset plot to {out_path}")
            time.sleep(2)

    return dc_means


def plot_dc_hist(
    rec_dir_a: str,
    rec_dir_b: str,
    pattern: str = "*.npz*",
    units: str = "voltage",
):
    """Plot overlayed histograms of mean DC levels for two directories."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. install it with `pip install matplotlib`.")
        return

    means_a = plot_dc_offsets(rec_dir_a, pattern=pattern, plot=False, units=units)
    means_b = plot_dc_offsets(rec_dir_b, pattern=pattern, plot=False, units=units)

    if not means_a and not means_b:
        print("No DC mean values found for either directory.")
        return

    plt.figure(figsize=(8, 4))
    if means_a:
        plt.hist(means_a, bins=30, alpha=0.6, label=os.path.basename(rec_dir_a))
    if means_b:
        plt.hist(means_b, bins=30, alpha=0.6, label=os.path.basename(rec_dir_b))
    x_label = "Mean DC level (uA)" if units == "current" else "Mean DC level (mV)"
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title("DC Offset Histogram")
    plt.legend()
    plt.tight_layout()

    out_path = "./live_figures/DC_offset_hist.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved DC offset histogram to {out_path}")


# ----------------------------------------------------------
# Quick demo when run directly
# ----------------------------------------------------------
if __name__ == "__main__":
    demo_filename = "demo_capture.npz"
    rec = PicoRecorder(voltage_range="5V", resolution="16BIT", sample_interval_us=10)
    rec.start(demo_filename)

    # try:
    #     while rec.is_recording:
    #         print(f"  {rec.sample_count} samples ...", end="\r")
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     pass

    time.sleep(1)

    rec.stop()
    
    # demo_filename = "/mnt/SpatialSequenceLearning/devices/headstage_devices/MEA1K24/recordings/2026-05-30_18.09_Imp_Bond_extCur10KSensePico_PT4_large_current_f1000Hz/el_config_El00859_StimUnit21_Ampl0409.npz"
    # demo_filename = "/mnt/SpatialSequenceLearning/devices/headstage_devices/MEA1K24/recordings/2026-05-30_18.09_Imp_Bond_extCur10KSensePico_PT4_large_current_f1000Hz/el_config_El26070_StimUnit14_Ampl0890.npz"
    # # Plot the result
    # plot_capture(demo_filename)
    
    
    
    rec_dir_a = "/home/houmannjava/Documents"
    rec_dir_a = "./"

    plot_dc_offsets(rec_dir_a, interval_s=(0.078, 0.0825), units="current")
    # plot_dc_offsets(rec_dir_b)

    # plot_dc_hist(rec_dir_a, rec_dir_b)

    
    