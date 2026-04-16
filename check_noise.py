import os
from scipy import signal


import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

# import parent dir with general modules
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import ephys_constants as EC
from mea1k_modules.mea1k_raw_preproc import read_raw_data
from mea1k_modules.mea1k_visualizations import draw_mea1k


def render_rms_violin(rms_all_chunks_1, rms_all_chunks_2=None, 
                     color1='#5A8FC6', color2='#CCCCCC',
                     label1='Session 1', label2='Session 2',
                     title="RMS Noise Distribution"):
    """
    Create histogram plot of RMS noise distribution across channels.
    Can plot one or two sessions with fine bins.
    
    Args:
        rms_all_chunks_1: Array of shape (n_chunks, n_channels) from calculate_rms_noise
        rms_all_chunks_2: Optional second array for comparison (default: None)
        color1: Color for first histogram (default: '#5A8FC6' - blue)
        color2: Color for second histogram (default: '#CCCCCC' - light gray)
        label1: Label for first session (default: 'Session 1')
        label2: Label for second session (default: 'Session 2')
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    # Determine if we're plotting one or two sessions
    single_session = rms_all_chunks_2 is None
    
    fig, ax = plt.subplots(figsize=(8 if single_session else 10, 6))
    
    # Flatten all RMS values into distributions
    rms_flat_1 = rms_all_chunks_1.flatten()
    
    if single_session:
        # Single histogram
        ax.hist(rms_flat_1, bins=100, color=color1, alpha=0.7, 
                edgecolor='black', linewidth=0.5, label=label1)
    else:
        # Two histograms
        rms_flat_2 = rms_all_chunks_2.flatten()
        ax.hist(rms_flat_1, bins=100, color=color1, alpha=0.6, 
                edgecolor='black', linewidth=0.5, label=label1)
        ax.hist(rms_flat_2, bins=100, color=color2, alpha=0.6, 
                edgecolor='black', linewidth=0.5, label=label2)
    
    # Calculate and display statistics
    stats_y_pos = 0.97
    stats_text = f"{label1}:\nMean: {rms_flat_1.mean():.2f} µV\nMedian: {np.median(rms_flat_1):.2f} µV\nStd: {rms_flat_1.std():.2f} µV"
    
    if not single_session:
        stats_text += f"\n\n{label2}:\nMean: {rms_flat_2.mean():.2f} µV\nMedian: {np.median(rms_flat_2):.2f} µV\nStd: {rms_flat_2.std():.2f} µV"
    
    ax.text(0.98, stats_y_pos, stats_text, 
           transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=9, family='monospace')
    
    ax.set_xlabel('RMS Noise (µV)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_rms_per_electrode(rms_dict, title="RMS Noise per Electrode"):
    """
    Scatter plot of RMS noise for each electrode across multiple recordings.
    Points are sorted by the difference in RMS between recordings.
    Each point is the mean RMS across all chunks for that electrode/file.
    
    Args:
        rms_dict: Dictionary of {filename: rms_per_channel_array}
                  where rms_per_channel is already the mean across chunks
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Get list of filenames and their RMS data
    fnames = list(rms_dict.keys())
    rms_data = list(rms_dict.values())
    
    # Get all electrode indices that appear in any file
    all_electrodes = set()
    for rms_per_channel in rms_data:
        all_electrodes.update(range(len(rms_per_channel)))
    all_electrodes = sorted(list(all_electrodes))
    
    # Calculate the difference for each electrode (for sorting)
    differences = {}
    for el_idx in all_electrodes:
        # Get RMS values for this electrode from each file
        values = []
        for rms_per_channel in rms_data:
            if el_idx < len(rms_per_channel):
                values.append(rms_per_channel[el_idx])
        
        # Calculate difference (0 if only one file has this electrode)
        if len(values) == 2:
            differences[el_idx] = abs(values[0] - values[1])
        else:
            differences[el_idx] = 0
    
    # Sort electrodes by difference (descending)
    sorted_electrodes = sorted(all_electrodes, key=lambda x: -differences[x])
    
    # Create x positions based on sorted order
    x_positions = {el_idx: i for i, el_idx in enumerate(sorted_electrodes)}
    
    # Use a colormap for different files
    colors = plt.cm.tab10(np.linspace(0, 1, len(fnames)))
    
    # Plot each file
    for (fname, rms_per_channel), color in zip(rms_dict.items(), colors):
        x = []
        y = []
        for el_idx in sorted_electrodes:
            if el_idx < len(rms_per_channel):
                x.append(x_positions[el_idx])
                y.append(rms_per_channel[el_idx])
        
        ax.scatter(x, y, s=50, label=fname, color=color, alpha=0.7, 
                  edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('Electrode Index (sorted by RMS difference)', fontsize=11)
    ax.set_ylabel('RMS Noise (µV)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig 
def calculate_rms_noise(subdir, fname, n_samples, ampl_ids=None, n_chunks=10, chunk_duration_sec=1.0, output_plot_path=None):
    """
    Calculate RMS noise for each channel after bandpass filtering.
    
    Samples multiple chunks across the recording, applies bandpass filter (300-5000 Hz)
    like spike sorting preprocessing, then calculates RMS per channel.
    Excludes clipped/dead channels (detected by low variance < 1.0).
    
    Args:
        subdir: Path to the directory containing the raw data files
        fname: Name of the raw data file
        n_samples: Total number of samples in the file
        ampl_ids: Array of amplifier IDs (row indices) to use, or None for all channels
        n_chunks: Number of chunks to sample across the recording (default: 10)
        chunk_duration_sec: Duration of each chunk in seconds (default: 1.0s)
        output_plot_path: Path to save example traces plot (updated each chunk)
        
    Returns:
        rms_per_channel: Array of shape (n_channels,) - RMS noise averaged over chunks (excluding clipped)
        rms_all_chunks: Array of shape (n_chunks, n_channels) - RMS per chunk for violin plot (excluding clipped)
    """
    sampling_rate = 20_000  # Hz
    chunk_size = int(chunk_duration_sec * sampling_rate)
    
    # Design bandpass filter (Butterworth, order 4, 300-5000 Hz)
    nyquist = sampling_rate / 2
    low_freq = 300 / nyquist
    high_freq = 5000 / nyquist
    sos = signal.butter(4, [low_freq, high_freq], btype='bandpass', output='sos')
    
    # Sample chunk positions evenly across the recording
    chunk_positions = np.linspace(0, n_samples - chunk_size, n_chunks, dtype=int)
    
    print(f"Calculating RMS noise from {n_chunks} chunks...")
    
    # First pass: identify clipped channels by low RMS
    print("Screening for clipped/dead channels (RMS < 0.1)...")
    clipped_mask = None
    first_chunk = read_raw_data(subdir, fname, convert2uV=True, to_df=True,
                               col_slice=slice(chunk_positions[0], 
                                             chunk_positions[0] + chunk_size))
    first_chunk = first_chunk.values.astype(float)
    
    # If filtering channels, apply that first
    if ampl_ids is not None:
        first_chunk = first_chunk[ampl_ids, :]
    
    # Apply bandpass filter to calculate RMS
    filtered_first = np.zeros_like(first_chunk, dtype=np.float32)
    for ch in range(first_chunk.shape[0]):
        filtered_first[ch, :] = signal.sosfiltfilt(sos, first_chunk[ch, :])
    
    # Calculate RMS and identify clipped channels
    rms_first = np.sqrt(np.mean(filtered_first**2, axis=1))
    clipped_mask = rms_first < 0.1  # Channels with RMS < 0.1 µV are considered clipped
    n_clipped = np.sum(clipped_mask)
    if n_clipped > 0:
        print(f"Found {n_clipped} clipped/dead channels (RMS < 0.1 µV), excluding from analysis")
        print(f"  Clipped channel RMS range: [{rms_first[clipped_mask].min():.4f}, {rms_first[clipped_mask].max():.4f}] µV")
    
    rms_all_chunks = []
    
    for i, chunk_start in enumerate(chunk_positions):
        chunk_end = chunk_start + chunk_size
        
        chunk_data = read_raw_data(subdir, fname, convert2uV=True, to_df=True,
                                   col_slice=slice(chunk_start, chunk_end))
        chunk_data = chunk_data.values.astype(float)  # Shape: (n_channels, chunk_size)
        
        # Filter channels if ampl_ids provided
        if ampl_ids is not None:
            chunk_data = chunk_data[ampl_ids, :]
        
        # Exclude clipped channels
        chunk_data = chunk_data[~clipped_mask, :]
        
        # Apply bandpass filter to each channel
        filtered_data = np.zeros_like(chunk_data, dtype=np.float32)
        for ch in range(chunk_data.shape[0]):
            filtered_data[ch, :] = signal.sosfiltfilt(sos, chunk_data[ch, :])
        
        # Calculate RMS per channel for this chunk
        rms = np.sqrt(np.mean(filtered_data**2, axis=1))
        rms_all_chunks.append(rms)
        
        # Generate and save example traces plot for this chunk
        if output_plot_path is not None:
            fig = plot_example_traces_with_rms(filtered_data, chunk_size, 
                                               chunk_num=i+1, n_examples=4)
            fig.savefig(output_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"\rProcessed chunk {i+1}/{n_chunks}", end='', flush=True)
    
    print("\nDone!")
    
    rms_all_chunks = np.array(rms_all_chunks)  # Shape: (n_chunks, n_channels)
    # rms_per_channel = np.mean(rms_all_chunks, axis=0)  # Average over chunks

    print("✅ RMS must be combined as a quadratic (power) mean")
    rms_per_channel = np.sqrt(np.mean(rms_all_chunks**2, axis=0))
    
    # Additional filter: exclude channels with very low RMS from final results
    low_rms_mask = rms_per_channel < 0.2  # Exclude if final RMS < 0.2 µV
    n_low_rms = np.sum(low_rms_mask)
    if n_low_rms > 0:
        print(f"Found {n_low_rms} additional channels with very low final RMS (< 0.2 µV), excluding from results")
        print(f"  Low RMS channel range: [{rms_per_channel[low_rms_mask].min():.4f}, {rms_per_channel[low_rms_mask].max():.4f}] µV")
        print(f"  Remaining channels RMS range: [{rms_per_channel[~low_rms_mask].min():.4f}, {rms_per_channel[~low_rms_mask].max():.4f}] µV")
        rms_per_channel = rms_per_channel[~low_rms_mask]
        rms_all_chunks = rms_all_chunks[:, ~low_rms_mask]
    
    return rms_per_channel, rms_all_chunks


def plot_example_traces_with_rms(filtered_data, chunk_size, sos=None, chunk_num=0, n_examples=3):
    """
    Plot example traces from a few channels with their RMS noise values.
    Shows the bandpass filtered traces for visualization.
    
    Args:
        filtered_data: Already filtered data array of shape (n_channels, chunk_size)
        chunk_size: Size of the chunk
        sos: Butterworth filter coefficients (not used, kept for compatibility)
        chunk_num: The chunk number (for title)
        n_examples: Number of example channels to plot
    """
    # Select random example channels
    n_channels = filtered_data.shape[0]
    example_channels = np.random.choice(n_channels, min(n_examples, n_channels), replace=False)
    
    # Calculate RMS for these channels
    rms_values = np.sqrt(np.mean(filtered_data**2, axis=1))
    
    # Set fixed y-axis limits
    y_lim = [-40, 40]
    
    # Plot
    fig, axes = plt.subplots(len(example_channels), 1, figsize=(12, 2.5*len(example_channels)))
    if len(example_channels) == 1:
        axes = [axes]
    
    time_axis = np.arange(chunk_size) / 20_000  # Convert to seconds
    
    for i, ch in enumerate(example_channels):
        axes[i].plot(time_axis, filtered_data[ch, :], linewidth=0.5, color='steelblue')
        axes[i].set_ylabel(f'Ch {ch}\n(µV)')
        axes[i].set_title(f'Channel {ch} - RMS: {rms_values[ch]:.2f} µV', fontweight='bold')
        axes[i].set_ylim(y_lim)
        # Draw horizontal lines for RMS amplitude
        axes[i].axhline(y=rms_values[ch], color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'+RMS')
        axes[i].axhline(y=-rms_values[ch], color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'-RMS')
        axes[i].grid(True, alpha=0.3)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Chunk {chunk_num}', fontsize=10, y=0.995)
    plt.tight_layout()
    return fig


def check_noise(subdir):
    rms_dict = {}  # Collect RMS per electrode for all files
    
    for fname in sorted(os.listdir(subdir)):
        if fname.endswith('raw.h5'):
            
            # Ensure output directory exists
            os.makedirs('live_figures', exist_ok=True)
            
            # call calculate_rms_noise for this file
            print(f"Processing {fname}...")
            output_path2 = 'live_figures/example_traces_with_rms.png'
            rms_per_channel, rms_all_chunks = calculate_rms_noise(subdir, fname, 
                                                                  n_samples=310200, 
                                                                  ampl_ids=None, 
                                                                  n_chunks=10, 
                                                                  chunk_duration_sec=1.0,
                                                                  output_plot_path=output_path2)
            
            # Store RMS per electrode
            rms_dict[fname] = rms_per_channel
            
            # Plot 1: RMS noise distribution violin plot
            print("\nPlotting RMS noise distribution...")
            fig1 = render_rms_violin(rms_all_chunks, 
                                     title=f"RMS Noise Distribution - {fname}")
            output_path1 = 'live_figures/rms_noise_distribution.png'
            fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
            plt.close(fig1)
            print(f"  Saved: {output_path1}")
            print(f"  Traces plot (updated each chunk): {output_path2}")
    
    # Plot 3: RMS per electrode for all files
    if rms_dict:
        print("\nPlotting RMS per electrode across all files...")
        fig3 = plot_rms_per_electrode(rms_dict, 
                                      title="RMS Noise per Electrode - All Recordings")
        output_path3 = 'live_figures/rms_per_electrode.png'
        fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"  Saved: {output_path3}")
            

def main():
    L = Logger()
    L.init_logger(None, None, "DEBUG")
    nas_dir = device_paths()[0]
    HEADSTAGE_DEVICE_NAME = 'MEA1K22'
    rec_name = f'Bond2_BothHalfs_ShubhamW3_16Shank_noisecheck'
    
    subdir = f"{nas_dir}/devices/headstage_devices/{HEADSTAGE_DEVICE_NAME}/recordings/{rec_name}"
    if not os.path.exists(subdir):
        print(f"Error: {os.path.join(subdir)} does not exist.")
        exit()
    
    check_noise(subdir)
    # vis_connectivity(subdir, input_ampl_mV, cmap_scaler=1)
    # create_implant_dir(subdir, nas_dir, HEADSTAGE_DEVICE_NAME, IMPLANT_DEVICE_NAME)
    
    
if __name__ == "__main__":
    main()