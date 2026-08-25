import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorsys
import matplotlib.colorbar as mcolorbar
# from mea1k_connectivity_scripts.analyze_shortcut_stim import get_aggr_results
# from mea1k_connectivity_scripts.analyze_shortcut_stim import build_connectivity_graph

from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.collections import LineCollection

from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

CHIP_W, CHIP_H = 220, 120

def draw_mea1k(bg='black', el_color='#111111', mapping=None, cmap_scaler=1, fast=False, 
               draw_pads_if_available=True):
    fig, ax = plt.subplots(figsize=(3850/300, 2100/300), dpi=300, facecolor='none')
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    # fig.patch.set_facecolor('black')

    if isinstance(bg, np.ndarray):
        ax.imshow(bg)
    else:
        ax.set_facecolor(bg)
    
    # draw 26400 colors from hsv colormap
    # cmap = plt.cm.get_cmap('hsv', 26400)
    if el_color == 'hsv':
        cmap = plt.get_cmap('hsv', 26400)
        colors = [list(col) for col in cmap(np.linspace(0, 1, 26400))]
    elif isinstance(el_color, list):
        colors = el_color
    else:
        colors = [el_color]*26400

    if fast and mapping is not None:
        # do scatter instead
        # draw = mapping.sort_values('mea1k_el')
        ax.scatter((mapping.mea1k_el % 220) * 17.5 +2.5, (mapping.mea1k_el // 220) * 17.5 +6, 
                   c=mapping.mea1k_connectivity*cmap_scaler, s=4, marker='s', cmap='gray', vmin=0, vmax=1,
                   edgecolors='none', alpha=1)
        ax.set_ylim(2100, 0)
        ax.set_xlim(0,3850)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        
        if draw_pads_if_available and 'pad_x_aligned' in mapping.columns and 'pad_y_aligned' in mapping.columns:
            pad_x = mapping['pad_x_aligned'].values
            pad_y = mapping['pad_y_aligned'].values
            c = mapping['pad_metal'].map({1: 'hotpink', 2: 'red'}).values
            ax.scatter(pad_x, pad_y, c='none', s=70, marker='o', 
                       edgecolors=c, alpha=.5, linewidth=0.1)
        return (fig, ax), None
    
    i = 0
    recs = []
    for y in np.arange(6, 2100, 17.5):
        for x in np.arange(2.5, 3850, 17.5):
            recs.append(plt.Rectangle((x, y), 9, 5, facecolor=colors[i], 
                                      edgecolor='none', alpha=.7))
            if mapping is not None:
                # change color to connectivity
                recs[-1].set_facecolor((1, 1, 1))
                conn = mapping[mapping['mea1k_el'] == i].mea1k_connectivity
                if pd.isna(conn).all():
                    print("Missing connectivity measurement for electrode", i)
                    recs[-1].set_facecolor((0,0,0))
                    recs[-1].set_edgecolor((.6, .6, 0)) # yellow
                else:
                    alpha = np.clip(conn.item()*cmap_scaler, 0, 1)
                    recs[-1].set_alpha(alpha)
            i += 1

    [ax.add_patch(rec) for rec in recs]
    ax.set_ylim(2100, 0)
    ax.set_xlim(0,3850)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    [ax.spines[spine].set_visible(False) for spine in ax.spines]
    return (fig, ax), recs

def adjust_saturation(rgb_color, offset):
    # Convert RGB (0-1 scale) to HSV
    r, g, b = rgb_color
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    # Adjust saturation by offset, ensuring it stays within [0, 1]
    s = max(0.0, min(1.0, s - offset))
    # Convert back to RGB
    return colorsys.hsv_to_rgb(h, s, v)

def _desaturate(color, amount=0.5):
    """Return `color` with its saturation scaled by `amount` (0=gray, 1=unchanged)."""
    r, g, b = mcolors.to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h, s * amount, v)

def _link_colors(data, valid, t0, t1, y0, y1, metal_colors, by_metal,
                 sat=0.75, alpha=0.15):
    """RGBA colors for the connecting lines between two datasets.

    When `by_metal` (and a 'pad_metal' column is present), each line takes the
    electrode's metal color, desaturated with a low constant alpha - thousands of
    lines overlap, so delta-based alpha scaling would just saturate to a solid
    block. Otherwise falls back to gray with alpha scaled by the magnitude of change.
    """
    n = int(valid.sum())
    if by_metal and 'pad_metal_'+t0 in data.columns:
        metal = data['pad_metal_'+t0].values[valid].astype(float)
        if 'pad_metal_'+t1 in data.columns:
            m1 = data['pad_metal_'+t1].values[valid].astype(float)
            metal = np.where(np.isnan(metal), m1, metal)
        rgba = {m: (*_desaturate(c, sat), alpha) for m, c in metal_colors.items()}
        gray = (0.5, 0.5, 0.5, alpha)
        return np.array([gray if np.isnan(m) else rgba.get(int(m), gray) for m in metal])

    delta = np.abs(y1[valid] - y0[valid])
    span = delta.max() - delta.min() if n else 0
    norm = (delta - delta.min()) / span if span else np.zeros(n)
    colors = np.zeros((n, 4))
    colors[:, :3] = 0.5
    colors[:, 3] = 0.2 + 0.4 * norm  # low delta -> 0.2, high delta -> 0.6
    return colors

def stim_unit_wise_vis_impedance(aggr_df, output_dir=None):
    aggr_df = aggr_df[aggr_df.impedance_Ohm > 0].copy()
    aggr_df['imp_kohm'] = aggr_df.impedance_Ohm / 1000
    
    # Calculate mean impedance per stim unit and sort
    mean_imp = aggr_df.groupby('stim_unit')['imp_kohm'].mean().sort_values()
    ordered_stim_units = mean_imp.index.tolist()
    ordered_stim_units = np.arange(32)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot each stim unit's data with x-jitter
    for x_pos, su in enumerate(ordered_stim_units):
        subset = aggr_df[aggr_df.stim_unit == su]
        jitter = np.random.normal(0, 0.15, size=len(subset))
        ax.scatter(x_pos + jitter, subset.imp_kohm, alpha=0.6, s=15, c=subset.stim_unit, 
                   cmap='tab20', vmin=0, vmax=31, )
        
    ax.set_xticks(range(len(ordered_stim_units)))
    ax.set_xticklabels(ordered_stim_units)
    ax.set_yscale('log')
    ax.set_xlabel('Stim Unit (Sorted by Mean Impedance)')
    ax.set_ylabel('Impedance (kOhm)')
    plt.title('Stim Unit-wise Impedance')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    # plt.savefig(f"./live_figures/stim_unit_wise_impedance.png")
    

def create_stim_sine_sequence_dummy(
    dac_id=0,
    amplitude=25,
    f=1000,
    ncycles=100,
    nreps=1,
    voltage_conversion=False,
    current_conversion=None,
    center_around=512,
    sample_rate=20_000,
    plot=False,
    ax=None,
):

    if voltage_conversion:
        # Provide the conversion explicitly when using this function outside MaxLab.
        raise ValueError("Pass amplitude already converted to bits, or use `current_conversion`.")

    if current_conversion is not None:
        amplitude = int(amplitude / current_conversion)
        print(f"Current conversion: {current_conversion} uA/bit, amplitude: {amplitude} bits")

    n_samples_per_cycle = max(1, int(sample_rate / f))
    t = np.linspace(0, 1, n_samples_per_cycle, endpoint=False)
    sine_wave = np.round(amplitude * np.sin(2 * np.pi * t)).astype(int)
    clipped_values = np.clip(center_around + sine_wave, 0, 1023)

    sequence = np.tile(clipped_values, ncycles * nreps)
    total_duration = len(sequence) / sample_rate
    time_vector = np.arange(len(sequence)) / sample_rate

    fig = None
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        else:
            fig = ax.figure

        ax.step(time_vector, sequence, where="post", lw=1.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("DAC value")
        ax.set_title(
            f"Generated sine sequence: {f} Hz, {ncycles} cycles x {nreps} reps, {total_duration:.3f} s"
        )
        # ax.set_ylim(0, 1023)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return time_vector, sequence, fig, ax

def plot_impedance_analysis(rec_dir, aggr_df, log_scale=True, use_imp_connectivity=False, 
                            use_external_impedance=False,
                            skip_bottom=True, ylims=None, color_by_metal=False):
    """Plot impedance analysis with connectivity, spatial distribution, and unit-wise impedance.
    
    Parameters:
    -----------
    rec_dir : str
        Recording directory name for title
    aggr_df : pd.DataFrame
        Merged impedance data with shank info
    log_scale : bool
        Apply log scale to impedance y-axes (default: True)
    use_imp_connectivity : bool
        Use 'connectivity' column instead of 'mea1k_connectivity' (default: False)
    skip_bottom : bool
        Skip plotting the bottom half of the spatial distribution (default: False)
    ylims : tuple
        Y-axis limits for impedance plots (default: (10, 1_400_000))
    """
    plt.close('all')
    fig = plt.figure(figsize=(11, 18))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1.1, 1, 1], hspace=0.35, wspace=0.25)
    
    conn_col = 'connectivity' if use_imp_connectivity else 'mea1k_connectivity'
    imp_col = 'impedance_Ohm' if not use_external_impedance else 'external_impedance_Ohm'
    
    # # Phase shift vs impedance
    ax = fig.add_subplot(gs[0, 0])
    if color_by_metal:
        for pad_metal_val, color in [(1, 'purple'), (2, 'red')]:
            mask = aggr_df['pad_metal'] == pad_metal_val
            ax.scatter(aggr_df.loc[mask, 'phase_shift_deg'], aggr_df.loc[mask, imp_col], 
                    alpha=0.15, s=15, c=color, label=f'pad_metal={pad_metal_val}',
                    edgecolors='none')
    else:
        ax.scatter(aggr_df.loc[:, 'phase_shift_deg'], aggr_df.loc[:, imp_col], 
                   alpha=0.15, s=15, )
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlabel("Phase Shift (deg)", fontsize=10, weight='bold')
    ax.set_ylabel("Impedance (Ohm)", fontsize=10, weight='bold')
    ax.set_title(f"Phase Shift vs Impedance{'(Log)' if log_scale else ''}", fontsize=11, pad=8)
    ax.set_xlim(-100, 100)
    left_min, right_max = ax.get_xlim()
    rect = plt.Rectangle((left_min, 30_000), right_max - left_min, 40_000, linewidth=1.5, edgecolor='r', 
                         facecolor='gray', linestyle='--', alpha=0.3, label='30-70 kOhm')
    ax.add_patch(rect)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    if ylims is not None:
        ax.set_ylim(ylims)
    
    
    # Connectivity vs Impedance
    ax = fig.add_subplot(gs[0, 1])
    if color_by_metal:
        for pad_metal_val, color in [(1, 'purple'), (2, 'red')]:
            mask = aggr_df['pad_metal'] == pad_metal_val
            ax.scatter(aggr_df.loc[mask.values, conn_col], aggr_df.loc[mask.values, imp_col], 
                        alpha=0.15, s=15, color=color, label=f'pad_metal={pad_metal_val}',
                        edgecolors='none')
    else:
        ax.scatter(aggr_df.loc[:, conn_col], aggr_df.loc[:, imp_col], 
                    alpha=0.15, s=15)
    if log_scale:
        ax.set_yscale('log')
        
    ax.set_xlabel("Connectivity", fontsize=10, weight='bold')
    ax.set_ylabel("Impedance (Ohm)", fontsize=10, weight='bold')
    ax.set_title(f"Connectivity vs Impedance{'(Log)' if log_scale else ''}", fontsize=11, pad=8)
    ax.grid(True, alpha=0.3)
    left_min_conn, right_max_conn = ax.get_xlim()
    rect_conn = plt.Rectangle((left_min_conn, 30_000), right_max_conn - left_min_conn, 40_000, linewidth=1.5, edgecolor='r', 
                         facecolor='gray', linestyle='--', alpha=0.3, label='30-70 kOhm')
    ax.add_patch(rect_conn)
    ax.legend(fontsize=8)
    if ylims is not None:
        ax.set_ylim(ylims)
    
    # Impedance histogram (90 degree flipped)
    ax_hist = fig.add_subplot(gs[0, 2])
    if log_scale:
        bins = np.logspace(np.log10(aggr_df[imp_col].min()), 
                          np.log10(aggr_df[imp_col].max()), 100)
    else:
        bins = 1000
    
    if color_by_metal:
        for pad_metal_val, color in [(1, 'purple'), (2, 'red')]:
            mask = aggr_df['pad_metal'] == pad_metal_val
            ax_hist.hist(aggr_df.loc[mask, imp_col], bins=bins, orientation='horizontal', 
                        alpha=0.6, color=color, edgecolor='none', label=f'pad_metal={pad_metal_val}')
    else:
        ax_hist.hist(aggr_df[imp_col], bins=bins, orientation='horizontal', alpha=0.7, color='purple', edgecolor='none')
    
    if log_scale:
        ax_hist.set_yscale('log')
        
    if ylims is not None:
        ax_hist.set_ylim(ylims)
    ax_hist.set_ylabel("Impedance (Ohm)", fontsize=10, weight='bold')
    ax_hist.set_xlabel("Count", fontsize=10, weight='bold')
    bottom_min_hist, top_max_hist = ax_hist.get_xlim()
    rect_hist = plt.Rectangle((bottom_min_hist, 30_000), top_max_hist - bottom_min_hist, 40_000, linewidth=1.5, edgecolor='r', 
                         facecolor='gray', linestyle='--', alpha=0.3, label='30-70 kOhm')
    ax_hist.add_patch(rect_hist)
    ax_hist.legend(fontsize=8)
    
    # Spatial CMOS distribution (never log)
    ax = fig.add_subplot(gs[1, :])
    heatmap_x = (aggr_df.electrode % 220) * 17.5 + 17.5/4
    heatmap_y = (aggr_df.electrode // 220) * 17.5 + 17.5/4
    if ylims is not None:
        scatter = ax.scatter(heatmap_x,
                             heatmap_y,
                             c=aggr_df[imp_col], s=8, marker='s',
                             vmin=ylims[0], vmax=ylims[1],
                             cmap='plasma', edgecolors='none')
        cbar = plt.colorbar(scatter, ax=ax, label='Impedance (Ohm)', pad=0.02)
    else:
        cmap = plt.cm.colors.ListedColormap(plt.cm.turbo(np.linspace(0.01, 0.8, 256)))
        scatter = ax.scatter(heatmap_x,
                             heatmap_y,
                             c=aggr_df[imp_col], s=8, marker='s', vmin=10_000, vmax=1_400_000,
                             cmap=cmap, edgecolors='none')
        cbar = plt.colorbar(scatter, ax=ax, label='Impedance (Ohm)', pad=0.02)
        cbar.set_ticks([10_000, 50_000, 100_000, 200_000, 400_000, 600_000, 800_000, 1_400_000])
        cbar.set_ticklabels(['10K', '50K', '100K', '200K', '400K', '600K', '800K', '1.4M'], fontsize=8)
    ax.set_ylim(2100, 0)
    ax.set_xlim(0, 3850)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('CMOS Spatial Distribution', fontsize=11, pad=8)
    
    # Prepare unit-wise data
    df_clean = aggr_df[aggr_df[imp_col] > 0].copy()
    df_clean['imp_kohm'] = df_clean[imp_col] / 1000
    
    if not skip_bottom:
        # Stim unit-wise impedance (full width)
        ax = fig.add_subplot(gs[2, :])
        cmap_tab20 = plt.cm.tab20
        for x_pos, su in enumerate(np.arange(32)):
            subset = df_clean[df_clean.stim_unit == su]
            if len(subset) > 0:
                jitter = np.random.normal(0, 0.15, size=len(subset))
                ax.scatter(x_pos + jitter, subset.imp_kohm, alpha=0.6, s=15, c=subset.stim_unit, 
                        cmap=cmap_tab20, vmin=0, vmax=31, edgecolors='none')
        ax.set_xticks(range(32))
        ax.set_xticklabels(range(32), fontsize=8)
        if log_scale:
            ax.set_yscale('log')
        ax.set_xlabel('Stim Unit', fontsize=10, weight='bold')
        ax.set_ylabel('Impedance (kOhm)', fontsize=10, weight='bold')
        ax.set_title(f'Stim Unit-wise{"(Log)" if log_scale else ""}', fontsize=11, pad=8)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        sm = plt.cm.ScalarMappable(cmap=cmap_tab20, norm=plt.Normalize(vmin=0, vmax=31))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Stim Unit', pad=0.02)
        
        # Shank-wise impedance (full width)
        ax = fig.add_subplot(gs[3, :])
        if 'shank_id' in df_clean.columns:
            shanks = np.sort(df_clean[df_clean.shank_id.notna()].shank_id.unique())
            for x_pos, shank in enumerate(shanks):
                subset = df_clean[df_clean.shank_id == shank]
                if len(subset) > 0:
                    jitter = np.random.normal(0, 0.15, size=len(subset))
                    ax.scatter(x_pos + jitter, subset.imp_kohm, alpha=0.6, s=15, c=subset.stim_unit, 
                            cmap=cmap_tab20, vmin=0, vmax=31, edgecolors='none')
            ax.set_xticks(range(len(shanks)))
            ax.set_xticklabels([int(s) for s in shanks], fontsize=8)
            ax.set_xlabel('Shank ID', fontsize=10, weight='bold')
        if log_scale:
            ax.set_yscale('log')
        ax.set_ylabel('Impedance (kOhm)', fontsize=10, weight='bold')
        ax.set_title(f'Shank-wise{"(Log)" if log_scale else ""}', fontsize=11, pad=8)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        sm = plt.cm.ScalarMappable(cmap=cmap_tab20, norm=plt.Normalize(vmin=0, vmax=31))
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, label='Stim Unit', pad=0.02)
        cbar = plt.colorbar(sm, ax=ax, label='Stim Unit', pad=0.02)
    fig.suptitle(f'Recording: {rec_dir}', fontsize=12, fontweight='bold', y=0.998)
    return fig
    
def stim_unit_wise_vis_impedance(aggr_df, aggr_df2=None, title1=None, title2=None):
    """
    Plot stim unit-wise impedance for one or two datasets.
    
    Parameters:
    -----------
    aggr_df : pd.DataFrame
        First impedance dataframe
    aggr_df2 : pd.DataFrame, optional
        Second impedance dataframe for comparison
    title1, title2 : str, optional
        Titles for the plots
    """
    plt.close('all')
    
    # Create 32 unique colors using HSV colormap for stim units
    colors_32 = plt.cm.hsv(np.linspace(0, 1, 32))
    color_dict = {su: colors_32[su] for su in range(32)}
    
    # Setup figure
    if aggr_df2 is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10))
    else:
        fig, ax1 = plt.subplots(figsize=(16, 4.5))
        ax2 = None
        ax3 = None
    
    # Plot first dataset
    df1 = aggr_df[aggr_df.impedance_Ohm > 0].copy()
    df1['imp_kohm'] = df1.impedance_Ohm / 1000
    
    for x_pos, su in enumerate(range(32)):
        subset = df1[df1.stim_unit == su]
        if len(subset) > 0:
            jitter = np.random.normal(0, 0.08, size=len(subset))
            ax1.scatter(x_pos + jitter, subset.imp_kohm, alpha=0.6, s=15, 
                       color=color_dict[su], edgecolors='none',
                       label="MEA1K electrodes per Stim Unit" if x_pos == 0 else "")
    
    ax1.set_xticks(range(32))
    ax1.set_xticklabels(range(32))
    ax1.set_yscale('log')
    ax1.set_xlabel('Stim Unit', fontsize=11)
    ax1.set_ylabel('Impedance (kOhm)', fontsize=11)
    if title1:
        ax1.set_title(title1, fontsize=12)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=9)
    
    # Plot second dataset if provided
    if ax2 is not None:
        df2 = aggr_df2[aggr_df2.impedance_Ohm > 0].copy()
        df2['imp_kohm'] = df2.impedance_Ohm / 1000
        
        for x_pos, su in enumerate(range(32)):
            subset = df2[df2.stim_unit == su]
            if len(subset) > 0:
                jitter = np.random.normal(0, 0.08, size=len(subset))
                ax2.scatter(x_pos + jitter, subset.imp_kohm, alpha=0.6, s=15, 
                           color=color_dict[su], edgecolors='none')
        
        ax2.set_xticks(range(32))
        ax2.set_xticklabels(range(32))
        ax2.set_yscale('log')
        ax2.set_xlabel('Stim Unit', fontsize=11)
        ax2.set_ylabel('Impedance (kOhm)', fontsize=11)
        if title2:
            ax2.set_title(title2, fontsize=12)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # Plot difference (df2 - df1) for matching electrodes
        df1_renamed = df1.rename(columns={'impedance_Ohm': 'imp1_Ohm', 'imp_kohm': 'imp1_kohm'})
        df2_renamed = df2.rename(columns={'impedance_Ohm': 'imp2_Ohm', 'imp_kohm': 'imp2_kohm'})
        
        merged = pd.merge(df1_renamed[['electrode', 'stim_unit', 'imp1_kohm']], 
                         df2_renamed[['electrode', 'stim_unit', 'imp2_kohm']], 
                         on=['electrode', 'stim_unit'], how='inner')
        
        merged['diff_kohm'] = merged['imp1_kohm'] - merged['imp2_kohm']
        
        # Calculate mean difference for each stim unit to find top 3
        su_means = merged.groupby('stim_unit')['diff_kohm'].mean()
        top_3_sus = su_means.abs().nlargest(3).index.tolist()
        
        for x_pos, su in enumerate(range(32)):
            subset = merged[merged.stim_unit == su]
            if len(subset) > 0:
                jitter = np.random.normal(0, 0.08, size=len(subset))
                ax3.scatter(x_pos + jitter, subset.diff_kohm, alpha=0.6, s=15, 
                           color=color_dict[su], edgecolors='none')
                # Plot median for this stim unit
                median_su = subset['diff_kohm'].median()
                if su in top_3_sus:
                    # Top 3: use stim unit color with label
                    ax3.plot([x_pos - 0.25, x_pos + 0.25], [median_su, median_su], 
                            color='k', linewidth=2.5, alpha=0.9, 
                            label=f'StimUnit {su} (median: {su_means[su]:.2f})')
                else:
                    # Others: use black
                    ax3.plot([x_pos - 0.25, x_pos + 0.25], [median_su, median_su], 
                            color='black', linewidth=2, alpha=0.8)
        
        ax3.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.3)
        ax3.set_xticks(range(32))
        ax3.set_xticklabels(range(32))
        # ax3.set_yscale('log')
        ax3.set_xlabel('Stim Unit', fontsize=11)
        ax3.set_ylabel('Δ Impedance (kOhm)', fontsize=11)
        ax3.set_title(f'Impedance Difference ({title1} − {title2})\n(positive: {title1} > {title2},  negative: {title2} > {title1})', fontsize=12)
        ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax3.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def phase_impedance_comparison(aggr_df1, aggr_df2, aggr_df3=None, title1=None, title2=None, title3=None, color_by='phase'):
    """
    Compare phase-impedance profiles across 2-3 datasets.
    
    Parameters:
    -----------
    aggr_df1, aggr_df2 : pd.DataFrame
        First and second impedance dataframes
    aggr_df3 : pd.DataFrame, optional
        Third impedance dataframe for comparison
    title1, title2, title3 : str, optional
        Labels for each dataset
    color_by : str, optional
        Metric to color arrows by: 'phase', 'magnitude', or 'impedance'
    """
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    dataframes = [aggr_df1, aggr_df2]
    if aggr_df3 is not None:
        dataframes.append(aggr_df3)
    
    titles = [title1, title2, title3][:len(dataframes)]
    
    # Connect corresponding points across datasets with arrows (match on electrode column)
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    if len(dataframes) >= 2:
        # Draw arrows from df1 to df2
        common_12 = set(dataframes[0]['electrode'].unique()) & set(dataframes[1]['electrode'].unique())
        
        # Calculate color values for all arrows
        color_vals = []
        for elec in common_12:
            p1 = dataframes[0][dataframes[0]['electrode'] == elec]
            p2 = dataframes[1][dataframes[1]['electrode'] == elec]
            
            if len(p1) > 0 and len(p2) > 0:
                x1, y1 = p1['phase_shift_deg'].iloc[0], p1['impedance_Ohm'].iloc[0]
                x2, y2 = p2['phase_shift_deg'].iloc[0], p2['impedance_Ohm'].iloc[0]
                
                if color_by == 'phase':
                    val = abs(x2 - x1)
                    color_label = 'Phase Change (deg)'
                elif color_by == 'impedance':
                    val = abs(y2 - y1)
                    color_label = 'Impedance Change (Ohm)'
                else:  # magnitude
                    val = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    color_label = 'Arrow Magnitude'
                
                color_vals.append(val)
        
        if color_vals:
            norm = Normalize(vmin=min(color_vals), vmax=max(color_vals))
            cmap = cm.get_cmap('viridis')
            
            # Draw arrows with coloring
            for elec, color_val in zip(common_12, color_vals):
                p1 = dataframes[0][dataframes[0]['electrode'] == elec]
                p2 = dataframes[1][dataframes[1]['electrode'] == elec]
                
                x1, y1 = p1['phase_shift_deg'].iloc[0], p1['impedance_Ohm'].iloc[0]
                x2, y2 = p2['phase_shift_deg'].iloc[0], p2['impedance_Ohm'].iloc[0]
                
                color = cmap(norm(color_val))
                arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                                       color=color, alpha=0.7, linewidth=1.2, 
                                       mutation_scale=12, zorder=1)
                ax.add_patch(arrow)
            
            # Add colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(color_label, fontsize=10, weight='bold')
        
        # If 3 datasets, draw arrows from df2 to df3 in gray
        if len(dataframes) == 3:
            common_23 = set(dataframes[1]['electrode'].unique()) & set(dataframes[2]['electrode'].unique())
            for elec in common_23:
                p2 = dataframes[1][dataframes[1]['electrode'] == elec]
                p3 = dataframes[2][dataframes[2]['electrode'] == elec]
                
                if len(p2) > 0 and len(p3) > 0:
                    x2, y2 = p2['phase_shift_deg'].iloc[0], p2['impedance_Ohm'].iloc[0]
                    x3, y3 = p3['phase_shift_deg'].iloc[0], p3['impedance_Ohm'].iloc[0]
                    arrow = FancyArrowPatch((x2, y2), (x3, y3), arrowstyle='->', 
                                           color='gray', alpha=0.25, linewidth=0.8, 
                                           mutation_scale=10, zorder=0)
                    ax.add_patch(arrow)
    
    ax.set_yscale('log')
    ax.set_xlabel("Phase Shift (deg)", fontsize=10, weight='bold')
    ax.set_ylabel("Impedance (Ohm)", fontsize=10, weight='bold')
    
    # Create title
    title_str = " vs ".join([t for t in titles if t])
    ax.set_title(f"Phase Shift vs Impedance: {title_str}", fontsize=11, pad=8)
    
    ax.set_xlim(-90, 90)
    
    # Add reference rectangle
    left_min, right_max = ax.get_xlim()
    rect = plt.Rectangle((left_min, 30_000), right_max - left_min, 40_000, linewidth=1.5, edgecolor='r', 
                         facecolor='gray', linestyle='--', alpha=0.3, label='30-70 kOhm')
    ax.add_patch(rect)
    
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_many_connectivities(conn_dfs, title_list, color_metal=False, ylims=None,
                                link_color_by_metal=None):
    """
    Compare connectivity distributions across multiple datasets.

    Parameters:
    -----------
    conn_dfs : list of pd.DataFrame
        List of connectivity dataframes to compare
    title_list : list of str, optional
        Titles for each dataset (used in legend)
    color_metal : bool, optional
        Whether to color points by pad metal type (default: False) Needs to be joined in from implant_mapping.
        When True, a second row of violin plots (one per metal, per dataset) is drawn below the scatter.
    link_color_by_metal : bool or None, optional
        Whether to color the connecting lines by pad metal (desaturated, low constant
        alpha since thousands of lines overlap). Defaults to the value of `color_metal`.
    """
    plt.close('all')
    if link_color_by_metal is None:
        link_color_by_metal = color_metal

    data = []
    for df, title in zip(conn_dfs, title_list):
        if 'mea1k_connectivity' not in df.columns or 'mea1k_el' not in df.columns:
            raise ValueError("Each dataframe must contain 'mea1k_connectivity' and 'mea1k_el' columns.")
        # append the title to each column name to avoid overlap
        df = df.set_index('mea1k_el')
        df = df.rename(columns=dict(zip(df.columns, [f"{col}_{title}" for col in df.columns])))
        data.append(df)
    data = pd.concat(data, axis=1).sort_index()

    metal_colors = {1: 'purple', 2: 'red'}
    if color_metal:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3*len(conn_dfs), 7), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 2]})
    else:
        fig, ax = plt.subplots(figsize=(3*len(conn_dfs), 4))
        ax2 = None

    for i, title in enumerate(title_list):
        df = data[[col for col in data.columns if col.endswith(f"_{title}")]].copy()
        df = df.sort_values('mea1k_connectivity_'+title)
        args = {'color': 'blue'} if not color_metal else {'c': df['pad_metal_'+title].map({**metal_colors, np.nan: 'gray'}).values, 'cmap': None}
        ax.scatter(df.index/(26400*2) + i, df['mea1k_connectivity_'+title].values,
                   alpha=0.2, label=title, s=4, **args)

    # connect each electrode's connectivity across consecutive datasets
    x_base = data.index.values / (26400*3)
    for i, (t0, t1) in enumerate(zip(title_list[:-1], title_list[1:])):
        y0 = data['mea1k_connectivity_'+t0].values
        y1 = data['mea1k_connectivity_'+t1].values
        valid = ~(np.isnan(y0) | np.isnan(y1))
        p0 = np.column_stack([x_base[valid] + i,     y0[valid]])
        p1 = np.column_stack([x_base[valid] + i + 1, y1[valid]])
        segments = np.stack([p0, p1], axis=1)
        colors = _link_colors(data, valid, t0, t1, y0, y1, metal_colors,
                              link_color_by_metal)
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.2, zorder=0))

    # center of each dataset's point cluster along x (points span index/(26400*2))
    x_off = data.index.values / (26400*2)
    center = (np.nanmin(x_off) + np.nanmax(x_off)) / 2

    # violin plots per metal, per dataset, below the scatter
    if color_metal:
        for i, title in enumerate(title_list):
            for j, (metal, color) in enumerate(metal_colors.items()):
                mask = data['pad_metal_'+title] == metal
                vals = data.loc[mask, 'mea1k_connectivity_'+title].dropna().values
                if len(vals) == 0:
                    continue
                pos = i + center + (0.09 if j else -0.09)  # metal 1 left, metal 2 right, slight overlap
                parts = ax2.violinplot(vals, positions=[pos], widths=0.22,
                                       showmeans=True, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor(color)
                    pc.set_alpha(0.5)
                if 'cmeans' in parts:
                    parts['cmeans'].set_color(color)
        ax2.set_ylabel('Connectivity')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
        legend_handles = [Patch(facecolor=c, alpha=0.5, label=f'metal {m}')
                          for m, c in metal_colors.items()]
        ax2.legend(handles=legend_handles, fontsize=8, loc='upper right')

    ax.set_xlim(-.3, len(conn_dfs)-1 + np.nanmax(x_off) + .3)
    # tick/label under the center of each dataset's point cluster
    tick_ax = ax2 if color_metal else ax
    tick_ax.set_xticks(np.arange(len(title_list)) + center)
    tick_ax.set_xticklabels(title_list)
    ax.set_ylabel('Connectivity')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    if ylims is not None:
        ax.set_ylim(ylims)
    return fig, ax
        




    
def compare_impedance(aggr_dfs, title_list, show_scatter=True, color_metal=False,
                      log_scale=False, link_color_by_metal=None):
    """
    Compare impedance across multiple datasets with connecting lines.

    Follows the layout of compare_many_connectivities: each dataset occupies its
    own x-cluster, electrodes present in consecutive datasets are joined by lines,
    and the lines are colored by the magnitude of the impedance change.

    Parameters:
    -----------
    aggr_dfs : list of pd.DataFrame
        Impedance dataframes to compare. Each must contain 'electrode' and
        'impedance_Ohm' columns. A 'pad_metal' column is required when
        color_metal is True.
    title_list : list of str
        Titles for each dataset (used for x-tick labels / legend).
    show_scatter : bool, optional
        Whether to show scatter points for individual electrodes (default: True).
    color_metal : bool, optional
        Whether to color points by pad metal type (default: False). Needs
        'pad_metal' to be joined in from implant_mapping. When True, a second row
        of violin plots (one per metal, per dataset) is drawn below the scatter.
    log_scale : bool, optional
        Whether to use a logarithmic scale for the y-axis (default: False).
    link_color_by_metal : bool or None, optional
        Whether to color the connecting lines by pad metal (desaturated, low constant
        alpha since thousands of lines overlap). Defaults to the value of `color_metal`.
    """
    plt.close('all')
    if link_color_by_metal is None:
        link_color_by_metal = color_metal

    metal_colors = {1: 'purple', 2: 'red'}

    # build a combined frame indexed by electrode with per-dataset suffixed columns
    data = []
    for df, title in zip(aggr_dfs, title_list):
        if 'impedance_Ohm' not in df.columns or 'electrode' not in df.columns:
            raise ValueError("Each dataframe must contain 'electrode' and 'impedance_Ohm' columns.")
        keep = ['impedance_Ohm'] + (['pad_metal'] if 'pad_metal' in df.columns else [])
        df = df.set_index('electrode')[keep]
        df = df[~df.index.duplicated(keep='first')]
        df = df.rename(columns={col: f"{col}_{title}" for col in df.columns})
        data.append(df)
    data = pd.concat(data, axis=1).sort_index()

    if color_metal:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3*len(aggr_dfs), 7), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 2]})
    else:
        fig, ax = plt.subplots(figsize=(3*len(aggr_dfs), 4))
        ax2 = None

    # normalized within-cluster x position per electrode (shared across datasets)
    idx = data.index.values.astype(float)
    span_idx = idx.max() - idx.min()
    spread = 0.6
    x_base = (idx - idx.min()) / span_idx * spread if span_idx else np.zeros_like(idx)
    center = spread / 2

    # scatter per dataset
    for i, title in enumerate(title_list):
        if not show_scatter:
            break
        y = data['impedance_Ohm_'+title].values
        if color_metal and 'pad_metal_'+title in data.columns:
            c = data['pad_metal_'+title].map({**metal_colors, np.nan: 'gray'}).values
            # shuffle draw order so neither metal is fully overdrawn by the other
            order = np.random.permutation(len(y))
            ax.scatter((x_base + i)[order], y[order], c=c[order], alpha=0.6, s=2, zorder=3)
        else:
            ax.scatter(x_base + i, y, color='steelblue', alpha=0.6, s=20, label=title, zorder=3)

    # connect each electrode's impedance across consecutive datasets; lines colored
    # by metal (desaturated) when requested, else gray with alpha scaled by change
    for i, (t0, t1) in enumerate(zip(title_list[:-1], title_list[1:])):
        y0 = data['impedance_Ohm_'+t0].values
        y1 = data['impedance_Ohm_'+t1].values
        valid = ~(np.isnan(y0) | np.isnan(y1))
        p0 = np.column_stack([x_base[valid] + i,     y0[valid]])
        p1 = np.column_stack([x_base[valid] + i + 1, y1[valid]])
        segments = np.stack([p0, p1], axis=1)
        if len(segments) == 0:
            print(f"No valid segments to connect between {t0} and {t1}.")
            continue
        colors = _link_colors(data, valid, t0, t1, y0, y1, metal_colors,
                              link_color_by_metal)
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.5, zorder=0))

    # violin plots per metal, per dataset, below the scatter
    if color_metal:
        for i, title in enumerate(title_list):
            if 'pad_metal_'+title not in data.columns:
                continue
            for j, (metal, color) in enumerate(metal_colors.items()):
                mask = data['pad_metal_'+title] == metal
                vals = data.loc[mask, 'impedance_Ohm_'+title].dropna().values
                vals = vals[vals > 0]
                if len(vals) == 0:
                    continue
                pos = i + center + (0.09 if j else -0.09)  # metal 1 left, metal 2 right
                # KDE in log space with a tight bandwidth so the bimodal
                # (low/high impedance) structure is not smoothed away
                parts = ax2.violinplot(np.log10(vals), positions=[pos], widths=0.22,
                                       showmeans=True, showextrema=False, bw_method=0.15)
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor(color)
                    pc.set_alpha(0.5)
                if 'cmeans' in parts:
                    parts['cmeans'].set_color(color)
        ax2.set_ylabel('Impedance (Ohm)')
        ax2.yaxis.set_major_locator(MultipleLocator(1))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: rf'$10^{{{v:.0f}}}$'))
        ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
        legend_handles = [Patch(facecolor=c, alpha=0.5, label=f'metal {m}')
                          for m, c in metal_colors.items()]
        ax2.legend(handles=legend_handles, fontsize=8, loc='upper right')

    if log_scale:
        ax.set_yscale('log')
    ax.set_xlim(-.3, len(aggr_dfs)-1 + spread + .3)
    # tick/label under the center of each dataset's point cluster
    tick_ax = ax2 if color_metal else ax
    tick_ax.set_xticks(np.arange(len(title_list)) + center)
    tick_ax.set_xticklabels(title_list)
    ax.set_ylabel('Impedance (Ohm)', fontsize=10, weight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    if show_scatter and not color_metal:
        ax.legend(fontsize=10, loc='upper right')
    return fig, ax
    
def plot_ext_current_comparison(ext_cur_data, diff_max=2_000_000, title=None):
    alt_imp = ext_cur_data.amplitude_uV / ext_cur_data.external_current_uA  # Ohm
    imp_diff = ext_cur_data.impedance_Ohm - alt_imp
    cbar_label = "Chip - External Imp (Ohm)\nBlue: Chip imp too low\nRed: Chip imp too high"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    sc0 = axes[0].scatter(ext_cur_data.impedance_Ohm, alt_imp, c=imp_diff, cmap='coolwarm', vmin=-diff_max, vmax=diff_max, alpha=0.7)
    axes[0].plot([0, 2_000_000], [0, 2_000_000], 'k--', label='y=x')
    axes[0].set_xlabel('StimUnit (chip) impedance (Ohm)')
    axes[0].set_ylabel('External current sense impedance (Ohm)')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].legend()
    fig.colorbar(sc0, ax=axes[0]).set_label(cbar_label, fontweight='bold')

    sc1 = axes[1].scatter(ext_cur_data.current_uA, ext_cur_data.external_current_uA, c=imp_diff, cmap='coolwarm', vmin=-diff_max, vmax=diff_max)
    axes[1].plot([0, .2], [0, .2], 'k--')
    axes[1].set_xlabel('Current set via chip (uA)')
    axes[1].set_ylabel('True Current through GND (uA)')
    fig.colorbar(sc1, ax=axes[1]).set_label(cbar_label, fontweight='bold')

    sc2 = axes[2].scatter(ext_cur_data.current_uA, ext_cur_data.external_current_uA, c=alt_imp, vmin=10_000, vmax=5_000_000)
    axes[2].plot([0, .2], [0, .2], 'k--')
    axes[2].set_xlabel('Current set via chip (uA)')
    axes[2].set_ylabel('True Current through GND (uA)')
    fig.colorbar(sc2, ax=axes[2], label='Impedance (Ohm)')
    if title:
        fig.suptitle(title, fontweight='bold')
    # plt.show()
    return fig

def vis_mea1k_config(dirname, config_fname, implant_mapping=None):
    (fig, ax), els = draw_mea1k()

    config_paths = sorted(glob.glob(os.path.join(dirname, config_fname)))
    draw_data = pd.concat([pd.read_csv(p) for p in config_paths], ignore_index=True)
    if 'electrode' not in draw_data.columns:
        draw_data['electrode'] = draw_data['el']
    if 'stim' not in draw_data.columns:
        draw_data['stim'] = False

    rec_els = draw_data.electrode.astype(int).unique()
    stim_els = draw_data.loc[draw_data.stim.astype(bool), 'electrode'].astype(int).unique()

    if implant_mapping is None and os.path.exists(os.path.join(dirname, "implant_mapping.csv")):
        implant_mapping = pd.read_csv(os.path.join(dirname, "implant_mapping.csv"))

    if implant_mapping is not None:
        for _, row in implant_mapping.dropna(subset=['mea1k_el']).iterrows():
            el_i = int(row['mea1k_el'])
            mea1k_el_conn = np.clip(float(row.get('mea1k_connectivity', 0)), 0, 1)
            els[el_i].set_facecolor((1, 1, 1))
            els[el_i].set_alpha(min(0.45, 0.1 + 0.35 * mea1k_el_conn))

    for el_i in rec_els:
        el_rec = els[el_i]
        ax.scatter(el_rec.get_x() + 4.75, el_rec.get_y() + 4.75,
                   s=10, marker='s', color='gray', zorder=5)

    for el_i in stim_els:
        el_rec = els[el_i]
        ax.scatter(el_rec.get_x() + 4.75, el_rec.get_y() + 4.75,
                   s=12, marker='x', color='red', zorder=6, linewidths=0.8)

    ax.set_title(f"{os.path.basename(config_fname)} ({len(config_paths)} cfg)")
    plt.show()
    return fig, ax, els