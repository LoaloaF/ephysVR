import os
import sys

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths
# import parent dir with general modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LogNorm

from mea1k_modules.mea1k_visualizations import draw_mea1k

def build_connectivity_graph(data, thresh=0.5):
    """Directed edge list source->target (target lit up >= thresh when source stim'd).
    reciprocal=True iff reverse edge exists. short_island_id labels connected
    components of the SYMMETRIC subgraph (NaN for one-way edges).
    Long format -> join later on 'source' or 'target' (electrode_id)."""
    edges = []
    print("Building connectivity graph with threshold: ", thresh)
    for (c, tile), sub in data.groupby(['config', 'tile'], sort=False):
        stim_row = sub[sub.stim]
        if not len(stim_row):
            continue
        src = int(stim_row.electrode.iloc[0])
        resp = sub[(~sub.stim) & (sub.tile_connectivity >= thresh)]
        for el, conn in zip(resp.electrode, resp.tile_connectivity):
            edges.append((src, int(el), float(conn)))
    g = pd.DataFrame(edges, columns=['source', 'target', 'connectivity'])
    pairset = set(zip(g.source, g.target))
    g['reciprocal'] = [(t, s) in pairset for s, t in zip(g.source, g.target)]

    # union-find over reciprocal edges -> island id per electrode
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    for s, t in zip(g.source[g.reciprocal], g.target[g.reciprocal]):
        union(s, t)
    roots = {find(x) for x in parent}
    island_of = {r: i for i, r in enumerate(sorted(roots))}   # 0..n_islands-1
    g['short_island_id'] = [island_of[find(s)] if r else np.nan
                            for s, r in zip(g.source, g.reciprocal)]
    g['threshold'] = thresh
    return g

def get_aggr_results(processed_dir):
    out_fullfname = os.path.join(processed_dir, "extracted_sine_voltages.csv")

    csv_files = [
        f for f in os.listdir(processed_dir)
        if f.endswith('.csv') and f.startswith('el_config_')
    ]
    all_data = []
    print("Aggregating results from processed recordings...")
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(processed_dir, csv_file))
        df['config'] = csv_file.replace('.csv', '')
        all_data.append(df)
    if not all_data:
        return pd.DataFrame()
    aggr_data = pd.concat(all_data, ignore_index=True)
    # save extracted data for future use
    aggr_data.to_csv(out_fullfname, index=False)
    print(aggr_data)
    g = build_connectivity_graph(aggr_data)
    # save
    g.to_csv(os.path.join(processed_dir, "extracted_connectivity_graph.csv"), index=False)
    return aggr_data

def plot_amplitude_ridges(subdir, bins=60, n_groups=8, shorted_thresh=0.8,
                          min_stim_uV=None, dpi=100, rebuilt_graph=False):
    output_dir = os.path.join(subdir, "processed")
    if os.path.exists(os.path.join(output_dir, "extracted_sine_voltages.csv")) and not rebuilt_graph:
        data = pd.read_csv(os.path.join(output_dir, "extracted_sine_voltages.csv"))
    else:
        data = get_aggr_results(output_dir)
    configs = sorted(data.config.unique())

    # flag stim rows that were driven below min_stim_uV, then neuter their
    # stim flag so they drop out of build_connectivity_graph / n_shorted /
    # the chip panel like a config+tile that was never stimulated. Keep the
    # original flag around (stim_below_thresh) so the amp panel can still
    # show them, just in a different color.
    if min_stim_uV is not None:
        below_thresh = data.stim & (data.sine_voltage_uV < min_stim_uV)
        # a tile whose stim amplitude never cleared the threshold has
        # unreliable tile_connectivity on EVERY row (it's a ratio with a
        # tiny denominator), not just on the stim electrode's own row -> drop
        # the whole tile from the heatmap/histogram, not just its stim row
        tile_max_amp = data[data.stim].groupby(['config', 'tile'])['sine_voltage_uV'].max()
        bad_tiles = set(tile_max_amp[tile_max_amp < min_stim_uV].index)
        print(f"{below_thresh.sum()} stim rows / {len(bad_tiles)} tiles below "
              f"{min_stim_uV} uV")
    else:
        below_thresh = pd.Series(False, index=data.index)
        bad_tiles = set()
    data['stim_below_thresh'] = below_thresh
    data.loc[below_thresh, 'stim'] = False

    # configs = configs[-3:]
    # data = data[data.config.isin(configs)]  # use first config to get min/max for histogram edges
    # print(data[['config', 'tile_connectivity', 'stim', 'electrode']].head(50))
    # print(configs)

    is_bad_tile = data.set_index(['config', 'tile']).index.isin(bad_tiles)
    data_heat = data[~is_bad_tile]   # feeds the histogram/heatmap/dendrogram + n_shorted

    vals = data_heat['tile_connectivity']
    edges = np.linspace(vals.min(), vals.max(), bins + 1)
    print(f"Computing histograms for {len(configs)} configs...")
    M = np.vstack([np.histogram(vals[data_heat.config == c], edges)[0] for c in configs])

    print("Performing hierarchical clustering...")
    Z = linkage(M, method='ward')
    cut = (Z[-n_groups, 2] + Z[-n_groups + 1, 2]) / 2
    group_of = fcluster(Z, t=cut, criterion='distance')   # 1..K, aligned with `configs`

    h = max(6, len(configs) * 0.02)
    fig = plt.figure(figsize=(16, h + 9), dpi=dpi)
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 4, 0.9, 0.9, 0.18],
                          height_ratios=[h, 9], wspace=0.05, hspace=0.12)
    axd   = fig.add_subplot(gs[0, 0])
    ax    = fig.add_subplot(gs[0, 1])
    ax_n  = fig.add_subplot(gs[0, 2])
    ax_amp= fig.add_subplot(gs[0, 3])
    ax_cb = fig.add_subplot(gs[0, 4])
    ax_chip = fig.add_subplot(gs[1, :])

    print("Performing hierarchical clustering and plotting dendrogram...")
    dn = dendrogram(Z, orientation='left', color_threshold=cut, ax=axd, no_labels=True)
    axd.axvline(cut, color='0.5', lw=2, ls='--')
    axd.set_xticks([])
    for s in axd.spines.values():
        s.set_visible(False)

    order = dn['leaves']
    M = M[order]
    configs_ord = [configs[i] for i in order]

    # ---- main heatmap ----
    im = ax.pcolormesh(edges, np.arange(len(configs_ord) + 1), M, cmap='turbo',
                       norm=LogNorm(vmin=1, vmax=max(M.max(), 2)))
    ax.axvline(shorted_thresh, color='red', lw=2, ls=':', zorder=3)
    ax.set_xlabel('tile_connectivity (rel. to stim)')
    ax.set_yticks([])
    ax.set_ylim(0, len(configs_ord))

    rows = np.arange(len(configs_ord)) + 0.5

    # tile-wise: one point per tile within a config, instead of one number
    # pooled across all its tiles -> collapses to a single point when a
    # config only has one tile
    n_shorted_x, n_shorted_y = [], []
    stim_amp_x, stim_amp_y, stim_amp_c = [], [], []
    for row_y, c in zip(rows, configs_ord):
        for t, tile_sub in data[data.config == c].groupby('tile'):
            if (c, t) not in bad_tiles:
                # exclude both still-stim rows and neutered (below-thresh)
                # ones -- the latter are still the driven electrode's own
                # reading, not a genuine responder
                not_stim = ~(tile_sub.stim | tile_sub.stim_below_thresh)
                n_shorted_x.append((not_stim &
                                    (tile_sub.tile_connectivity >= shorted_thresh)).sum())
                n_shorted_y.append(row_y)
            # original stim rows = still-stim ones + those we just neutered
            stim_rows = tile_sub[tile_sub.stim | tile_sub.stim_below_thresh]
            stim_amp_x.extend(stim_rows.sine_voltage_uV.values)
            stim_amp_y.extend([row_y] * len(stim_rows))
            stim_amp_c.extend(np.where(stim_rows.stim_below_thresh, 'dimgray', 'gold'))

    ax_n.scatter(n_shorted_x, n_shorted_y, s=6, color='red')
    ax_n.set_xscale('symlog')
    ax_n.set_xlabel(f'# el ≥ {shorted_thresh}')
    ax_n.grid(axis='x', lw=0.4, alpha=0.5)
    ax_n.set_ylim(0, len(configs_ord)); ax_n.set_yticks([])
    ax_n.spines[['top', 'right']].set_visible(False)
    ax_n.set_facecolor('#f0f0f0')

    ax_amp.scatter(stim_amp_x, stim_amp_y, s=6, c=stim_amp_c, edgecolors='0.4', linewidths=0.3)
    ax_amp.set_xlabel('stim sine uV')
    ax_amp.grid(axis='x', lw=0.4, alpha=0.5)
    ax_amp.set_ylim(0, len(configs_ord)); ax_amp.set_yticks([])
    ax_amp.spines[['top', 'right']].set_visible(False)
    ax_amp.set_facecolor('#f0f0f0')
    # set min value to 0
    ax_amp.set_xlim(left=0)

    fig.colorbar(im, cax=ax_cb, label='count (log)')
    
    # ---- chip panel: stim electrodes as scatter, colored to match dendrogram ----
    leaf_color = {configs[leaf]: col
                  for leaf, col in zip(dn['leaves'], dn['leaves_color_list'])}

    def elxy(el):                       # electrode id -> grid (x, y); valid for any el
        return (el % 220, el // 220)

    xs, ys, cs = [], [], []
    for c in configs:
        stim_row = data[(data.config == c) & data.stim]
        if not len(stim_row):
            continue
        x, y = elxy(int(stim_row.electrode.iloc[0]))
        xs.append(x); ys.append(y); cs.append(leaf_color[c])
    ax_chip.scatter(xs, ys, c=cs, s=20, edgecolors='none', zorder=3)

    # ---- draw connections ----
    # try to read connectivity graph from file, otherwise build it
    g_fname = os.path.join(output_dir, "extracted_connectivity_graph.csv")
    # if False:
    if os.path.exists(g_fname) or rebuilt_graph:
        g = pd.read_csv(g_fname)
    else:
        g = build_connectivity_graph(data, thresh=shorted_thresh)
        g.to_csv(os.path.join(output_dir, "extracted_connectivity_graph.csv"), index=False)
        print(g)

    # 6 reciprocal styles (3 grays x 2 dashes) cycled by island id -> adjacent islands differ
    grays = ['0.2', '0.4', '0.55']
    dashes = [(0, (4, 2)), (0, (1, 2))]            # dashed, dotted
    recip_styles = [(g_, d) for d in dashes for g_ in grays]

    for s, t, recip, isl in zip(g.source, g.target, g.reciprocal, g.short_island_id):
        if recip:
            col, ls = recip_styles[int(isl) % len(recip_styles)]
            alpha, lw = 0.9, 1.3
        else:
            col, ls, alpha, lw = '0.45', 'solid', 0.7, 0.9   # one-way: solid gray
        ax_chip.add_patch(FancyArrowPatch(
            elxy(s), elxy(t), connectionstyle='arc3,rad=0.2',
            arrowstyle='-|>', mutation_scale=8,
            color=col, alpha=alpha, lw=lw, linestyle=ls, zorder=2))
        
    ax_chip.set_xlim(0, 220)
    ax_chip.set_ylim(120, 0)         # origin upper: electrode 0 at top-left
    ax_chip.set_aspect('equal')
    ax_chip.set_title('stim electrode group + connections '
                      '(dark=reciprocal, faint=one-way)', fontsize=9)
    ax_chip.set_xticks([]); ax_chip.set_yticks([])
    
    fig.suptitle(f'Connectivity per config (n={len(configs_ord)}, ~{n_groups} groups): '
                 f'{os.path.basename(subdir)}', fontsize=10)
    fig.savefig(os.path.join(output_dir, "amplitude_heatmap.png"),
                dpi=dpi, bbox_inches='tight')
    print("Saved amplitude heatmap to \n", os.path.join(output_dir, "amplitude_heatmap.png"))
    return fig

def connected_islands(subdir, output_dir=None, output_fname=None, conn_thr=.8, draw_mea_args={'fast': True}):
    processed_dir = os.path.join(subdir, "processed")
    if os.path.exists(os.path.join(output_dir, "extracted_sine_voltages.csv")):
        data = pd.read_csv(os.path.join(output_dir, "extracted_sine_voltages.csv"))
    else:
        data = get_aggr_results(output_dir)
    print(f"Total rows: {len(data)}")

    g_fname = os.path.join(processed_dir, "extracted_connectivity_graph.csv")
    if os.path.exists(g_fname) or conn_thr is None:
        g = pd.read_csv(g_fname)
    else:
        g = build_connectivity_graph(data, thresh=conn_thr)
        g.to_csv(os.path.join(output_dir, "extracted_connectivity_graph.csv"), index=False)

    if g.empty:
        print("No connectivity edges found.")
        return

    g['reciprocal'] = g['reciprocal'].astype(str).str.lower().isin(['true', '1'])
    reciprocal = g[g.reciprocal & g.short_island_id.notna()]
    if reciprocal.empty:
        print("No reciprocal connectivity edges found.")
        return

    # Electrode id -> chip coordinates for plotting.
    pos = (data[['electrode', 'x', 'y']]
           .drop_duplicates('electrode')
           .set_index('electrode'))

    islands = reciprocal.groupby('short_island_id')
    print(f"Found {len(islands)} shorted islands.")

    (fig, ax), recs = draw_mea1k(**draw_mea_args)
    cmap = plt.cm.get_cmap('tab20')

    for i, (_, edges) in enumerate(islands):
        color = cmap(i % 20)
        members = [el for el in pd.unique(edges[['source', 'target']].values.ravel())
                   if el in pos.index]
        if not members:
            continue

        for s, t in zip(edges.source, edges.target):
            if s not in pos.index or t not in pos.index:
                continue
            x0, y0 = pos.loc[s, 'x']+2.5 + 5/2, pos.loc[s, 'y'] +6 +9/2
            x1, y1 = pos.loc[t, 'x']+2.5 + 5/2, pos.loc[t, 'y'] +6 +9/2
            ax.plot([x0, x1], [y0, y1], color=color, alpha=0.75, linewidth=1, zorder=-5)
    
    if output_dir is not None and output_fname is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_fullfname = os.path.join(output_dir, output_fname)
        print("Saving island plot to ", out_fullfname)
        fig.savefig(out_fullfname, dpi=300)
    plt.show()
    
    
def main(): 
    L = Logger()
    L.init_logger(None, None, "INFO")
    L.logger.debug("Starting shortcut analysis")
    nas_dir = device_paths()[0]
    # fix seed
    np.random.seed(42)
    
    subdirs = [
        f'devices/headstage_devices/MEA1K22/recordings/2026-07-10_14.10.08_ACF_TileShorts2_Try9', # all
        
        f'devices/well_devices/4988/recordings/2026-08-18_14.49.56_SC_Pt_MEA_els_PT0', # all
        f'devices/well_devices/4988/recordings/2026-08-18_16.27.16_SC_Pt_MEA_els_PT0', # all
        # f'devices/well_devices/4988/recordings/2026-08-19_11.38.31_SC_Pt_MEA_els_PT0', # all
        # f'devices/headstage_devices/MEA1K22/recordings/2026-08-21_12.54.30_Unkown_SC_TileShorts', # all
        f'devices/well_devices/4988/recordings/2026-08-24_08.36.39_SC_Pt_MEA_els_PT0', # all
    ]
    
    target_subdir = subdirs[-1]
    
    output_dir = os.path.join(nas_dir, target_subdir, 'processed')
    connected_islands(os.path.join(nas_dir, target_subdir), 
                      output_dir=output_dir,  conn_thr=.5,
                      output_fname=f"connected_islands.png")
    
if __name__ == "__main__":
    main()