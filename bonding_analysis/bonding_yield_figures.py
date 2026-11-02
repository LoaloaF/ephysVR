# %%
%load_ext autoreload
%autoreload 2

# %%
import os
import sys
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join('..',))
sys.path.append(os.path.join('..', '..'))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

import mea1k_modules.mea1k_visualizations as vis

L = Logger()
L.init_logger(None, None, "WARNING")

nas_dir = device_paths()[0]

# curated copies of the processed impedance sweeps, see the README in there
YIELD_DIR = os.path.join(nas_dir, "mea1k_paper", "bonding_yield", "100k_bonding_yield")
FIG_DIR = os.path.join(nas_dir, "mea1k_paper", "bonding_yield", "figures")

# nominal load soldered to the pads in all these tests
R_NOMINAL = 100_000


# %%
# ---- helpers ----

def list_timepoints(test):
    """All timepoint dirs of one test (e.g. "test4"), in chronological t00..tNN order."""
    return sorted(glob(os.path.join(YIELD_DIR, test, f"{test}_*_imp")))


def load_tp(tp_dir):
    """Impedance table of one timepoint dir. These dirs ARE the `processed` dirs of the
    original recordings, so the csv sits at the top level (no `processed` subdir)."""
    df = pd.read_csv(os.path.join(tp_dir, "all_impedance.csv"))
    df.attrs['name'] = os.path.basename(tp_dir)
    return df


def parse_name(tp_dir):
    """test1_MEA24_35umP_t09_PT5post64C_imp -> dict(test, chip, pads, t, pt)"""
    parts = os.path.basename(tp_dir).split('_')
    return {"test": parts[0], "chip": parts[1], "pads": parts[2],
            "t": parts[3], "pt": parts[4]}


def save_fig(fig, name, dpi=300):
    """Write the current figure to FIG_DIR as png (paper output lives next to the data)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    print("saved", out)
    return out


def last_timepoints():
    """The final timepoint of every test, in test order."""
    tests = sorted(d for d in os.listdir(YIELD_DIR)
                   if os.path.isdir(os.path.join(YIELD_DIR, d)))
    return [list_timepoints(t)[-1] for t in tests]


# %% [markdown]
# # Bonding yield figures
# Figures built from `mea1k_paper/bonding_yield/100k_bonding_yield` — the 100k reference-load
# bonding tests. A perfectly bonded pad reads ~100 kOhm, so impedance is read on a linear
# scale here: the interesting question is how tight the distribution sits on 100 kOhm, not
# how many decades the bad ones span.

# %% [markdown]
# %% [markdown]
# ## Fig 1 — final state of each sample
# Last timepoint of every test, one row per sample: CMOS spatial distribution of impedance
# next to its histogram. Linear impedance scale, shared by the heatmap and the histograms,
# so the single colorbar at the bottom doubles as the histogram x-axis.

# %%
# tests in this folder -> sample numbering used in the figure
SAMPLES = {
    "test1": "Sample 1",
    "test2": "Sample 2",
    "test4": "Sample 3",
    "test6": "Sample 4 (1/4 not soldered)",
    "test8": "Sample 5",
}
PAD_LABEL = {"35umP": "35 µm Pads", "36umP": "35 µm Pads", "61umP": "61 µm Pads"}

# Linear scale shared by the heatmap colors and the histogram x-axis. A perfectly bonded pad
# reads the 100k nominal load; the ends are clipped, so the top color / last bin holds every
# bad bond regardless of how bad. Nothing in this dataset reads below 64k.
VMIN, VMAX = 10_000, 1900_000
TICKS = [100_000, 1500_000]
TICKLABELS = ["100k", ">1500k"]
N_BINS = 60
Y_TICK_STEP = 1000        # histogram y ticks, one per 1000 MEA electrodes
SPINE_LW, SPINE_COLOR = .6, 'gray'
CMAP = 'plasma'

# figure geometry, in inches — set the row height and everything else follows from the
# chip aspect, so the heatmaps fill their column exactly and no space is wasted
ROW_H, HIST_W, CBAR_H = 1.2, 1.0, .11

tp_dirs = last_timepoints()
dfs = [load_tp(d) for d in tp_dirs]
n = len(tp_dirs)

# crop the chip to the band that carries electrodes (shared across samples so the heatmaps
# stay comparable); quantiles rather than min/max, a few strays sit far outside the band
all_y = np.concatenate([(d['electrode'] // 220) * 17.5 for d in dfs])
Y0, Y1 = np.quantile(all_y, .001) - 40, np.quantile(all_y, .999) + 40

chip_w = ROW_H * 3850 / (Y1 - Y0)
fig = plt.figure(figsize=(chip_w + HIST_W + .55, ROW_H*n + .85), dpi=200)
gs = fig.add_gridspec(n + 1, 2, width_ratios=[chip_w, HIST_W],
                      height_ratios=[1]*n + [CBAR_H/ROW_H], wspace=.10, hspace=.34)

for i, (tp_dir, df) in enumerate(zip(tp_dirs, dfs)):
    meta = parse_name(tp_dir)
    imp = df['impedance_Ohm'].clip(VMIN, VMAX)

    # --- chip heatmap
    ax = fig.add_subplot(gs[i, 0])
    sc = vis.scatter_mea1k(ax, df['electrode'], imp, vmin=VMIN, vmax=VMAX,
                           cmap=CMAP, s=1.1, cbar=False)
    ax.set_ylim(Y1, Y0)
    ax.set_title(f"{SAMPLES[meta['test']]}, {PAD_LABEL[meta['pads']]}",
                 fontsize=8, weight='normal', loc='left', pad=2)

    # --- histogram, linear in impedance, same scale as the heatmap
    ax = fig.add_subplot(gs[i, 1])
    ax.hist(imp, bins=N_BINS, range=(VMIN, VMAX), color='k', edgecolor='none')
    ax.set_xlim(VMIN, VMAX)
    ax.set_xticks(TICKS)
    ax.set_xticklabels([] if i < n - 1 else TICKLABELS, fontsize=8)
    # a tick every 1000 electrodes, but only the top one carries a number (rotated, so it
    # sits along the axis and costs no width). Always show at least one full 1000 step, so
    # the samples with few electrodes are not blown up to fill the panel.
    ytop = max(Y_TICK_STEP, ax.get_ylim()[1])
    ax.set_ylim(0, ytop)
    yticks = np.arange(0, ytop + 1, Y_TICK_STEP)
    ax.set_yticks(yticks)
    ylabels = [''] * len(yticks)
    ylabels[-1] = f"{int(yticks[-1])}"
    ax.set_yticklabels(ylabels, rotation=90, va='center', fontsize=8)
    ax.yaxis.tick_right()          # y axis on the right, away from the heatmap
    ax.tick_params(axis='both', labelsize=8, length=2.5, pad=1)
    ax.set_ylabel("n MEA electr.", fontsize=9, labelpad=2)
    ax.yaxis.set_label_position("right")   # follow the ticks to the right side
    for side in ('top', 'left'):
        ax.spines[side].set_visible(False)
    for side in ('right', 'bottom'):
        ax.spines[side].set_linewidth(SPINE_LW)
        ax.spines[side].set_color(SPINE_COLOR)
    # faint dashed guides at the 1000-electrode ticks, behind the bars
    ax.grid(axis='y', linestyle='--', linewidth=SPINE_LW, color=SPINE_COLOR, alpha=.45)
    ax.set_axisbelow(True)
    hist_ax = ax

# --- single colorbar under the histogram column, sharing its impedance axis
cax = fig.add_subplot(gs[n, 1])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
cb.set_ticks(TICKS)
cb.set_ticklabels(TICKLABELS)
cb.ax.tick_params(labelsize=8, length=2.5, pad=1)
cb.set_label("Impedance (Ohm)", fontsize=9, labelpad=3)
# align the colorbar exactly with the histogram column above it
fig.canvas.draw()
hb, cbb = hist_ax.get_position(), cax.get_position()
cax.set_position([hb.x0, cbb.y0, hb.width, cbb.height])

save_fig(fig, "fig1_final_state_per_sample")
plt.show()

# %%
