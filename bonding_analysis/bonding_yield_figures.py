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
import matplotlib.dates as mdates

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join('..',))
sys.path.append(os.path.join('..', '..'))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

import mea1k_modules.mea1k_visualizations as vis

L = Logger()
L.init_logger(None, None, "WARNING")

# nas_dir = device_paths()[0]

# curated copies of the processed impedance sweeps, see the README in there
YIELD_DIR = os.path.join('/Users/loaloa/homedataAir/phd/ratvr', "mea1k_paper", "bonding_yield", "100k_bonding_yield")
FIG_DIR = os.path.join('/Users/loaloa/homedataAir/phd/ratvr', "mea1k_paper", "bonding_yield", "figures")

# nominal load soldered to the pads in all these tests
R_NOMINAL = 100_000
# a pad counts as bonded when at least one of its electrodes reads the nominal load
# within this tolerance; everything else (unmeasured, open, shorted) counts as not bonded
R_TOL = 30_000

# el_id groups to drop from a test's pad mapping before the impedance join. test6's el_id 3
# is the electrode column that was deliberately left unsoldered, so its pads are not part of
# that sample's bonding yield at all (rather than counting as failed bonds).
EXCLUDE_EL_ID = {"test6": (0,)}


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


def load_pad_map(test):
    """mea1k electrode -> pad it sits under, plus the pad's aligned chip coordinates.
    One mapping per test (the chip/flex alignment is fixed for all its timepoints)."""
    pad2el = pd.read_csv(os.path.join(YIELD_DIR, test, "pad2el_mapping.csv"))
    drop = EXCLUDE_EL_ID.get(test)
    if drop is not None:
        pad2el = pad2el[~pad2el['el_id'].isin(drop)]
    return pad2el.set_index('mea1k_el')[['pad_id', 'pad_x_aligned', 'pad_y_aligned']]


def pad_bond_state(tp_dir, df):
    """Per-pad bond state of one timepoint: one row per pad with its aligned coordinates
    and `bonded`, True when ANY electrode under that pad reads R_NOMINAL +/- R_TOL.
    Electrodes that were never measured drop out as NaN and count as not bonded."""
    pad2el = load_pad_map(parse_name(tp_dir)['test'])
    imp = df.set_index('electrode')['impedance_Ohm'].reindex(pad2el.index)
    el = pad2el.assign(ok=imp.between(R_NOMINAL - R_TOL, R_NOMINAL + R_TOL).values)
    return el.groupby('pad_id').agg(bonded=('ok', 'any'),
                                    x=('pad_x_aligned', 'first'),
                                    y=('pad_y_aligned', 'first'))


# the curated dir names only carry the t00..tNN order; index.csv keeps the original
# recording name, which is where the wall-clock timestamp survived
INDEX = pd.read_csv(os.path.join(YIELD_DIR, "index.csv")).set_index('new_name')


def tp_time(tp_dir):
    """Wall-clock time a timepoint was recorded, parsed out of its original recording name
    (`2026-09-08_14.35_test_100k_...`) via index.csv."""
    orig = INDEX.loc[os.path.basename(tp_dir), 'orig_recording']
    return pd.to_datetime(orig[:16], format="%Y-%m-%d_%H.%M")


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
PAD_BONDED_C, PAD_OPEN_C = '#8fd18f', '#d9d9d9'   # pad map: bonded / not bonded
PAD_S = 2.2                       # pad square size, small enough to keep the grid gaps open
CHIP_W_UM, CHIP_H_UM = 3850, 2100   # full chip window, shared by every heatmap
DEBUG_AXES = False         # TEMPORARY: frame + label both heatmap types to check the window
WIN_MARGIN = 40           # µm of air around pads that stick out past the chip outline
Y_TICK_STEP = 1000        # histogram y ticks, one per 1000 MEA electrodes
SPINE_LW, SPINE_COLOR = .6, 'gray'
CMAP = 'plasma'

# figure geometry, in inches — set the row height and everything else follows from the
# chip aspect, so the heatmaps fill their column exactly and no space is wasted
ROW_H, HIST_W, CBAR_H = 1.2, .8, .11


def row_window(pad):
    """Plot window for one sample row: the chip, grown where that flex's pads stick out past
    it. The heatmap and the pad map of a row share it, so the two panels stay registered."""
    x0 = min(0, pad.x.min() - WIN_MARGIN)
    x1 = max(CHIP_W_UM, pad.x.max() + WIN_MARGIN)
    y0 = min(0, pad.y.min() - WIN_MARGIN)
    y1 = max(CHIP_H_UM, pad.y.max() + WIN_MARGIN)
    return x0, x1, y0, y1


def set_chip_window(ax, win):
    """Apply a row window and draw the chip outline as a thin line inside it (a rectangle,
    not axes spines: the window can be wider than the chip)."""
    x0, x1, y0, y1 = win
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect('equal', adjustable='box')
    ax.add_patch(plt.Rectangle((0, 0), CHIP_W_UM, CHIP_H_UM, fill=False, zorder=3,
                               linewidth=SPINE_LW, edgecolor=SPINE_COLOR))


def frame_chip_ax(ax):
    """TEMPORARY debug framing for the two heatmap types: perimeter spines plus chip
    coordinate ticks on both axes, so the 3850 x 2100 window can be read off the panel.
    Set DEBUG_AXES = False to go back to the bare panels."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LW)
        spine.set_color(SPINE_COLOR)
    ax.set_xticks([0, CHIP_W_UM // 2, CHIP_W_UM])
    ax.set_yticks([0, CHIP_H_UM // 2, CHIP_H_UM])
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True,
                   labelsize=5.5, length=2, pad=1)


tp_dirs = last_timepoints()
dfs = [load_tp(d) for d in tp_dirs]
n = len(tp_dirs)

# per-pad bond state, same last timepoint as the heatmaps. The aligned pad coordinates live
# in the chip frame, so the pad maps share the chip window and stay comparable row to row.
pads = [pad_bond_state(d, df) for d, df in zip(tp_dirs, dfs)]
wins = [row_window(p) for p in pads]

# widest row sets the column width; every row is height-limited to ROW_H and all of them
# share the same y span, so the µm-per-inch scale is identical across rows
chip_w = ROW_H * max((x1 - x0) / (y1 - y0) for x0, x1, y0, y1 in wins)
pad_w = chip_w

fig = plt.figure(figsize=(chip_w + HIST_W + pad_w + .8, ROW_H*n + .85), dpi=200)
gs = fig.add_gridspec(n + 1, 3, width_ratios=[chip_w, HIST_W, pad_w],
                      height_ratios=[1]*n + [CBAR_H/ROW_H], wspace=.14, hspace=.34)

# mean bonding yield over the samples (each sample weighted equally, not pad-weighted)
mean_yield = np.mean([p.bonded.mean() for p in pads])
fig.suptitle(f"Mean bonding yield {mean_yield:.0%}", fontsize=9, y=.995)

for i, (tp_dir, df, pad, win) in enumerate(zip(tp_dirs, dfs, pads, wins)):
    meta = parse_name(tp_dir)
    imp = df['impedance_Ohm'].clip(VMIN, VMAX)

    # --- chip heatmap
    ax = fig.add_subplot(gs[i, 0])
    sc = vis.scatter_mea1k(ax, df['electrode'], imp, vmin=VMIN, vmax=VMAX,
                           cmap=CMAP, s=1.1, cbar=False)
    set_chip_window(ax, win)
    if DEBUG_AXES:
        frame_chip_ax(ax)
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
    # ax.set_ylabel("n MEA electr.", fontsize=9, labelpad=2)
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

    # --- pad map: one square per bond pad of the flex, green when bonded
    ax = fig.add_subplot(gs[i, 2])
    ax.scatter(pad.x, pad.y, marker='s', s=PAD_S, edgecolors='none',
               c=np.where(pad.bonded, PAD_BONDED_C, PAD_OPEN_C))
    set_chip_window(ax, win)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if DEBUG_AXES:
        frame_chip_ax(ax)
    n_bonded = int(pad.bonded.sum())
    ax.set_title(f"{n_bonded}/{len(pad)} ({n_bonded/len(pad):.0%}) pads bonded",
                 fontsize=8, weight='normal', loc='left', pad=2)

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

# --- pad map legend, in the colorbar row under the pad column
lax = fig.add_subplot(gs[n, 2])
lax.set_axis_off()
handles = [plt.Line2D([], [], marker='s', linestyle='none', markersize=6,
                      markerfacecolor=PAD_BONDED_C, markeredgecolor='none',
                      label=f"bonded ({R_NOMINAL//1000}k $\\pm$ {R_TOL//1000}k)"),
           plt.Line2D([], [], marker='s', linestyle='none', markersize=6,
                      markerfacecolor=PAD_OPEN_C, markeredgecolor='none',
                      label="not bonded")]
lax.legend(handles=handles, loc='center left', bbox_to_anchor=(0, 1.3), ncol=2,
           frameon=False, fontsize=7.5, handletextpad=.4, columnspacing=1.6,
           borderpad=0, borderaxespad=0)

save_fig(fig, "fig1_final_state_per_sample")
plt.show()

# %%
# %% [markdown]
# ## Fig 2 — bonding yield over the timepoints
# Same per-pad bond criterion as Fig 1 (a pad is bonded when any electrode under it reads
# R_NOMINAL +/- R_TOL), evaluated at every timepoint of every test. One marker per sample,
# dashed connecting lines. Samples with a single sweep show up as a lone marker.

# %%
MARKERS = ['o', 's', '^', 'D', 'v']
RETIGHTEN_TP = 4          # t04 is the PT5 retightening sweep in both multi-sweep tests
# timepoints to leave out of the timeline, by their t<NN> index within the test
EXCLUDE_TP = {"test1": (3,)}

fig, ax = plt.subplots(figsize=(4.6, 2.4), dpi=200)
max_day = 0

for (test, label), mk in zip(SAMPLES.items(), MARKERS):
    tps = list_timepoints(test)
    keep = [k for k in range(len(tps)) if k not in EXCLUDE_TP.get(test, ())]
    # every sample starts at its own t00, so the samples are comparable in elapsed time
    # rather than in calendar date; day 1 is the day the bond was made
    t0 = tp_time(tps[0])
    days = [(tp_time(tps[k]) - t0).total_seconds() / 86400 for k in keep]
    yields = [pad_bond_state(tps[k], load_tp(tps[k])).bonded.mean() * 100 for k in keep]
    max_day = max(max_day, days[-1])
    line, = ax.plot(days, yields, linestyle='--', linewidth=.9, marker=mk,
                    markersize=4.5, label=label)
    # the press was retightened before t04; mark it on the samples that got that far
    if RETIGHTEN_TP in keep:
        j = keep.index(RETIGHTEN_TP)
        ax.annotate("* retighten\npress", (days[j], yields[j]),
                    textcoords='offset points', xytext=(3, 7), fontsize=6.5,
                    color=line.get_color(), ha='left', va='bottom', linespacing=.95)

ax.set_ylabel("Bonding yield (%)", fontsize=9)
ax.set_ylim(0, 100)
ax.set_xlim(-.15, max_day + .15)
day_ticks = np.arange(0, int(np.ceil(max_day)) + 1)
ax.set_xticks(day_ticks)
ax.set_xticklabels([f"Day {d+1}" for d in day_ticks], fontsize=8)
ax.tick_params(labelsize=8, length=2.5, pad=1)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_linewidth(SPINE_LW)
    ax.spines[side].set_color(SPINE_COLOR)
ax.grid(axis='y', linestyle='--', linewidth=SPINE_LW, color=SPINE_COLOR, alpha=.45)
ax.set_axisbelow(True)
ax.legend(fontsize=7, frameon=False, loc='lower left', bbox_to_anchor=(1.01, 0))

save_fig(fig, "fig2_yield_over_timepoints")
plt.show()

# %%
# %% [markdown]
# ## Preliminary figure caption
#
# **Figure X — Solder-bonded polyimide interconnect yields ~83 % of 768 pads and holds over
# days.** A single polyimide device, routing 768 electrode pads out to 4 solder pads, was
# bonded to a MEA1K CMOS array (**B**, bottom left). The chip accommodates two such devices;
# only one was mounted per chip here, for throughput. Each of the 4 solder pads was soldered
# to a 100 kOhm reference resistor routed to ground, so a correctly bonded pad presents the
# nominal 100 kOhm load. The chip's internal stimulation units delivered a 1 kHz sinusoidal
# current of 150 nA on each MEA electrode; the resulting voltage was read back on the
# amplifier and converted to an impedance magnitude. The phase offset was ~0 for all
# measurements (not shown).
#
# **(A, top)** Final state of each of the five samples (rows). Left: impedance of every
# measured MEA electrode at its position on the 3850 x 2100 µm chip (thin outline = chip
# boundary). Middle: distribution of the same values, on the linear impedance scale shared
# with the heatmap colours and the colourbar below (both ends clipped, so the last bin holds
# every bad bond regardless of magnitude). Right: per-pad bond state, one square per device
# pad at its position after alignment to the chip frame — green when at least one electrode
# under that pad reads 100 kOhm +/- 30 kOhm, grey otherwise. Yields ranged from 75 % to 90 %
# (mean 83 %, n = 5). Pad pitch was 35 µm (samples 1, 2, 5) or 61 µm (samples 3, 4). On
# sample 4 one quarter of the device was deliberately left unsoldered and is excluded from
# both the map and the denominator (576 of 768 pads).
#
# **(C, bottom right)** Bonding yield over repeated impedance sweeps of the same samples,
# each sample on its own elapsed-time axis from the day its bond was made. Marker shape
# encodes sample; dashed lines connect sweeps of one sample. Samples 2, 4 and 5 were swept
# once. Yield was stable across ~3 days; the step in sample 1 follows a retightening of the
# press (*), after which the yield held flat over the remaining sweeps.
#
# *Preliminary — panel letters are not yet drawn on the figure, and the assembly panel (B)
# is not produced by this script.*

# %% [markdown]
# ## test11 (MEA1K27) — this attempt only
# Single-row version of Fig 1 for the one recording, read straight from its `processed` dir
# (not part of the curated yield folder, so it is not in `SAMPLES` / `INDEX`).

# %%
T11_DIR = ("/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/devices/headstage_devices/"
           "MEA1K27/recordings/2026-10-28_14.57_test11_100k_36umP_PT7_Imp/processed")

df11 = pd.read_csv(os.path.join(T11_DIR, "all_impedance.csv"))
map11 = pd.read_csv(os.path.join(T11_DIR, "pad2el_mapping.csv"))
map11 = map11.dropna(subset=['mea1k_el']).astype({'mea1k_el': int, 'pad_id': int})
map11 = map11.set_index('mea1k_el')[['pad_id', 'pad_x_aligned', 'pad_y_aligned']]
# same bond criterion as the curated tests
imp11 = df11.set_index('electrode')['impedance_Ohm'].reindex(map11.index)
pad11 = (map11.assign(ok=imp11.between(R_NOMINAL - R_TOL, R_NOMINAL + R_TOL).values)
         .groupby('pad_id').agg(bonded=('ok', 'any'), x=('pad_x_aligned', 'first'),
                                y=('pad_y_aligned', 'first')))
win11 = row_window(pad11)
imp = df11['impedance_Ohm'].clip(VMIN, VMAX)

cw = ROW_H * 2 * (win11[1] - win11[0]) / (win11[3] - win11[2])
fig = plt.figure(figsize=(cw + HIST_W + cw + .8, ROW_H*2 + .85), dpi=200)
gs = fig.add_gridspec(2, 3, width_ratios=[cw, HIST_W, cw], height_ratios=[1, .1],
                      wspace=.14, hspace=.5)

ax = fig.add_subplot(gs[0, 0])
sc = vis.scatter_mea1k(ax, df11['electrode'], imp, vmin=VMIN, vmax=VMAX, cmap=CMAP, s=1.1,
                       cbar=False)
set_chip_window(ax, win11)
ax.set_title("test11, MEA1K27, 35 µm Pads", fontsize=8, weight='normal', loc='left', pad=2)

hax = fig.add_subplot(gs[0, 1])
hax.hist(imp, bins=N_BINS, range=(VMIN, VMAX), color='k', edgecolor='none')
hax.set_xlim(VMIN, VMAX)
hax.set_xticks(TICKS)
hax.set_xticklabels(TICKLABELS, fontsize=8)
hax.yaxis.tick_right()
# hax.set_ylabel("n MEA electr.", fontsize=9, labelpad=2)
hax.yaxis.set_label_position("right")
hax.tick_params(labelsize=8, length=2.5, pad=1)
for side in ('top', 'left'):
    hax.spines[side].set_visible(False)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(pad11.x, pad11.y, marker='s', s=PAD_S*3, edgecolors='none',
           c=np.where(pad11.bonded, PAD_BONDED_C, PAD_OPEN_C))
set_chip_window(ax, win11)
ax.axis('off')
nb11 = int(pad11.bonded.sum())
ax.set_title(f"{nb11}/{len(pad11)} ({nb11/len(pad11):.0%}) pads bonded",
             fontsize=8, weight='normal', loc='left', pad=2)

cax = fig.add_subplot(gs[1, 1])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
cb.set_ticks(TICKS)
cb.set_ticklabels(TICKLABELS)
cb.ax.tick_params(labelsize=8, length=2.5, pad=1)
cb.set_label("Impedance (Ohm)", fontsize=9, labelpad=3)
plt.show()

# %%
