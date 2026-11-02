"""Shared figure style for the publication figures: one place for font sizes, line widths,
column widths and the panel-letter helper, so all figures look consistent.

    from fig_style import setup_style, FULL_WIDTH_IN, panel_label
    setup_style()
"""
import matplotlib.pyplot as plt

FULL_WIDTH_IN = 7.2      # full page width of a two-column journal figure
COLUMN_WIDTH_IN = 3.5    # single column
BASE_FONT = 8.5          # pt, text; axis labels are +0.5, ticks / legends -0.5 / -1.5

# colour-blind-safe pair (Okabe-Ito blue / vermillion), used for negative / positive current
BLUE, VERMILLION, GREY = "#0072B2", "#D55E00", "0.3"


def setup_style(base=BASE_FONT):
    """Apply the project-wide matplotlib rcParams (editable text in PDF/SVG, sans-serif, light axes)."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base, "axes.labelsize": base + 0.5, "axes.titlesize": base,
        "xtick.labelsize": base - 0.5, "ytick.labelsize": base - 0.5, "legend.fontsize": base - 1.5,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
    })


def style_axes(ax, spines=("left", "bottom"), grid="both"):
    """Light grid instead of spines; only the listed spines are kept."""
    for name, sp in ax.spines.items():
        sp.set_visible(name in spines)
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid, color="0.9", lw=0.6)
    ax.tick_params(length=2.5 if spines else 0)


def panel_label(fig, x, y, letter, fontsize=11, **kw):
    """Bold panel letter at figure coordinates."""
    return fig.text(x, y, letter, fontweight="bold", fontsize=fontsize, va=kw.pop("va", "bottom"), **kw)
