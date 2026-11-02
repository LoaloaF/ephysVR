"""Render the 16 magnetic-ID barcode patterns as a decode reference card.

Each panel is one ID (0..15): filled horizontal bands at their normalized shank positions
(0 = the shank tip, where the DEEPEST electrode sits; 1 = just above the SHALLOWEST electrode,
i.e. the top of the 64-electrode array, not the top of the shank). Two always-on thick anchors (the
fattest = bottom, for orientation) bracket the present data bands, which encode the ID. Stripe
heights are auto-sized to the MRI well floor, so they depend on the electrode geometry -- the
legend therefore builds the real patterns from a default bundle.
"""
from typing import Optional

import matplotlib.pyplot as plt

#Allow running this file BY PATH (`python3 electrode_bundle/id_legend.py`) as well as by module
#(`python3 -m electrode_bundle.id_legend`). A path run has no parent package, so the relative
#imports below would fail with "attempted relative import with no known parent package"; putting
#the package's parent directory on sys.path and naming the package fixes that. No effect on
#normal imports, where __package__ is already set.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "electrode_bundle"

from .bundle import build_bundle
from .config import BundleConfig, IonpConfig
from .ionp import build_id_pattern


def render_id_legend(out_png: Optional[str] = None):
    el_locs = build_bundle(BundleConfig()).electrode_locs
    cfg = IonpConfig()
    fig, axes = plt.subplots(1, 16, figsize=(16, 6), sharey=True)
    for i, ax in enumerate(axes):
        for start, end in build_id_pattern(i, el_locs, cfg):
            ax.axhspan(start, end, color="black")
        ax.set_title(str(i))
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
    axes[0].set_ylabel("norm shank\n0 = tip (deepest electrode)  ->  1 = just above shallowest electrode")
    fig.suptitle("IONP magnetic-ID barcodes 0-15")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    return fig


if __name__ == "__main__":
    #Run as `python3 -m electrode_bundle.id_legend` (NOT `python3 electrode_bundle/id_legend.py`
    #-- a direct path run has no parent package, so the relative imports above fail).
    #The `legend` subcommand of main.py does the same thing.
    from .config import ensure_out_dir
    _out = f"{ensure_out_dir()}/ionp_id_legend.png"
    render_id_legend(_out)
    print(f"Saved legend -> {_out}")
