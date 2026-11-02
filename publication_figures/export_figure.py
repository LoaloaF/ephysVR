"""Export a matplotlib figure for a publication: PNG, SVG or PDF, optionally placed on an A4 page.

    from export_figure import export_figure
    export_figure(fig, "out/fig_DAC_calibr.svg")                 # format from the extension, A4 canvas
    export_figure(fig, "out/fig_DAC_calibr", fmt="png", canvas="tight")

Command line (exports a pickled figure, `pickle.dump(fig, open("f.pkl", "wb"))`):
    python export_figure.py f.pkl out/fig_DAC_calibr --fmt svg --canvas a4
"""
import argparse
import os
import pickle

A4_IN = (8.27, 11.69)


def _place_on_page(fig, page_in, margin_in, anchor):
    """Resize `fig` to the page and move everything in it (axes, figure-level texts, in figure
    coordinates) so the figure keeps its physical size, sitting at the top of the page. Returns an undo."""
    fw, fh = fig.get_size_inches()
    pw, ph = page_in
    ox = (pw - fw) / 2 if anchor == "center" else margin_in            # left offset (in)
    oy = ph - margin_in - fh                                           # bottom offset (in)
    sx, sy = fw / pw, fh / ph
    dx, dy = ox / pw, oy / ph
    undo = []
    for ax in fig.axes:
        old = ax.get_position().frozen()
        ax.set_position([old.x0 * sx + dx, old.y0 * sy + dy, old.width * sx, old.height * sy])
        undo.append(lambda ax=ax, old=old: ax.set_position(old))
    for t in fig.texts:
        x, y = t.get_position()
        t.set_position((x * sx + dx, y * sy + dy))
        undo.append(lambda t=t, x=x, y=y: t.set_position((x, y)))
    fig.set_size_inches(pw, ph, forward=False)
    undo.append(lambda: fig.set_size_inches(fw, fh, forward=False))
    return lambda: [u() for u in reversed(undo)]


def export_figure(fig, path, fmt=None, canvas="a4", dpi=300, margin_in=0.6, anchor="center", transparent=False):
    """Save `fig` to `path`.
    fmt: "png" | "svg" | "pdf"; default from the extension of `path` (png if none).
    canvas: "a4" = figure at its physical size on an A4 portrait page (top, centred; anchor="left" for
            the margin), "tight" = cropped to the figure.
    Returns the written file name."""
    root, ext = os.path.splitext(path)
    fmt = (fmt or ext.lstrip(".") or "png").lower()
    if fmt not in ("png", "svg", "pdf"):
        raise ValueError(f"fmt must be png, svg or pdf, got {fmt!r}")
    out = f"{root}.{fmt}"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if canvas == "a4":
        undo = _place_on_page(fig, A4_IN, margin_in, anchor)
        try:
            fig.savefig(out, format=fmt, dpi=dpi, transparent=transparent)
        finally:
            undo()
    elif canvas == "tight":
        fig.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight", transparent=transparent)
    else:
        raise ValueError("canvas must be 'a4' or 'tight'")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pickled_figure"); ap.add_argument("path")
    ap.add_argument("--fmt", choices=["png", "svg", "pdf"]); ap.add_argument("--canvas", default="a4", choices=["a4", "tight"])
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    print(export_figure(pickle.load(open(a.pickled_figure, "rb")), a.path, a.fmt, a.canvas, a.dpi))
