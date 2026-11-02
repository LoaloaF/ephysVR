"""Lay generated bundle DXFs next to each other in ONE DXF, for eyeballing them together.

    python3 -m electrode_bundle.side_by_side                    # every design in DESIGNS
    python3 -m electrode_bundle.side_by_side C1 C2 C3 C4 H1 H2  # just these (substring match)

Deliberately NOT WaferNesting: this only translates, never rotates, and keeps the order you
asked for (DESIGNS order, left to right, each labelled). It is a comparison drawing, not a
wafer layout -- no wafer outline, no clearance checks, no alignment marks.

Designs are lined up on a shared datum in y (`--align loop`, the default, or `top`), since
each one's absolute y depends on its own pitch and hook drop and they would otherwise sit
visibly staggered. Labels go on their own `Labels` layer
(WaferNesting's LAYER_MAP has no entry for it, so it would be dropped on a fab export
rather than silently becoming a mask feature).
"""
import argparse

import ezdxf
from ezdxf import bbox
from ezdxf.addons import importer
from ezdxf.enums import TextEntityAlignment

from .config import bundle_dxf_path, ensure_out_dir
from .design_sets import DESIGNS

GAP_UM = 1000.0          # clear space between neighbouring designs
LABEL_HEIGHT_UM = 700.0
LABEL_LAYER = "Labels"


def select_specs(filters):
    """DESIGNS entries whose name contains any of `filters` (all of them if empty)."""
    if not filters:
        return list(DESIGNS)
    picked = [s for s in DESIGNS if any(f in s.name for f in filters)]
    missing = [f for f in filters if not any(f in s.name for s in DESIGNS)]
    if missing:
        raise SystemExit(f"no design name matches {missing}; have: "
                         + ", ".join(s.name for s in DESIGNS))
    return picked


def datum_y(msp, align: str) -> float:
    """The y this design should be lined up by.

    Designs do NOT share an absolute y: the loop sits at -(delta_y + hook_drop + const), so
    a design's whole geometry is rigidly offset by its own pitch and hook drop (~200 um of
    spread across the current set). Laying them out without correcting for that leaves them
    visibly staggered, so pick a physical datum and put every design's on the same line.
    """
    if align == "loop":
        #the insertion loop: the LOWER of the two etch holes (the other is the rescue hook)
        etches = sorted(msp.query('*[layer=="Etching"]'),
                        key=lambda e: bbox.extents([e]).extmin.y)
        if not etches:
            raise SystemExit("align='loop' needs the Etching layer; use --align top")
        b = bbox.extents([etches[0]])
        return (b.extmin.y + b.extmax.y) / 2
    if align == "top":
        return bbox.extents(msp).extmax.y     # the connector end of the polyimide
    raise SystemExit(f"unknown --align {align!r}; use 'loop' or 'top'")


def build_side_by_side(specs, gap_um=GAP_UM, align="loop"):
    target = ezdxf.new(setup=True)
    target.layers.add(LABEL_LAYER)
    tmsp = target.modelspace()

    x_cursor = 0.0
    for spec in specs:
        src = ezdxf.readfile(bundle_dxf_path(spec.name))
        smsp = src.modelspace()
        ext = bbox.extents(smsp)

        #left edge to the cursor, and the chosen datum onto y = 0
        dx = x_cursor - ext.extmin.x
        dy = -datum_y(smsp, align)
        for entity in smsp:
            entity.translate(dx, dy, 0)
        ext = bbox.extents(smsp)

        imp = importer.Importer(src, target)
        imp.import_modelspace(target_layout=tmsp)
        imp.finalize()

        #labels run VERTICALLY, upward from the top of each design: a name is up to ~10 mm of text
        #on a 3.1 mm wide design, so horizontal labels would overlap their neighbours.
        tmsp.add_text(
            spec.name, height=LABEL_HEIGHT_UM,
            dxfattribs={"layer": LABEL_LAYER, "rotation": 90},
        ).set_placement((x_cursor + ext.size.x / 2, ext.extmax.y + LABEL_HEIGHT_UM),
                        align=TextEntityAlignment.MIDDLE_LEFT)

        print(f"  {spec.name:28s} x {x_cursor:9.1f} .. {x_cursor + ext.size.x:9.1f} um")
        x_cursor += ext.size.x + gap_um

    return target


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Put generated bundle DXFs side by side in one DXF (no rotation)")
    p.add_argument("names", nargs="*", help="name substrings to include; default all of DESIGNS")
    p.add_argument("--gap", type=float, default=GAP_UM, help="um between designs")
    p.add_argument("--align", choices=("loop", "top"), default="loop",
                   help="line the designs up by the insertion loop (default) or by their "
                        "connector end")
    p.add_argument("--out", help="output DXF path; default derived from the names")
    args = p.parse_args(argv)

    specs = select_specs(args.names)
    print(f"side by side, left to right: {len(specs)} designs")
    doc = build_side_by_side(specs, gap_um=args.gap, align=args.align)

    out = args.out or f"{ensure_out_dir()}/side_by_side_" + (
        "-".join(args.names) if args.names else "all") + ".dxf"
    doc.saveas(out)
    print(f"Saved -> {out}")
    return out


if __name__ == "__main__":
    main()
