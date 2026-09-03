"""CLI entry point: one command for the whole electrode-bundle pipeline.

    python3 -m electrode_bundle.main all       # build + map + IONP, one in-memory pass
    python3 -m electrode_bundle.main bundle     # just the electrode bundle DXF + PNG
    python3 -m electrode_bundle.main mapping     # just the channel->flex JSON
    python3 -m electrode_bundle.main ionp        # stamp IONP wells onto the existing IONP DXF
    python3 -m electrode_bundle.main all --pattern-i 11   # pick magnetic-ID 0..15 (-> _id11 output files)

`all` is the point of the consolidation: build the bundle ONCE and hand its in-memory
results (contact locations + channel->pad assignment) straight to the two consumers, so the
mapping never re-derives the routing order and IONP never re-reads the DXF or needs a rename.
"""
import argparse
import json

from .config import BundleConfig, IonpConfig, MappingConfig, ensure_out_dir
from . import bundle as bundle_mod
from . import mapping as mapping_mod
from . import ionp as ionp_mod


def _ensure_usable_font():
    """Repair a truncated matplotlib font cache before drawing any text.

    Seen on macOS: ~/.matplotlib/fontlist-v*.json can be left holding only a partial scan
    (e.g. 35 system fonts, none of them the DejaVu Sans that ships WITH matplotlib). Every
    text draw then dies with "Failed to find font DejaVu Sans ... and fallback to the
    default font was disabled" -- matplotlib disables the fallback while resolving the
    default itself, so there is nothing left to fall back to. Rescanning fixes it and
    rewrites the cache, so this costs one slow run and never fires again.
    """
    from matplotlib import font_manager, rcParams
    have = {f.name for f in font_manager.fontManager.ttflist}
    if any(name in have for name in rcParams["font.sans-serif"]):
        return
    print("matplotlib's font cache lists no usable sans-serif font -- rebuilding it "
          "(one-off, ~10 s)")
    font_manager.fontManager = font_manager._load_fontmanager(try_read_cache=False)


def _save_png(result, out_png, ionp_xy=None, title=None):
    import matplotlib
    matplotlib.use("Agg")
    _ensure_usable_font()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(20, 20))
    bundle_mod.render_bundle(result, ax)
    if ionp_xy is not None:
        ax.scatter(ionp_xy[0], ionp_xy[1], color="blue", s=2, alpha=0.5)
    if title:
        ax.set_title(title)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved render -> {out_png}")


def _write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"Wrote {path}")


def cmd_bundle(bcfg: BundleConfig, render: bool = True):
    result = bundle_mod.build_bundle(bcfg)
    result.doc.saveas(bcfg.out_dxf)
    print(f"Saved -> {bcfg.out_dxf}")
    if render:
        _save_png(result, bcfg.out_dxf.replace(".dxf", ".png"),
                  title=f"electrode bundle ({result.num_channels} ch)")
    return result


def build_mapping_summary(bcfg: BundleConfig, mcfg: MappingConfig, result=None):
    """The channel->flex-pad summary dict, without writing any file.

    Prefer the real assignment from a bundle build; else derive it standalone. Used both by
    the `mapping` subcommand (which writes the standalone JSON) and by the `all`/`ionp`
    pipelines (which embed it into the IONP info JSON instead)."""
    chan_pad = result.chan_pad if result is not None else \
        mapping_mod.standalone_chan_pad(mcfg.n_channels, bcfg.delta_x)
    return mapping_mod.build_mapping(chan_pad, mcfg)


def _print_mapping(summary, n_channels):
    for ci in range(n_channels):
        print(f"  ch{ci:>2} -> flex {summary['mapping'][str(ci)]:>2}")


def cmd_mapping(bcfg: BundleConfig, mcfg: MappingConfig, result=None):
    """Standalone `mapping` subcommand: no IONP info JSON exists here to embed into, so this
    is the one path that still writes electrode_to_flex_mapping.json."""
    summary = build_mapping_summary(bcfg, mcfg, result=result)
    _write_json(summary, mcfg.out_file)
    _print_mapping(summary, mcfg.n_channels)
    return summary


def cmd_ionp_standalone(bcfg: BundleConfig, mcfg: MappingConfig, icfg: IonpConfig):
    """Stamp wells onto an existing DXF (the old gen_ionp_bundle in-place behaviour)."""
    import ezdxf
    doc = ezdxf.readfile(icfg.dxf_file)
    msp = doc.modelspace()
    el_locs = ionp_mod.electrode_locs_from_dxf(msp, icfg)
    ionp_x, ionp_y, el_locs, pattern = ionp_mod.add_ionp_wells(doc, msp, el_locs, icfg)
    doc.saveas(icfg.dxf_file)
    print(f"Saved -> {icfg.dxf_file}")
    #No bundle result here; derive the mapping standalone and embed it in the info JSON.
    flex_mapping = build_mapping_summary(bcfg, mcfg)
    ionp_mod.write_pattern_summary_to_json(
        pattern, ionp_mod.effective_y_top(el_locs, icfg), icfg.y_bottom,
        el_locs, ionp_x, ionp_y, icfg, icfg.json_file, flex_mapping=flex_mapping)


def cmd_all(bcfg: BundleConfig, mcfg: MappingConfig, icfg: IonpConfig, render: bool = True):
    #1) build the bundle once
    result = bundle_mod.build_bundle(bcfg)
    #2) mapping from the same assignment the geometry used (embedded in the info JSON, not
    #   written as a standalone file)
    flex_mapping = build_mapping_summary(bcfg, mcfg, result=result)
    _print_mapping(flex_mapping, mcfg.n_channels)
    #3) IONP wells onto the same doc, from the in-memory contact locations (no DXF re-read)
    ionp_x, ionp_y, el_locs, pattern = ionp_mod.add_ionp_wells(result.doc, result.msp,
                                                               result.electrode_locs, icfg)
    #4) save the DXF (IONP superset) + summary (with embedded mapping) + render
    result.doc.saveas(icfg.dxf_file)
    print(f"Saved -> {icfg.dxf_file}")
    ionp_mod.write_pattern_summary_to_json(
        pattern, ionp_mod.effective_y_top(el_locs, icfg), icfg.y_bottom,
        el_locs, ionp_x, ionp_y, icfg, icfg.json_file, flex_mapping=flex_mapping)
    if render:
        _save_png(result, icfg.dxf_file.replace(".dxf", ".png"), ionp_xy=(ionp_x, ionp_y),
                  title=f"electrode bundle + IONP ({len(ionp_x)} wells)")
    return result


def cmd_legend():
    """Render the 16 magnetic-ID barcodes as a decode reference card."""
    from .id_legend import render_id_legend
    out_png = f"{ensure_out_dir()}/ionp_id_legend.png"
    render_id_legend(out_png)
    print(f"Saved legend -> {out_png}")


def _build_parser():
    p = argparse.ArgumentParser(description="Electrode-bundle pipeline")
    p.add_argument("command", choices=["bundle", "mapping", "ionp", "all", "legend"],
                   help="which step to run")
    p.add_argument("--pattern-i", type=int, choices=range(16), default=0, metavar="0-15",
                   help="IONP magnetic-ID pattern to stamp (0-15); sets the output filenames")
    p.add_argument("--no-render", action="store_true", help="skip PNG rendering")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    print(f"Designs -> {ensure_out_dir()}")

    bcfg, mcfg = BundleConfig(), MappingConfig()
    icfg = IonpConfig(pattern_i=args.pattern_i)
    render = not args.no_render

    if args.command == "bundle":
        cmd_bundle(bcfg, render=render)
    elif args.command == "mapping":
        cmd_mapping(bcfg, mcfg, result=None)
    elif args.command == "ionp":
        cmd_ionp_standalone(bcfg, mcfg, icfg)
    elif args.command == "all":
        cmd_all(bcfg, mcfg, icfg, render=render)
    elif args.command == "legend":
        cmd_legend()


if __name__ == "__main__":
    main()
