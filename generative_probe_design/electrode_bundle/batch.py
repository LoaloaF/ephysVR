"""Build every design in design_sets.DESIGNS.

    python3 -m electrode_bundle.batch
    python3 -m electrode_bundle.batch --no-render

Each DesignSpec becomes one BundleConfig: `sites` resolves to the spacing knobs, then
`fiber_length`/`loop_offset` are solved for (lengths.py). A design with no `magnetic_id`
writes only the bundle DXF + the flex-pad mapping JSON; one with an id runs the same
in-memory `all` pipeline main.py uses (build once, mapping + IONP off the same result).
Outputs are named after the design, so variants never overwrite each other.

Finishes by writing `all_designs_side_by_side.dxf`, every design laid out in a row in one
drawing -- open that one file to inspect the whole set at once.
"""
import argparse
from dataclasses import replace

from . import bundle as bundle_mod
from .config import BundleConfig, IonpConfig, MappingConfig, bundle_dxf_path, ensure_out_dir
from .design_sets import DESIGNS
from .lengths import solve_lengths
from .main import _save_png, cmd_all, cmd_mapping
from .side_by_side import build_side_by_side


def config_for(spec) -> BundleConfig:
    """The BundleConfig a DesignSpec describes, lengths solved."""
    kwargs = dict(spec.sites.delta_y_kwargs(spec.n_channels), num_channels=spec.n_channels)
    cfg = solve_lengths(BundleConfig(**kwargs),
                        fiber_length=spec.fiber_length, loop_offset=spec.loop_offset)

    if spec.overall_length is not None:
        #the ribbon absorbs whatever the fiber does not use, so designs with different
        #fiber lengths still end at the same connector position
        ribbon = spec.overall_length - spec.fiber_length
        if ribbon <= cfg.polyimide_fan_height:
            raise ValueError(
                f"{spec.name}: overall_length {spec.overall_length} leaves only {ribbon} um "
                f"of ribbon for a {cfg.polyimide_fan_height} um fanout -- it must exceed "
                f"fiber_length ({spec.fiber_length}) by more than that")
        cfg = replace(cfg, ribbon_length=ribbon)
    return cfg


def _run_bundle_and_mapping_only(bcfg, spec, out_dir, render):
    """No IONP: just the geometry DXF/PNG + the flex-pad mapping JSON."""
    result = bundle_mod.build_bundle(bcfg)
    dxf_path = bundle_dxf_path(spec.name)
    result.doc.saveas(dxf_path)
    print(f"Saved -> {dxf_path}")
    if render:
        _save_png(result, dxf_path.replace(".dxf", ".png"),
                  title=f"{spec.name} ({result.num_channels} ch) - {spec.doc_id}")

    mcfg = MappingConfig(out_file=f"{out_dir}/electrode_to_flex_mapping_{spec.name}.json")
    cmd_mapping(bcfg, mcfg, result=result)


def run_batch(render: bool = True, side_by_side: bool = True):
    out_dir = ensure_out_dir()
    for spec in DESIGNS:
        bcfg = config_for(spec)
        print(f"\n=== {spec.name} ({spec.doc_id}) ===")

        if spec.magnetic_id is None:
            _run_bundle_and_mapping_only(bcfg, spec, out_dir, render)
        else:
            icfg = IonpConfig(
                pattern_i=spec.magnetic_id,
                dxf_file=f"{out_dir}/electrode_bundle_{spec.name}_id{spec.magnetic_id:02d}.dxf",
                json_file=f"{out_dir}/{spec.name}_id{spec.magnetic_id:02d}_info.json",
            )
            cmd_all(bcfg, MappingConfig(), icfg, render=render)

    if side_by_side:
        #every design in one DXF, in DESIGNS order, for eyeballing them together
        print(f"\n=== all {len(DESIGNS)} side by side ===")
        out = f"{out_dir}/all_designs_side_by_side.dxf"
        build_side_by_side(DESIGNS).saveas(out)
        print(f"Saved -> {out}")


def _build_parser():
    p = argparse.ArgumentParser(
        description="Build every electrode_bundle design in design_sets.DESIGNS")
    p.add_argument("--no-render", action="store_true", help="skip PNG rendering")
    p.add_argument("--no-side-by-side", action="store_true",
                   help="skip the combined all-designs-in-one DXF")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    run_batch(render=not args.no_render, side_by_side=not args.no_side_by_side)


if __name__ == "__main__":
    main()
