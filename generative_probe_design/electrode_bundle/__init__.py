"""electrode_bundle: one pipeline for the 64-channel electrode bundle.

A producer (`bundle.build_bundle`) plus two consumers (`mapping`, `ionp`), driven by
`main` (CLI: bundle | mapping | ionp | all). Replaces the three loosely-wired scripts
hook_bundle_generator.py, build_electrode_flex_mapping.py and gen_ionp_bundle.py.
"""
from .config import BundleConfig, IonpConfig, MappingConfig
from .bundle import build_bundle, render_bundle, BundleResult

__all__ = [
    "BundleConfig", "IonpConfig", "MappingConfig",
    "build_bundle", "render_bundle", "BundleResult",
]
