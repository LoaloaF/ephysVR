"""Config -- the three connector designs, reproducing the fabricated 2026-07 wafer.

DEFAULT config (used when WAFERNEST_CONFIG is unset).

NOTE FOR THE HANDOVER: this config needs the connector DXFs from the repository's
NewInterconnect/ folder, which is NOT part of the electrode_bundle + WaferNesting
handover. Without it, stages 01-04 will stop with a clear "missing DXF" message naming
DESIGN_DIR below. It is kept because reference_run_2026-07/ (the fabricated wafer's
footprints and poses) is the seed and the provenance for everything here, and because it
is the worked example of a multi-design, seeded nest.

For a runnable example out of the box, use config_electrode_bundle.py instead.
"""
import os, sys

# =============================================================================
# PATHS -- the only lines that depend on where the folders live
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))     # .../WaferNesting
ROOT = os.path.dirname(HERE)                          # the common folder above it

# Folder holding the connector DXFs listed in DESIGNS. In the original repository this is
# mea1k_connector/NewInterconnect. Put an absolute path here if they live elsewhere.
DESIGN_DIR = os.path.join(ROOT, "NewInterconnect")

# The base wafer GDS (wafer outline + alignment marks). A copy ships inside this folder.
WAFER_TEMPLATE = os.path.join(HERE, "new_wafer_actually.GDS")
MARKS_WKT      = os.path.join(HERE, "marks_union.wkt")


# =============================================================================
# WHAT TO NEST
# =============================================================================
# One entry per distinct design.
#
#   key    short identifier; names the output files and the GDS cell (CONN_<KEY>)
#   dxf    filename inside DESIGN_DIR. Coordinates MUST be in micrometres, and only
#          LWPOLYLINE and HATCH entities are read
#   kind   nest-slot group. Designs sharing a kind are nested against the union ENVELOPE
#          of their footprints and are interchangeable at export time, so only group
#          designs of genuinely similar shape -- every slot of that kind reserves
#          envelope-sized space. A design alone in its kind uses its own exact footprint.
#   count  how many copies go on the wafer
RUN_TAG = "example_4x3"

DESIGNS = [
    {"key": "big",
     "dxf": "new_interconnect_with_connector_chamfered_12Block_56_15.dxf",
     "kind": "big",   "count": 4},
    {"key": "small_51_35",
     "dxf": "new_interconnect_with_connector_8blocks_chamfered_1layer_51_35.dxf",
     "kind": "small", "count": 4},
    {"key": "small_71_15",
     "dxf": "new_interconnect_with_connector_8blocks_chamfered_1layer_71_15.dxf",
     "kind": "small", "count": 4},
]


# =============================================================================
# FOOTPRINT EXTRACTION (stage 01)
# =============================================================================
# The silhouette is the union of EVERY layer -- see HANDOVER.md 3.2.
CLOSE_GAP         = 0.08
SIMPLIFY          = 0.03
ENVELOPE_PAD      = 0.01
ENVELOPE_SIMPLIFY = 0.01   # MUST stay <= ENVELOPE_PAD (HANDOVER.md gotcha 4)


# =============================================================================
# WAFER GEOMETRY AND CLEARANCES (stage 02)
# =============================================================================
WAFER_R      = 50.0
WAFER_FLAT_Y = -47.285

# Measurements of the fabricated wafer, not preferences -- HANDOVER.md 3.3.
EDGE_EXCL = 3.0
GAP       = 0.05
MARK_BUF  = 0.7


# =============================================================================
# NESTING SEARCH (stage 02)
# =============================================================================
# "seeded"  start from SEED_POSES, matching pieces to same-kind seed poses; anything the
#           seed has no pose for goes into the slot of a seed piece not being placed this
#           time, then the whole layout is polished at zero temperature before restarts.
# "free"    no seed; pieces arrange themselves from random layouts.
SEED_MODE  = "seeded"
SEED_POSES = os.path.join(HERE, "reference_run_2026-07", "wafer_new_4x3_slot_poses.json")

CARDINAL_ONLY = False
RESTARTS = 40
ITERS    = 160000
RNG_SEED = 11


# =============================================================================
# FAB LAYER MAP (stages 03 and 04)
# =============================================================================
# ANY LAYER NOT LISTED IS SILENTLY DELETED ON EXPORT. Stage 03 prints what it mapped and
# what it dropped -- read that output. Positive Polyimide is dropped deliberately.
LAYER_MAP = {
    "polyimide_negative": (1, 0),
    "etchingpad":         (1, 0),
    "metal1":             (3, 0),
    "etching":            (5, 0),
    "via":                (5, 0),
    "metal2":             (6, 0),
    "metal3":             (7, 0),
}

import config_common; config_common.derive(sys.modules[__name__])
