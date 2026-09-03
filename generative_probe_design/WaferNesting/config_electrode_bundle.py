"""Config -- electrode bundles from the electrode_bundle generator.

General: point BUNDLE_DXF at any bundle the generator produced and set COUNT. Nothing in
this file is specific to one magnetic ID.

    WAFERNEST_CONFIG=config_electrode_bundle python3 WaferNesting/01_extract_footprints.py

Run 00_check_setup.py first -- it resolves every path below and tells you exactly which
line to edit if something is missing.
"""
import os, sys

# =============================================================================
# PATHS -- the only lines that depend on where the folders live
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))     # .../WaferNesting
ROOT = os.path.dirname(HERE)                          # the common folder above it

# Expected layout (both folders side by side in one common folder):
#
#     mea1k/
#       electrode_bundle/
#         designs/           <- the generator writes its DXFs here
#       WaferNesting/        <- this folder
#
# If your electrode_bundle folder is somewhere else, put its designs/ path here --
# an absolute path is fine.
DESIGN_DIR = os.path.join(ROOT, "electrode_bundle", "designs")

# The base wafer GDS (wafer outline + alignment marks). A copy ships inside this folder,
# so this normally needs no editing.
WAFER_TEMPLATE = os.path.join(HERE, "new_wafer_actually.GDS")
MARKS_WKT      = os.path.join(HERE, "marks_union.wkt")


# =============================================================================
# WHAT TO NEST
# =============================================================================
# A DXF inside DESIGN_DIR. The generator names them:
#     electrode_bundle.dxf                  (no magnetic ID)
#     electrode_bundle_IONP_id{NN}.dxf      (with the ID barcode, NN = 00..15)
#
# The DXFs shipped in designs/ carry an "Example" suffix, so that is what this points at.
# After you generate your own with
#     python3 -m electrode_bundle.main all --pattern-i NN
# the file will be named WITHOUT that suffix -- update this line to match. Run
# 00_check_setup.py and it will list every DXF actually present in designs/.
BUNDLE_DXF = "electrode_bundle_IONP_id11Example.dxf"

# How many copies. This is an INPUT, not a maximum -- the nester places exactly this many
# and stops as soon as the layout is legal; it has no density objective and will not fill
# the wafer for you. See HANDOVER.md 3.1 before raising it: past ~40 pieces the solver
# gets slow, and for a near-rectangular strip a structured nest fits far more (66 vs 24
# for the id11 bundle).
COUNT = 24

RUN_TAG = os.path.splitext(BUNDLE_DXF)[0]     # outputs go to runs/<RUN_TAG>/

DESIGNS = [
    {"key": "bundle", "dxf": BUNDLE_DXF, "kind": "bundle", "count": COUNT},
]


# =============================================================================
# FOOTPRINT EXTRACTION (stage 01)
# =============================================================================
# The silhouette is the union of EVERY layer -- see HANDOVER.md 3.2.
CLOSE_GAP         = 0.08   # mm; closes hairline splits between adjacent shapes
SIMPLIFY          = 0.03   # mm; silhouette simplification tolerance
ENVELOPE_PAD      = 0.01   # only used when a kind groups >1 design
ENVELOPE_SIMPLIFY = 0.01   # MUST stay <= ENVELOPE_PAD (HANDOVER.md gotcha 4)


# =============================================================================
# WAFER GEOMETRY AND CLEARANCES (stage 02)
# =============================================================================
WAFER_R      = 50.0     # mm, physical wafer radius
WAFER_FLAT_Y = -47.285  # mm, y of the primary flat

# Measurements of the fabricated wafer, not preferences -- HANDOVER.md 3.3.
EDGE_EXCL = 3.0
GAP       = 0.05
MARK_BUF  = 0.7


# =============================================================================
# NESTING SEARCH (stage 02)
# =============================================================================
# No comparable prior layout exists for a bundle, so the pieces arrange themselves.
SEED_MODE  = "free"
SEED_POSES = None

CARDINAL_ONLY = False   # True restricts every piece to 0/90/180/270 degrees
RESTARTS = 20
ITERS    = 120000
RNG_SEED = 7


# =============================================================================
# FAB LAYER MAP (stages 03 and 04)
# =============================================================================
# DXF layer name (lower-cased) -> (GDS layer, datatype).
# ANY LAYER NOT LISTED IS SILENTLY DELETED ON EXPORT. Stage 03 prints what it mapped and
# what it dropped -- read that output.
#
# !!! PROVISIONAL -- NOT CONFIRMED FOR FABRICATION, see HANDOVER.md 6.1 !!!
#   metal        -> 3/0  single-metal design, takes the Metal1 slot
#   electrodes   -> 3/0  geometrically a no-op: Electrodes lies entirely inside Metal
#                        (0.000000 mm2 outside), so merging cannot change the mask
#   pattern_ionp -> 5/0  the IONP wells joined to Etching on the assumption they are
#                        etched features. NOT a no-op -- this is the one to confirm.
#   polyimide            dropped; fab uses the negative
LAYER_MAP = {
    "polyimide_negative": (1, 0),
    "metal":              (3, 0),
    "electrodes":         (3, 0),
    "etching":            (5, 0),
    "pattern_ionp":       (5, 0),
}

import config_common; config_common.derive(sys.modules[__name__])
