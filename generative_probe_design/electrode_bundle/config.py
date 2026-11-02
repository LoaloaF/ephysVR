"""Tuning knobs for the whole pipeline, as dataclasses with today's values as defaults.

One place to change parameters instead of the three separate module-level constant
blocks that used to live in hook_bundle_generator.py, gen_ionp_bundle.py and
build_electrode_flex_mapping.py. To tweak the design, either edit a default here or
construct a config with overrides, e.g. ``BundleConfig(wide_channel=None, hook_drop=300)``.

`num_channels` / `n_pads_per_column` are shared here rather than re-declared per script.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

#All generated output lands in probe_designs_output/ -- one folder, next to the package
#rather than inside it, so the generated probes sit somewhere obvious instead of buried in the
#code. Resolved from the module location, so no path here needs editing on a new machine; move
#the package and the output folder follows it.
#NOTE: the shipped `...Example` DXFs stay in electrode_bundle/designs/ -- they are inputs, not
#output, and WaferNesting's config still points at them there.
_OUT_DIR = str(Path(__file__).resolve().parent.parent / "probe_designs_output")


def ensure_out_dir() -> str:
    """Create the designs/ output folder if it doesn't exist yet. Call before writing."""
    Path(_OUT_DIR).mkdir(parents=True, exist_ok=True)
    return _OUT_DIR


def bundle_dxf_path(tag: str) -> str:
    """Where a tagged, IONP-free bundle DXF lives: written by batch.py, read by
    side_by_side.py. One definition so the two cannot drift apart."""
    return f"{_OUT_DIR}/electrode_bundle_{tag}.dxf"


@dataclass
class BundleConfig:
    """Everything the electrode-bundle generator needs (hook_bundle_generator knobs)."""
    # --- core geometry (micron) ---
    wire_width: float = 2          # width of the wire
    delta_y: float = 8000 / 63     # default staircase step between electrode depths
    l: float = 13                  # side length of the electrode pad
    l_contact: float = 11          # side length of the contact pad (= expected - 2um lateral etch)
    num_channels: int = 64         # channels per shank
    bottom_elec: float = 12000      # depth of the bottom-most electrode
    padding: float = 4
    delta_x: float = 24            # channel pitch (> wire_width + 2*padding)

    # --- per-electrode wire-length control ---
    # channel_index -> wire_length_micron; anything unlisted keeps the default staircase.
    # length_overrides: Dict[int, float] = field(default_factory=lambda: {2: 1000})
    length_overrides: Dict[int, float] = None
    # per-gap spacing overrides (parametric mode only): channel_index -> distance (um) BEFORE
    # that electrode, i.e. the gap between channel i-1 and i. Unlisted channels use `delta_y`.
    # Positions are the cumulative sum of gaps, so changing one gap shifts every electrode above
    # it. Valid keys are 1..num_channels-1 (channel 0 has no gap before it). Ignored when
    # electrode_lengths_profile is set (the profile replaces the whole base staircase).
    delta_y_overrides: Optional[Dict[int, float]] = field(default_factory=lambda:{i: 30.0 for i in range(33, 49)})
    # delta_y_overrides: Optional[Dict[int, float]] = None
    #
    # ANCHOR -- which end stays put when you retune gaps. The staircase is built BOTTOM-UP
    # (bundle.py seeds cy_stack = -delta_y and accumulates upward), so the DEEP TIP (channel 0,
    # at `bottom_elec`) is pinned and the shallow end absorbs every change: tightening gaps pulls
    # the electrodes ABOVE the edited band DOWNWARD, so their wires get LONGER. There is no
    # spacing_anchor option, and none is needed -- pinning the shallow end instead differs by a
    # pure rigid vertical offset (identical spacings), so it is reachable with the existing knob:
    #
    #     span   = 63*delta_y - sum(delta_y - new_gap)   over every overridden gap
    #     shift  = sum(delta_y - new_gap)                # how far the shallow end drops
    #     to pin the shallow end instead: bottom_elec -= shift
    #
    # With today's values (delta_y = 8000/63 = 126.98, 16 gaps tightened to 30.0) the shallow end
    # sits 1551.75 um lower than a uniform staircase, and the span is 6448.25 um (not 8000).
    #
    # INVARIANT either way: the fanout, the bond pads and the common `wire_top` (= bottom_elec -
    # l/2) do NOT depend on delta_y or on the gaps, and never move. Only the electrode contact
    # positions and their own vertical wire lengths change.
    # Optional full formula-driven profile i -> length (replaces the base for ALL channels);
    # length_overrides still applies on top. None -> use the staircase.
    electrode_lengths_profile: Optional[Callable[[int], float]] = None

    # --- corner radii ---
    polyimide_arc_radius: float = 2   # pad corners AND pad<->trace fillets
    metal_pad_arc_radius: float = 2   # metal electrode pad corners (< l/2)
    contact_pad_arc_radius: float = 2 # etched contact pad corners (< l_contact/2)

    # --- fanout / routing knobs ---
    fan_top_tw: float = 10         # trace width up the riser/diagonal
    fan_neck_len: float = 200      # vertical neck above wire_top, at source pitch
    fan_fan_len: float = 2000      # vertical rise of the gather fan
    bundle_pitch: float = 25       # lane pitch of the centred bundle
    pad_approach_deg: float = 140  # interior bend angle riser->diagonal into the pad

    # --- widened channel ---
    wide_channel: Optional[int] = 0   # index of the widened trace, or None for uniform

    # --- hook anchor on the wide channel ---
    hook_on_wide_channel: bool = True
    hook_scale: float = 1.0
    hook_drop: float = 200
    hook_etch_rx: float = 17.5
    hook_etch_ry: float = 17.5
    hook_etch_dx: float = 0.0
    hook_etch_dy: float = -10.0

    # --- rescue ('lollipop') hook on the second channel ---
    second_channel: int = 1
    hook_on_second_channel: bool = True
    rescue_hook_drop: float = 50
    rescue_hook_angle_deg: float = 155
    rescue_hook_arm_len: float = 50
    rescue_hook_circle_width: float = 20
    rescue_hook_circle_r: float = 25
    rescue_hook_etch_r: float = 12.5

    # --- bond pads ---
    pad_pitch: float = 300         # vertical stacking pitch within a column
    pad_diam: float = 200
    pad_row_pitch: float = 1950    # between the two columns -> columns at x = +/-975
    # RIBBON -- the run from the SHOULDER (where the fibers end and the fanout begins, i.e.
    # wire_top) up to the first solder pad. The connector end is fixed geometry, so this is
    # what stays constant when the fiber gets longer or shorter: `pad_first_y` is DERIVED
    # from it rather than set directly, so changing `bottom_elec` slides the whole
    # fanout+pad assembly instead of stretching the ribbon. The default reproduces the
    # historical pad_first_y = 15800 exactly (l_max 11993.5 + 3806.5).
    ribbon_length: float = 3806.5
    n_pads_per_column: int = 33    # 1 REF/GND + 32 connected -> 66 total, 64 connected

    # --- polyimide body / negative ---
    polyimide_width: float = 2700
    polyimide_bundle_margin: float = 0
    polyimide_fan_hw: float = 950
    polyimide_fan_height: float = 2000
    polyimide_curve_offset: float = 550
    polyimide_curve_r: float = 150
    polyimide_join_r: float = 5
    polyimide_negative_margin: float = 200

    # --- optional reference-DXF electrode source (None = parametric grid) ---
    ref_dxf: Optional[str] = None
    ref_layer: str = 'electrode'

    # --- output ---
    out_dxf: str = f"{_OUT_DIR}/electrode_bundle.dxf"

    # --- derived helpers ---
    @property
    def wire_hw(self) -> float:
        """Default polyimide-trace half-width."""
        return self.wire_width / 2 + self.padding

    @property
    def wide_wire_hw(self) -> float:
        """Widened channel's polyimide half-width (= 2*(wire_width/2 + padding))."""
        return 2 * (self.wire_width / 2 + self.padding)

    @property
    def fan_neck_tw(self) -> float:
        """Neck/fan trace width at the electrode (narrow) end."""
        return self.wire_width

    @property
    def l_max(self) -> float:
        return self.bottom_elec - self.l / 2

    @property
    def pad_first_y(self) -> float:
        """y of the bottom (REF/GND) pad = the shoulder plus the ribbon run.

        Derived, not a knob -- set `ribbon_length` instead. Uses `l_max`, the PARAMETRIC
        shoulder; under `ref_dxf` build_bundle can raise wire_top above l_max, and the
        ribbon then runs short by that difference.
        """
        return self.l_max + self.ribbon_length


# 16 magnetic-ID barcode STRIPE CENTERS, normalized 0 -> 1 along the ELECTRODE ARRAY (not the
# whole shank): 0 = the tip, where the deepest electrode sits; 1 = `id_top_margin` um above the
# SHALLOWEST electrode. The array spans ~6.4 mm, so 1 is ~6.5 mm up from the tip -- well below
# the fanout at `wire_top`. Above 1 there are no wells.
# Read by MRI, which needs >=700 wells/stripe, so stripe HEIGHTS are auto-sized at build time
# from the real electrode positions (ionp.build_id_pattern) -- they are NOT fixed here.
# Two always-on thick anchors bracket 7 DATA slots: the BOTTOM anchor is pinned at the tip
# (grown up to the well floor); the TOP anchor is centred near the top. The 16 IDs are encoded
# across the 7 slots with a Hamming[7,4,3] code (`id_codeword`) so every pair of IDs differs in
# >=3 stripes (much more distinguishable than a plain 4-bit code). Presence/position only.
ID_TOP_ANCHOR_CENTER = 0.96                  # always-on thick anchor near the top
ID_SLOT_CENTERS = (0.21, 0.32, 0.43, 0.54, 0.64, 0.75, 0.86)   # 7 Hamming data-stripe centers


def id_codeword(i: int) -> tuple:
    """Hamming[7,4,3]: ID i (0..15) -> 7 stripe on/off bits, min pairwise distance 3.

    4 info bits d0..d3 (the ID) + 3 parity bits; bit k -> data slot at ID_SLOT_CENTERS[k].
    """
    assert 0 <= i < 16, f"ID {i} out of range 0..15"
    d = [(i >> k) & 1 for k in range(4)]
    p = [d[0] ^ d[1] ^ d[3], d[0] ^ d[2] ^ d[3], d[1] ^ d[2] ^ d[3]]
    return tuple(d + p)


@dataclass
class IonpConfig:
    """IONP well pattern knobs (gen_ionp_bundle CONFIG + PATTERN + design rules)."""
    dxf_file: Optional[str] = None   # None -> per-ID f"{_OUT_DIR}/electrode_bundle_IONP_id{NN}.dxf"
    el_layer: str = 'Electrodes'                          # where the 64 contacts live
    ionp_layer: str = 'pattern_ionp'                      # layer the wells are written to
    # vertical extent of the well region (straight-shank only; fanout starts ~4990)
    y_bottom: float = 0.0
    y_top: float = 4990.0
    # well geometry (um)
    ionp_well_d: float = 1.5
    ionp_well_distance: float = 5.0
    ionp_firstwell_distance: float = 15.0
    well_resolution: int = 12
    n_els: int = 64
    pattern_i: int = 0
    json_file: Optional[str] = None   # None -> per-ID f"{_OUT_DIR}/64ch_ionp_id{NN}_info.json"
    # None -> built at runtime from pattern_i + real electrode positions (geometry-dependent
    # heights, so it can't be precomputed). An explicit array still overrides.
    pattern: Optional[np.ndarray] = None
    keepout_radius: float = 30.0      # um; drop wells within this of any contact centre
    # ID region top: sits `id_top_margin` um above the HIGHEST electrode contact (so the
    # barcode lands just above the electrodes, not up by the fan). None -> use y_top (fan).
    id_top_margin: Optional[float] = 220.0
    # anchor thickness (um). Bottom anchor is pinned at the tip and grown UP from this seed to
    # the well floor (sparse tip -> ends up ~840um, the fattest stripe = orientation cue). Top
    # anchor is centred and this height (grown further only if needed for the well floor).
    id_bot_anchor_h_um: float = 550.0
    id_top_anchor_h_um: float = 350.0
    # design-rule limits. min_n_wells is the HARD MRI floor every stripe is auto-sized to meet.
    min_n_wells: int = 700
    max_n_wells: int = 10_000
    min_strip_distance: float = 300   # um

    def __post_init__(self):
        if self.dxf_file is None:
            self.dxf_file = f"{_OUT_DIR}/electrode_bundle_IONP_id{self.pattern_i:02d}.dxf"
        if self.json_file is None:
            self.json_file = f"{_OUT_DIR}/64ch_ionp_id{self.pattern_i:02d}_info.json"


@dataclass
class MappingConfig:
    """Electrode-channel -> flex-pad mapping knobs (build_electrode_flex_mapping)."""
    # 33 ints each, left->right as in 64ch_flex.PNG; entry 0 = REF/GND (65 top / 66 bottom).
    top_row: List[int] = field(default_factory=lambda: [
        65, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
        32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2])
    bottom_row: List[int] = field(default_factory=lambda: [
        66, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
        63, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33])
    n_channels: int = 64
    n_per_column: int = 33   # 1 REF/GND + 32 connected
    out_file: str = f"{_OUT_DIR}/electrode_to_flex_mapping.json"
