"""Electrode-channel -> flex-PCB-pad mapping.

Maps each generator channel index (0-63) to the pad number on the LEFT connector of the
64ch flex PCB. The channel -> bond-pad-position assignment comes from the SAME
`shapes.assign_channels_to_pads` the generator routes with, so the mapping can never drift
from the geometry (it used to be reproduced by hand in
build_electrode_flex_mapping.derive_channel_to_pad).

The flex pad NUMBER at each position comes from the image labels in
config.MappingConfig.top_row / bottom_row (read left->right; entry 0 is the REF/GND pad,
65 top / 66 bottom).

Orientation (bundle top rotated 90deg CW to form the flex left connector):
  bundle LEFT column  -> image TOP row      (top_row)
  bundle RIGHT column -> image BOTTOM row   (bottom_row)
  bundle low-y (REF)  -> image LEFT end of each row (entry 0)
"""
from . import shapes as sh
from .config import MappingConfig


def standalone_chan_pad(n_channels, delta_x=24.0):
    """
    channel -> (column, pad_row) without a full bundle build. The routing order depends
    only on channel x-ORDER (width-independent), so a uniform-width parametric layout gives
    the same assignment build_bundle would. Prefer passing the BundleResult.chan_pad when
    you have it; this is the fallback for running `mapping` on its own.
    """
    wire_hw = 1.0
    xpos = sh.channel_x_positions(n_channels, delta_x,
                                  {i: wire_hw for i in range(n_channels)}, wire_hw)
    top_points = [(xpos[i], 0.0) for i in range(n_channels)]
    return sh.assign_channels_to_pads(top_points)


def print_template(chan_pad, cfg: MappingConfig):
    """Print the per-position table so the user can verify orientation and paste numbers."""
    print("Channel per pad position (row 0 = REF/GND, rows 1..32 = connected):\n")
    for col, header in (('left', 'top_row  (bundle LEFT column)'),
                        ('right', 'bottom_row (bundle RIGHT column)')):
        ref = 65 if col == 'left' else 66
        print(f"{header}:  position 0 = REF/GND = {ref}")
        row = {pr: ci for ci, (c, pr) in chan_pad.items() if c == col}
        cells = [f"pos{pr:>2}->ch{row[pr]:>2}" for pr in range(1, cfg.n_per_column)]
        for i in range(0, len(cells), 8):
            print("   " + "  ".join(cells[i:i + 8]))
        print()


def build_mapping(chan_pad, cfg: MappingConfig = None) -> dict:
    """Join the channel->pad assignment with the flex pad numbers into the summary dict."""
    cfg = cfg or MappingConfig()
    for name, row in (('top_row', cfg.top_row), ('bottom_row', cfg.bottom_row)):
        if len(row) != cfg.n_per_column:
            raise ValueError(f"{name} must have {cfg.n_per_column} entries, got {len(row)}")

    mapping = {}
    used = {}
    for ci in range(cfg.n_channels):
        col, pr = chan_pad[ci]
        flex = (cfg.top_row if col == 'left' else cfg.bottom_row)[pr]
        mapping[str(ci)] = int(flex)
        used.setdefault(int(flex), []).append(ci)

    assert len(mapping) == cfg.n_channels, "missing channels"
    dupes = {f: chs for f, chs in used.items() if len(chs) > 1}
    assert not dupes, f"flex pads reused: {dupes}"
    assert 65 not in used and 66 not in used, "REF/GND pad appeared in mapping"

    return {
        "description": "generator channel index (0-63) -> left-connector flex pad number",
        "flex_side": "left connector (bundle-mating)",
        "ref_gnd_pads": {"left_column_top_row": cfg.top_row[0],
                         "right_column_bottom_row": cfg.bottom_row[0]},
        "n_channels": cfg.n_channels,
        "mapping": mapping,
    }
