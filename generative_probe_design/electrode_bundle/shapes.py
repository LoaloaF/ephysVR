"""Domain shapes for the electrode bundle, built on the primitives in ``geometry.py``.

These functions know about the *device* -- polyimide pads and traces, the metal
electrode pad+wire, the fanout body, the bond-pad columns, the channel->pad routing,
and the two anchor "hooks". Some use shapely for boolean unions / morphological
rounding. Moved verbatim from hook_bundle_generator.py, except:

  - ``assign_channels_to_pads`` is newly extracted from ``build_pad_routes`` so the
    channel->pad ordering has a single home. ``build_pad_routes`` now calls it, and
    so does ``mapping.py`` -- the flex mapping no longer *reproduces* the ordering by
    hand (it used to, in build_electrode_flex_mapping.derive_channel_to_pad).
"""
import numpy as np

from .geometry import _arc, stroke_centerline_to_polygon, create_polygon_circle


def create_polyimide_pads_with_arcs(x0, y0, w, l, arc_radius):
    """
    Create a polyimide pad with rounded corners (arcs) around the electrode pad.
    The arcs are quarter circles with the specified radius. The resulting shape is
    a single closed polygon defined by its vertices.

    Args:
        x0: X coordinate of the bottom-left corner of the pad
        y0: Y coordinate of the bottom-left corner of the pad
        w: Width of the pad
        l: Length of the pad
        arc_radius: Radius of the rounded corners (must be less than min(w,l)/2)

    Returns:
        Tuple of (x, y) coordinate arrays tracing one closed rounded rectangle CCW.
        Sign of w/l is irrelevant: the pad is built from the actual x/y bounds, so
        the left-side channels (which pass a negative w) work the same as the right.
    """
    # Normalize to positive-extent bounds so negative w/l (left-side channels) work.
    xmin, xmax = sorted([x0, x0 + w])
    ymin, ymax = sorted([y0, y0 + l])
    width, length = xmax - xmin, ymax - ymin

    if arc_radius >= min(width, length) / 2:
        raise ValueError("Arc radius must be less than half of the minimum dimension (width or length).")

    # (corner center, start_angle) for each rounded corner, in perimeter order.
    # Each arc ends where the next begins, so the straight edges are drawn for free.
    corners = [
        ((xmax - arc_radius, ymin + arc_radius), 1.5 * np.pi),  # bottom-right: 270->360
        ((xmax - arc_radius, ymax - arc_radius), 0.0),          # top-right:      0->90
        ((xmin + arc_radius, ymax - arc_radius), 0.5 * np.pi),  # top-left:      90->180
        ((xmin + arc_radius, ymin + arc_radius), 1.0 * np.pi),  # bottom-left:  180->270
    ]

    x, y = [], []
    for (cx, cy), a0 in corners:
        angles = np.linspace(a0, a0 + np.pi / 2, num=24)
        x.extend(cx + arc_radius * np.cos(angles))
        y.extend(cy + arc_radius * np.sin(angles))

    return np.array(x), np.array(y)


def create_polyimide_outline(xc, cy, pad_hw, wire_hw, wire_top_y, wire_bot_y, tip_y, R):
    """
    Build the full polyimide outline of one channel as a single closed polygon:
    a flat-topped upper trace, the (wider) pad with convex rounded corners, and a
    lower trace ending in a sharp triangular tip. The four pad<->trace steps are
    smoothed with *concave* fillet arcs of radius R (same radius as the pad
    corners), so a narrow trace meets the wide pad with a rounded transition
    instead of a sharp shoulder.

    The shape is symmetric about the channel centre xc (the pad and both traces
    share that centre line), so the +/-1 side only translates xc -- no mirroring
    of the shape is needed.

    Args:
        xc: channel centre x (microns)
        cy: pad centre y (microns)
        pad_hw: pad half-width (=10.5 for l+2*padding)
        wire_hw: trace half-width (=5 for wire_width+2*padding)
        wire_top_y: y of the flat top edge of the upper trace
        wire_bot_y: y of the bottom edge of the lower trace rectangle (tip base)
        tip_y: y of the triangular tip point (below wire_bot_y)
        R: fillet / corner radius. Must be < (pad_hw - wire_hw)/2 so the pad edge
           between a convex corner and the adjacent concave fillet stays positive.

    Returns:
        Tuple of (x, y) coordinate arrays tracing one closed polygon (clockwise).
    """
    if R >= (pad_hw - wire_hw) / 2:
        raise ValueError("Fillet radius must be < (pad_hw - wire_hw)/2 so the pad shoulder stays positive.")

    pad_top, pad_bot = cy + pad_hw, cy - pad_hw
    HALF = np.pi / 2
    x, y = [], []

    def add(seg):
        x.extend(seg[0]); y.extend(seg[1])

    # Clockwise from the top-left of the upper trace.
    x.append(xc - wire_hw); y.append(wire_top_y)                                  # upper trace top-left
    add(_arc(xc - wire_hw - R, pad_top + R, R, 0.0, -HALF))                       # concave fillet: top-left junction
    add(_arc(xc - pad_hw + R, pad_top - R, R, HALF, np.pi))                       # convex: pad top-left
    add(_arc(xc - pad_hw + R, pad_bot + R, R, np.pi, 1.5 * np.pi))                # convex: pad bottom-left
    add(_arc(xc - wire_hw - R, pad_bot - R, R, HALF, 0.0))                        # concave fillet: bottom-left junction
    x.append(xc - wire_hw); y.append(wire_bot_y)                                  # lower trace bottom-left
    x.append(xc);           y.append(tip_y)                                       # sharp tip
    x.append(xc + wire_hw); y.append(wire_bot_y)                                  # lower trace bottom-right
    add(_arc(xc + wire_hw + R, pad_bot - R, R, np.pi, HALF))                      # concave fillet: bottom-right junction
    add(_arc(xc + pad_hw - R, pad_bot + R, R, 1.5 * np.pi, 2 * np.pi))            # convex: pad bottom-right
    add(_arc(xc + pad_hw - R, pad_top - R, R, 0.0, HALF))                         # convex: pad top-right
    add(_arc(xc + wire_hw + R, pad_top + R, R, 1.5 * np.pi, np.pi))               # concave fillet: top-right junction
    x.append(xc + wire_hw); y.append(wire_top_y)                                  # upper trace top-right

    return np.array(x), np.array(y)


def create_metal_pad_wire_outline(xc, cy, pad_hw, wire_hw, wire_top_y, pad_R, shoulder_R):
    """
    One closed metal polygon per channel: a rounded-corner electrode PAD (half-width pad_hw,
    centred at (xc, cy)) with the narrow WIRE (half-width wire_hw) rising from the pad's top
    edge up to wire_top_y. The four pad corners are convex arcs of radius pad_R; the two
    pad<->wire shoulders are concave fillet arcs of radius shoulder_R, so the wire meets the
    wider pad with a rounded transition instead of a sharp 90-deg step. Symmetric about xc, so
    left-side channels (xc < 0) work unchanged. Returns (x, y) tracing one closed polygon.
    """
    if pad_R >= pad_hw:
        raise ValueError("pad_R must be < pad_hw.")
    if pad_R + wire_hw + shoulder_R >= pad_hw:
        raise ValueError("Metal pad radii too large: need pad_R + wire_hw + shoulder_R < pad_hw "
                         f"({pad_R} + {wire_hw} + {shoulder_R} vs {pad_hw}).")
    HALF = np.pi / 2
    pad_top, pad_bot = cy + pad_hw, cy - pad_hw
    x, y = [], []
    def add(seg): x.extend(seg[0]); y.extend(seg[1])
    # Clockwise from the wire top-left.
    x.append(xc - wire_hw); y.append(wire_top_y)                                        # wire top-left
    add(_arc(xc - wire_hw - shoulder_R, pad_top + shoulder_R, shoulder_R, 0.0, -HALF))  # concave shoulder (left)
    add(_arc(xc - pad_hw + pad_R, pad_top - pad_R, pad_R, HALF, np.pi))                 # convex pad top-left
    add(_arc(xc - pad_hw + pad_R, pad_bot + pad_R, pad_R, np.pi, 1.5 * np.pi))          # convex pad bottom-left
    add(_arc(xc + pad_hw - pad_R, pad_bot + pad_R, pad_R, 1.5 * np.pi, 2 * np.pi))      # convex pad bottom-right
    add(_arc(xc + pad_hw - pad_R, pad_top - pad_R, pad_R, 0.0, HALF))                   # convex pad top-right
    add(_arc(xc + wire_hw + shoulder_R, pad_top + shoulder_R, shoulder_R, 1.5 * np.pi, np.pi))  # concave shoulder (right)
    x.append(xc + wire_hw); y.append(wire_top_y)                                        # wire top-right
    return np.array(x), np.array(y)


def build_polyimide_fanout_body(bot_hw, fan_hw, top_hw,
                                y_bottom, y_fan_top, y_curve, y_top, R, corner_R=0.0):
    """
    Closed polyimide body outline (x, y arrays), symmetric about x=0, built bottom->top:
      a flat BOTTOM edge (half-width bot_hw) hugging the wire bundle -> a diagonal FAN-OUT
      to (fan_hw, y_fan_top) -> a vertical STEM at fan_hw -> a tangent-fillet FLARE (radius R)
      out to the wide BLOCK (half-width top_hw) at y_curve -> straight up to y_top over the
      pad columns. Only the right-hand profile is built; it is mirrored to the left so the
      whole body is one closed, symmetric ring (top and bottom edges close implicitly).

    Args:
        bot_hw: bottom half-width (hugs the wire bundle + margin)
        fan_hw: stem half-width the fan-out opens to
        top_hw: wide block half-width over the pad columns
        y_bottom: y of the flat bottom edge (= wire_top)
        y_fan_top: y where the diagonal fan-out finishes (stem begins)
        y_curve: y of the horizontal ledge where the flare turns out to the block
        y_top: y of the top edge over the pads
        R: tangent fillet radius of the two flare arcs
        corner_R: if > 0, round the two outer bottom corners (where the flat bottom meets the
            fan-out diagonal) with a convex fillet of this radius. The trace<->body junction
            rounding downstream only rounds concave corners, so these convex corners are
            filleted here instead.
    """
    if fan_hw < bot_hw:
        print(f"WARNING: POLYIMIDE_FAN_HW ({fan_hw}) < bottom half-width ({bot_hw:.1f}); "
              "the polyimide fans IN rather than out.")
    if fan_hw + R > top_hw - R:
        raise ValueError("No room for both fillets: need fan_hw + R <= top_hw - R "
                         f"({fan_hw} + {R} vs {top_hw} - {R}).")
    if not (y_bottom < y_fan_top < y_curve - R and y_curve + R < y_top):
        raise ValueError("Polyimide fanout heights out of order: need "
                         "y_bottom < y_fan_top < y_curve - R and y_curve + R < y_top.")

    HALF = np.pi / 2
    rx, ry = [], []
    def add(seg): rx.extend(seg[0]); ry.extend(seg[1])
    if corner_R > 0:
        #Convex fillet of the outer bottom corner (bot_hw, y_bottom) where the horizontal
        #bottom meets the diagonal fan-out. Tangent length t = corner_R / tan(theta/2), with
        #theta the interior angle between the two edges (bottom -> corner, corner -> stem top).
        e_in = np.array([bot_hw - 0.0, 0.0])                     # direction into the corner (bottom, +x)
        e_out = np.array([fan_hw - bot_hw, y_fan_top - y_bottom])# direction out (up the diagonal)
        e_in = e_in / np.hypot(*e_in); e_out = e_out / np.hypot(*e_out)
        theta = np.arccos(np.clip(np.dot(-e_in, e_out), -1.0, 1.0))  # interior angle at the corner
        t = corner_R / np.tan(theta / 2.0)
        p_bot = np.array([bot_hw - t, y_bottom])                 # tangent point on the flat bottom
        p_dia = np.array([bot_hw, y_bottom]) + e_out * t         # tangent point on the diagonal
        cen = p_bot + np.array([0.0, corner_R])                  # arc centre (corner_R above the bottom)
        a0 = np.arctan2(p_bot[1] - cen[1], p_bot[0] - cen[0])
        a1 = np.arctan2(p_dia[1] - cen[1], p_dia[0] - cen[0])
        rx.append(p_bot[0]); ry.append(p_bot[1])                 # bottom (start of the fillet)
        add(_arc(cen[0], cen[1], corner_R, a0, a1))              # convex outer-bottom fillet
        rx.append(fan_hw); ry.append(y_fan_top)                  # end of diagonal fan-out
    else:
        rx.append(bot_hw); ry.append(y_bottom)                   # bottom (hugs bundle + margin)
        rx.append(fan_hw); ry.append(y_fan_top)                  # end of diagonal fan-out
    add(_arc(fan_hw + R, y_curve - R, R, np.pi, HALF))            # convex: stem -> horizontal ledge
    add(_arc(top_hw - R, y_curve + R, R, 1.5 * np.pi, 2 * np.pi)) # concave: ledge -> block edge
    rx.append(top_hw); ry.append(y_top)                           # up the block to the top edge
    x = rx + [-v for v in rx[::-1]]                               # mirror to the left, top->bottom
    y = ry + ry[::-1]
    return np.array(x), np.array(y)


def merge_polyimide_with_fillets(body_xy, part_xys, band_ylo, band_yhi, R):
    """
    Union the fanout body with every per-channel polyimide part (traces + hooks) into ONE
    polygon, then ROUND the trace<->body junction corners with radius R. The rounding is a
    morphological closing (dilate by R, erode by R) applied ONLY inside the horizontal band
    [band_ylo, band_yhi] where the junctions live, so everything outside the band (pads,
    tips, hook barbs) is preserved exactly.

    Because closing narrows every gap by 2R before restoring it, R must be < half the
    inter-trace gap or neighbouring traces fuse; that also caps the notch-ceiling fillet
    radius at R.

    Inputs:
        body_xy: (x_array, y_array) of the fanout body outline.
        part_xys: list of (x_array, y_array), one per polyimide trace/hook.
        band_ylo, band_yhi: y-range of the junction band to round within.
        R: fillet radius.

    Returns:
        list of exterior rings [[(x, y), ...], ...] (usually one).
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    polys = [Polygon(list(zip(body_xy[0], body_xy[1]))).buffer(0)]
    for px, py in part_xys:
        polys.append(Polygon(list(zip(px, py))).buffer(0))
    merged = unary_union(polys)
    xmin, _, xmax, _ = merged.bounds
    band = box(xmin - 1.0, band_ylo, xmax + 1.0, band_yhi)
    closed = merged.intersection(band).buffer(R, join_style=1).buffer(-R, join_style=1)
    result = unary_union([merged.difference(band), closed.intersection(band)])
    geoms = list(result.geoms) if result.geom_type == "MultiPolygon" else [result]
    return [list(g.exterior.coords) for g in geoms]


def polygon_to_seamed_ring(g):
    """
    Flatten a shapely Polygon-with-holes into ONE closed ring (list of (x, y)) so a single
    LWPOLYLINE renders as a filled region with EMPTY holes: each hole is tied to the outer
    boundary by a zero-width "keyhole" seam (traversed there-and-back), leaving the hole a
    true void even in viewers that fill each closed polyline independently. Holes are spliced
    at their nearest vertex to the current ring, keeping the seam short.
    """
    ring = list(g.exterior.coords)[:-1]                       # drop the closing duplicate
    for interior in g.interiors:
        inn = list(interior.coords)[:-1]
        best = None                                           # nearest (ring vertex, hole vertex)
        for a, (ex, ey) in enumerate(ring):
            for b, (ix, iy) in enumerate(inn):
                dd = (ex - ix) ** 2 + (ey - iy) ** 2
                if best is None or dd < best[0]:
                    best = (dd, a, b)
        _, a, b = best
        inn_seq = inn[b:] + inn[:b] + [inn[b]]                # traverse the hole, start & end at b
        ring = ring[:a + 1] + inn_seq + ring[a:]              # splice in with the there-and-back seam
    ring.append(ring[0])
    return ring


def build_pad_columns(row_pitch, first_y, pitch, n_per_column):
    """
    Centres of the two pad columns ("rows"), each a vertical stack of n_per_column pads
    at `pitch` spacing starting at `first_y`. Returns (left, right) lists of (x, y),
    each sorted by ascending y. Index 0 (lowest) is the REF/GND pad; indices 1.. connect.
    """
    col_x = row_pitch / 2.0
    left = [(-col_x, first_y + i * pitch) for i in range(n_per_column)]
    right = [(col_x, first_y + i * pitch) for i in range(n_per_column)]
    return left, right


def assign_channels_to_pads(top_points):
    """
    channel index -> (column, pad_row) reproducing the crossing-free routing order.

    Channels are sorted by x; the lowest-x half serves the LEFT pad column, the
    highest-x half the RIGHT. Within the left column the outer lane (lowest x) peels to
    the LOWEST connected pad; within the right column the outer lane (highest x) peels to
    the lowest pad (hence the right half is reversed). pad_row is 1..N-1 (row 1 = lowest
    connected pad, nearest the REF/GND pad at index 0; the REF/GND pad is never assigned).

    This is the single source of truth for the ordering: build_pad_routes() consumes it to
    place each channel's diagonal into the right pad, and mapping.build_mapping() consumes
    it to attach flex-pad numbers. (It used to be duplicated in
    build_electrode_flex_mapping.derive_channel_to_pad.)
    """
    n = len(top_points)
    order = sorted(range(n), key=lambda i: top_points[i][0])   # channels by x ascending
    half = n // 2
    left_ch, right_ch = order[:half], order[half:]
    chan_pad = {}
    for k, ci in enumerate(left_ch):                # left column: outer lane -> lowest pad
        chan_pad[ci] = ('left', k + 1)
    for k, ci in enumerate(right_ch[::-1]):         # right column: reversed, outer lane -> lowest pad
        chan_pad[ci] = ('right', k + 1)
    return chan_pad


def build_pad_routes(top_points, pads_left, pads_right,
                     neck_tw, top_tw, neck_len, fan_len,
                     bundle_pitch, approach_deg):
    """
    Route every electrode wire up into its bond pad, keeping the traces as ONE compact
    bundle in the MIDDLE and fanning OUT to the side pad columns only near the top. Each
    channel = one closed polygon: a vertical NECK above the wire top, a short FAN that
    gathers the wire into a centred bundle lane, a vertical RISER straight up the middle,
    and a straight DIAGONAL that peels out into the pad. The diagonal meets the riser at
    `approach_deg` (interior bend angle), i.e. it climbs `approach_deg - 90` degrees above
    horizontal (140 deg -> a 50 deg climb). Width tapers neck_tw -> top_tw.

    Crossing-free by construction: channels are sorted by x into one centred bundle (so the
    middle is filled, no gap); the lowest 32 lanes serve the left column, the highest 32 the
    right (see assign_channels_to_pads). Within a side the OUTER lane (nearest its column)
    peels to the LOWEST pad and the inner lane to the highest, and every diagonal shares the
    same angle, so the parallel diagonals and the vertical risers never cross.

    Inputs:
        top_points (list[(x_src, y_top)]): one wire top per channel.
        pads_left, pads_right (list[(x, y)]): pad centres per column, ascending y, index 0
            = REF/GND (skipped); indices 1.. are the connected pads.
        neck_tw, top_tw, neck_len, fan_len: gather-fan geometry (micron).
        bundle_pitch (float): lane pitch of the centred bundle.
        approach_deg (float): interior bend angle of the diagonal into the pad.

    Returns:
        dict[int, list[(x, y)]]: per-channel closed polygon ring (None entries skipped).
    """
    n = len(top_points)
    order = sorted(range(n), key=lambda i: top_points[i][0])   # channels by x ascending
    mid = (n - 1) / 2.0
    lane_x = {ci: (pos - mid) * bundle_pitch for pos, ci in enumerate(order)}  # centred bundle
    slope = np.tan(np.radians(approach_deg - 90.0))            # diagonal climb above horizontal
    chan_pad = assign_channels_to_pads(top_points)             # shared ordering (single source of truth)
    pad_cols = {'left': pads_left, 'right': pads_right}

    polys = {}
    for ci in range(n):
        col, pad_row = chan_pad[ci]
        x_src, y_top = top_points[ci]
        x_lane = lane_x[ci]
        pad_x, y_pad = pad_cols[col][pad_row]                  # index 0 = REF/GND; pad_row 1.. connect
        y_neck = y_top + neck_len
        y_fan = y_neck + fan_len
        y_bend = y_pad - abs(pad_x - x_lane) * slope           # peel-off height of the diagonal
        centerline = [
            (x_src, y_top),    # joins the straight wire
            (x_src, y_neck),   # top of neck
            (x_lane, y_fan),   # gathered into the centred bundle lane
            (x_lane, y_bend),  # straight up the middle to the peel-off height
            (pad_x, y_pad),    # diagonal out into the pad centre
        ]
        widths = [neck_tw, neck_tw, top_tw, top_tw, top_tw]
        polys[ci] = stroke_centerline_to_polygon(centerline, widths)
    return polys


def channel_x_positions(n, delta_x, half_widths, wire_hw):
    """
    Signed x of each channel so the edge-to-edge GAP between neighbouring polyimide traces
    stays constant (= delta_x - 2*wire_hw) even when some channels are wider than wire_hw.

    Channels are laid out on alternating sides from the centre: odd i -> +slot, even i ->
    -slot, matching the original `bx = x_counter*delta_x` interleave. Each slot is placed by
    accumulating half-widths + gap outward from the centre, so:
      - with uniform widths it reproduces the original delta_x grid exactly,
      - a wider channel keeps its gaps by pushing only the channels FURTHER from the centre
        outward; every pitch that doesn't touch a wide channel is unchanged.

    The alternating layout gives one more slot on the +side than the -side for even n (e.g.
    64 -> slots -31..+32), so the raw bundle is off-centre by ~delta_x/2. A final shift
    recentres it so its outer EDGES are symmetric about x=0 -- this keeps the bundle aligned
    with the symmetric polyimide body and the +/-PAD_ROW_PITCH/2 pad columns, giving equal
    margins on both sides. (Channel 0 therefore lands slightly off x=0 for even n.)

    half_widths: dict {channel_index: polyimide-trace half-width}.
    """
    gap = delta_x - 2 * wire_hw
    slot_of = {i: (0 if i == 0 else ((i + 1) // 2 if i % 2 else -(i // 2))) for i in range(n)}
    chan_at = {s: i for i, s in slot_of.items()}
    pos = {0: 0.0}
    s = 1
    while s in chan_at:
        pos[s] = pos[s - 1] + half_widths[chan_at[s - 1]] + gap + half_widths[chan_at[s]]
        s += 1
    s = -1
    while s in chan_at:
        pos[s] = pos[s + 1] - (half_widths[chan_at[s + 1]] + gap + half_widths[chan_at[s]])
        s -= 1
    #Recentre on the bundle's outer edges so both margins are equal (see docstring).
    smax, smin = max(pos), min(pos)
    center = 0.5 * ((pos[smax] + half_widths[chan_at[smax]]) +
                    (pos[smin] - half_widths[chan_at[smin]]))
    return {i: pos[slot_of[i]] - center for i in range(n)}


#Hook template taken from the reference file (micron). The ~20um-wide stem is at the TOP
#(max y); the anchor/barb body hangs below it. create_hook_polygon() normalises this so the
#stem centre is at x=0 and the stem top at y=0, then places it at a given attach point with
#the body hanging toward -y (i.e. below the electrode trace it continues).
HOOK_TEMPLATE = np.array([
    (58.33000, 14.75800), (53.50000, 15.71800), (48.95000, 17.60300), (41.34100, 21.47300),
    (34.38000, 27.20600), (30.53000, 32.77500), (27.41800, 39.49100), (18.00000, 87.07200),
    (33.29200, 53.51700), (33.29200, 64.76100), (33.40500, 67.69200), (33.92200, 69.37300),
    (35.56000, 73.20900), (37.45700, 75.19100), (42.75800, 76.44100), (45.55900, 77.90600),
    (47.80000, 80.36300), (49.48100, 83.55200), (51.79200, 90.70700), (51.79200, 131.31300),
    (71.79200, 131.31300), (71.79200, 90.70700), (73.10300, 83.55200), (74.78400, 80.36300),
    (77.02500, 77.90600), (85.12700, 75.19100), (86.97200, 71.31200), (88.17900, 67.69200),
    (88.29200, 64.76100), (88.29200, 53.61200), (103.58400, 87.07200), (94.16600, 39.49100),
    (91.05400, 32.77500), (87.20400, 27.20600), (80.24300, 21.47300), (73.23900, 18.00800),
    (68.08400, 15.71800), (63.25400, 14.75800),
])
_HOOK_STEM_CX = 0.5 * (51.79200 + 71.79200)   # 61.792 -> stem centre, aligns with the trace
_HOOK_STEM_TOP = 131.31300                 # max y -> attach point 131.31300


def create_hook_polygon(cx, y_top, scale=1.0, drop=0.0):
    """
    The hook anchor as a closed (x, y) polygon, with its 20um stem centred at (cx, y_top)
    and the body hanging below (toward -y). `scale` scales the whole shape about the stem top.

    `drop` extends the hook further toward the bottom: the whole template is placed `drop`
    microns lower and a straight rectangular stem (the template's native ~20um stem width)
    is spliced back up to y_top, so the anchor reaches `drop` deeper without changing its
    shape. drop=0 reproduces the original polygon exactly. Returns (x_array, y_array).
    """
    t = HOOK_TEMPLATE.astype(float).copy()
    t[:, 0] = (t[:, 0] - _HOOK_STEM_CX) * scale + cx
    t[:, 1] = (t[:, 1] - _HOOK_STEM_TOP) * scale + (y_top - drop)
    x, y = list(t[:, 0]), list(t[:, 1])
    if drop > 0:
        # Splice a vertical riser up to y_top between the two stem-top corners (indices 19/20).
        x = x[:20] + [x[19], x[20]] + x[20:]
        y = y[:20] + [y_top,  y_top]  + y[20:]
    return np.asarray(x), np.asarray(y)


def create_rescue_hook_polygon(cx, y_top, angle_deg, arm_len, wire_end_width, circle_end_width,
                               circle_radius, drop=0.0, scale=1.0, resolution=48):
    """
    A 'lollipop' rescue anchor for the second channel -- deliberately NOT the barbed
    shape of create_hook_polygon. From the attach point (cx, y_top) the polyimide drops
    straight DOWN by `drop` (extending it toward the bottom), then a TAPERED neck heads
    off at angle_deg and ends in a plain CIRCLE of circle_radius. The neck STARTS at the
    wire with `wire_end_width` (pass the polyimide trace width above it for a seamless
    continuation) and tapers to `circle_end_width` where it meets the circle.

    `angle_deg` is the arm heading measured from the UPWARD wire direction (+y), opening
    toward +x (i.e. to the RIGHT): 0 -> straight up, 90 -> right, 155 -> down and to the
    right. `scale` scales arm_len, drop, the widths and circle_radius together.

    Returns ONE continuous closed (x, y) polygon (the neck boolean-unioned with the circle),
    just like create_hook_polygon, so it merges seamlessly with the polyimide trace it
    continues. Use rescue_hook_circle_centre() for the matching etch hole.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    arm_len *= scale
    drop *= scale
    r = circle_radius * scale
    th = np.radians(90.0 - angle_deg)               # heading: 0deg=+y, +ve toward +x
    d = np.array([np.cos(th), np.sin(th)])           # unit arm direction (base -> circle)
    base = np.array([float(cx), float(y_top) - drop])   # bottom of the vertical drop stem
    C = base + d * arm_len                           # circle centre
    C_edge = C - d * r                               # near edge of the circle

    # The VISIBLE taper widens from the wire (= the polyimide trace width) to circle_end_width
    # right at the circle's outer edge (C_edge), then continues at that width into the centre C
    # so the neck and circle truly overlap. The C_edge->C stretch is hidden inside the circle
    # after the union, so the taper appears to stop at the edge, not run toward the middle.
    centerline = [(cx, y_top), (base[0], base[1]), (C_edge[0], C_edge[1]), (C[0], C[1])]
    widths = [wire_end_width * scale, wire_end_width * scale,
              circle_end_width * scale, circle_end_width * scale]
    neck = stroke_centerline_to_polygon(centerline, widths)
    circle = create_polygon_circle(C[0], C[1], r, resolution)

    parts = [Polygon(circle[:-1]).buffer(0)]        # buffer(0) repairs any self-touching ring
    if neck is not None:
        parts.append(Polygon(neck).buffer(0))
    merged = unary_union(parts)
    if merged.geom_type == "MultiPolygon":          # keep the largest piece (defensive)
        merged = max(merged.geoms, key=lambda g: g.area)
    mx, my = merged.exterior.coords.xy
    return np.asarray(mx), np.asarray(my)


def rescue_hook_circle_centre(cx, y_top, angle_deg, arm_len, drop=0.0, scale=1.0):
    """Centre of the circular head of create_rescue_hook_polygon (for the etch hole)."""
    th = np.radians(90.0 - angle_deg)
    base_y = y_top - drop * scale
    return (cx + np.cos(th) * arm_len * scale, base_y + np.sin(th) * arm_len * scale)
