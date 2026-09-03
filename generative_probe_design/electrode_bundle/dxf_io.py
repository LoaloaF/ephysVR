"""DXF read/write helpers -- the only module that touches ezdxf documents.

Readers (extract_*) are ported from hook_bundle_generator.py and gen_ionp_bundle.py;
`ensure_layer` / `add_ring` are thin writer wrappers factored out of the old __main__
so bundle.py reads as orchestration rather than ezdxf bookkeeping.
"""
import numpy as np

from .geometry import create_polygon_circle


# ----------------------------------------------------------------- readers
def entity_xy_points(e):
    """
    Return [(x, y), ...] for a polyline entity, handling both DXF encodings:
    - LWPOLYLINE: coordinates live in e.get_points()
    - POLYLINE (old R12/AC1009 files, e.g. the *_compatible_with_FPC.dxf):
      coordinates live in e.vertices via v.dxf.location.x/.y
    Returns [] for any other entity type.
    """
    if e.dxftype() == "LWPOLYLINE":
        return [(float(p[0]), float(p[1])) for p in e.get_points()]
    if e.dxftype() == "POLYLINE":
        return [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in e.vertices]
    return []


def extract_from_dxf(path, layer=None):
    """
    Read a DXF and return a list of Nx2 point arrays, one per polyline entity.
    If `layer` is given, keep only entities on that layer. Coordinates are
    returned in the file's own units (the FPC reference DXFs are already micron,
    the same units this generator uses, so no scaling is needed).
    """
    import ezdxf
    doc = ezdxf.readfile(path)
    polys = []
    for e in doc.modelspace():
        if layer is not None and e.dxf.layer != layer:
            continue
        pts = entity_xy_points(e)
        if len(pts) >= 2:
            polys.append(np.asarray(pts))
    return polys


def extract_centroids_from_dxf(path, layer=None):
    """
    Per-entity centroids (Nx2, micron), sorted by ascending y so the channel
    order is stable. Works on the POLYLINE-based reference files too.
    """
    polys = extract_from_dxf(path, layer)
    centroids = np.array([p.mean(axis=0) for p in polys])
    if len(centroids):
        centroids = centroids[np.argsort(centroids[:, 1])]
    return centroids


def extract_circle_centroids(path, layer='metalization', diam_range=(150, 300),
                             round_tol=0.15, dedupe_tol=1.0):
    """
    Centroids (Nx2, micron, sorted by ascending y) of round polylines on `layer`
    whose bounding-box diameter falls in diam_range. Defaults target the ~200 um,
    300 um-pitch bond pads (the "top circles") on the metalization layer -- these
    sit high up in y, well above the 11 um electrode contacts.

    A diameter + roundness filter is all that's needed: non-round wide traces on
    the same layer are rejected by round_tol, and the small electrode contacts by
    diam_range, so no y-cutoff is required. Near-coincident duplicates (the HD
    file has overlapping pad polylines) are merged within dedupe_tol microns.
    """
    cents = []
    for p in extract_from_dxf(path, layer):          # reuse the existing reader
        w = p[:, 0].max() - p[:, 0].min()
        h = p[:, 1].max() - p[:, 1].min()
        if w <= 0 or h <= 0:
            continue
        if abs(w - h) / max(w, h) > round_tol:        # roughly round (drops traces/squares)
            continue
        d = 0.5 * (w + h)
        if diam_range[0] <= d <= diam_range[1]:        # ~200 um pads only
            cents.append(p.mean(axis=0))
    cents = np.array(cents)
    if not len(cents):
        return cents
    cents = cents[np.argsort(cents[:, 1])]
    # dedupe near-coincident centroids (HD file has duplicate pad polylines)
    keep = [cents[0]]
    for c in cents[1:]:
        if np.min(np.linalg.norm(np.array(keep) - c, axis=1)) > dedupe_tol:
            keep.append(c)
    return np.array(keep)


def extract_electrode_centroids_from_msp(msp, el_layer):
    """
    Reduce each electrode polyline on `el_layer` (an already-open modelspace) to its
    centroid. Returns an Nx2 array in modelspace order (no sort). This is the
    gen_ionp_bundle reader with the matplotlib side effect removed -- the caller renders.
    """
    el_locs = []
    for e in msp:
        if e.dxftype() == 'LWPOLYLINE' and e.dxf.layer == el_layer:
            pts = np.array([(x, y) for x, y, *_ in e.get_points()])
            el_locs.append(pts.mean(axis=0))
    return np.stack(el_locs) if el_locs else np.empty((0, 2))


# ----------------------------------------------------------------- writers
def ensure_layer(doc, name, color=7, linetype='CONTINUOUS'):
    """Create `name` on `doc` if it does not already exist. Returns the layer name."""
    if name not in doc.layers:
        doc.layers.new(name=name, dxfattribs={'linetype': linetype, 'color': color})
    return name


def add_ring(msp, ring, layer):
    """
    Emit one closed LWPOLYLINE on `layer`. `ring` may be an Nx2 array or a list of
    (x, y) tuples; the entity is flagged closed so the first/last point need not repeat.
    """
    pts = [(float(x), float(y)) for x, y in ring]
    msp.add_lwpolyline(pts, close=True, dxfattribs={'layer': layer})


def write_wells_into_dxf(doc, msp, xs, ys, well_d, layer_name, resolution):
    """Idempotently add well circles onto `layer_name` in the open document."""
    # remove any wells from a previous run
    stale = [e for e in msp if e.dxf.layer == layer_name]
    for e in stale:
        msp.delete_entity(e)
    if stale:
        print(f"Removed {len(stale)} pre-existing wells on '{layer_name}'")

    ensure_layer(doc, layer_name, color=7)

    for x, y in zip(xs, ys):
        circ = create_polygon_circle(x, y, well_d / 2, resolution=resolution)
        msp.add_lwpolyline(circ, close=True, dxfattribs={'layer': layer_name})
    print(f"Added {len(xs)} IONP wells to layer '{layer_name}'")
