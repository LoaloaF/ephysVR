"""Pure geometry primitives (numpy only -- no ezdxf, no shapely, no matplotlib).

Every function here returns raw point data (either an (x, y) tuple of arrays or an
Nx2 array) and holds no drawing or file state, so it can be unit-tested in isolation
and reused by the domain shapes in ``shapes.py``.

Moved verbatim from hook_bundle_generator.py; ``create_polygon_circle`` also replaces
the duplicate that used to live in gen_ionp_bundle.py.
"""
import numpy as np


def convert_rectangle_to_polyline(rectangle):
    """Close an (x, y) outline into an Nx2 array whose last point repeats the first."""
    points = np.asarray(tuple(zip(rectangle[0], rectangle[1])))
    first_point = np.expand_dims(points[0], 0)
    points = np.concatenate((points, first_point))
    return points


def create_rectangle(x0, y0, w, l):
    x = [x0, x0 + w, x0 + w, x0]
    y = [y0, y0, y0 + l, y0 + l]
    return (x, y)


def create_triangle(x0, y0, w, l):
    x = [x0, x0 + w, x0 + w / 2]
    y = [y0, y0, y0 + l]
    return (x, y)


def create_rectangle_with_tip(x0, y0, w, depth, tip):
    """
    A rectangle of width w starting at (x0,y0) and extending by `depth` in y,
    terminated by a triangular tip extending a further `tip` in y. Returns a
    single closed polygon outline (x,y) so there is no internal seam between
    the rectangle and the triangle.
    """
    x = [x0,   x0 + w, x0 + w,        x0 + w / 2,          x0]
    y = [y0,   y0,     y0 + depth,    y0 + depth + tip,    y0 + depth]
    return (x, y)


def stroke_centerline_to_polygon(centerline, widths):
    """
    Turn a centerline (list of (x, y)) with a per-vertex full trace width into a
    CLOSED polygon outline (list of (x, y)) using miter joins. Ported from
    route_append_to_interconnect_12blocks_actualpolygons.py so the fanout can taper
    its width continuously along the fan. Returns None for degenerate input.
    """
    pts = np.asarray(centerline, dtype=float)
    w = np.asarray(widths, dtype=float)
    # Drop consecutive duplicate vertices.
    keep = [0]
    for i in range(1, len(pts)):
        if not np.allclose(pts[i], pts[keep[-1]]):
            keep.append(i)
    pts, w = pts[keep], w[keep]
    n = len(pts)
    if n < 2:
        return None
    seg = pts[1:] - pts[:-1]
    d = seg / np.hypot(seg[:, 0], seg[:, 1])[:, None]     # unit segment directions
    seg_n = np.stack([-d[:, 1], d[:, 0]], axis=1)         # left normals per segment
    left = np.empty((n, 2))
    right = np.empty((n, 2))
    for i in range(n):
        h = w[i] / 2.0
        if i == 0:
            off = seg_n[0] * h
        elif i == n - 1:
            off = seg_n[-1] * h
        else:
            m = seg_n[i - 1] + seg_n[i]                   # miter (bisector) direction
            ml = np.hypot(m[0], m[1])
            if ml < 1e-9:                                  # ~180 deg reversal: fall back
                off = seg_n[i] * h
            else:
                m = m / ml
                cos = max(float(np.dot(m, seg_n[i])), 0.25)   # clamp sharp miters
                off = m * (h / cos)
        left[i] = pts[i] + off
        right[i] = pts[i] - off
    ring = np.vstack([left, right[::-1]])                 # forward left, backward right
    return [(float(x), float(y)) for x, y in ring]


def create_polygon_circle(center_x: float, center_y: float, radius: float,
                          resolution: int = 16) -> np.ndarray:
    """Generate points for a circular polyline with adjustable resolution.

    Args:
        center_x: X coordinate of circle center
        center_y: Y coordinate of circle center
        radius: Distance from center to edge
        resolution: Number of vertices (min 3)

    Returns:
        Array of shape (resolution+1, 2) with x,y coordinates
    """
    resolution = max(3, resolution)
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    points = np.column_stack((x, y))
    return np.vstack((points, points[0]))


def _arc(cx, cy, r, a0, a1, n=10):
    """Sample a circular arc (n points) from angle a0 to a1 (radians) about (cx,cy)."""
    angles = np.linspace(a0, a1, n)
    return list(cx + r * np.cos(angles)), list(cy + r * np.sin(angles))


def create_ellipse_polygon(center_x, center_y, rx, ry, resolution=48):
    """Closed elliptical polyline (resolution+1 points) centred at (center_x, center_y)."""
    ang = np.linspace(0, 2 * np.pi, max(3, resolution), endpoint=False)
    pts = np.column_stack((center_x + rx * np.cos(ang), center_y + ry * np.sin(ang)))
    return np.vstack((pts, pts[0]))


def polygon_centroid(x, y):
    """Area-weighted centroid (shoelace) of a closed polygon given as x, y arrays."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x1 = np.roll(x, -1); y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return float(x.mean()), float(y.mean())
    cx = ((x + x1) * cross).sum() / (6.0 * a)
    cy = ((y + y1) * cross).sum() / (6.0 * a)
    return float(cx), float(cy)
