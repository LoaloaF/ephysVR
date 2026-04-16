# together with Claude 3.5 
import json
import ezdxf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def create_polygon_circle(
    center_x: float, 
    center_y: float, 
    radius: float, 
    resolution: int = 16
) -> np.ndarray:
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

def filter_to_xlocs_in_pattern(
    xlocs: np.ndarray,
    pattern: np.ndarray, 
    y_top: float,
    y_bottom: float
) -> np.ndarray:
    """Filter x locations based on pattern intervals."""
    mask = np.zeros_like(xlocs, dtype=bool)
    yrange = y_top - y_bottom
    
    for start, end in pattern:
        y_start = start * yrange + y_bottom
        y_end = end * yrange + y_bottom
        interval_mask = (xlocs >= y_start) & (xlocs <= y_end)
        mask = mask | interval_mask
        
    return xlocs[mask]

def extract_electrodes_from_dxf(
    input_fname: str,
    el_layer: str = "Electrodes"
) -> np.ndarray:
    """Extract electrode locations from DXF file."""
    doc = ezdxf.readfile(input_fname)
    msp = doc.modelspace()

    fig, ax = plt.subplots(figsize=(10, 10))
    el_locs = []

    for entity in msp:
        if entity.dxftype() == "LWPOLYLINE" and entity.dxf.layer == el_layer:
            points = np.array([((x), (y)) for x, y, *_ in entity.get_points()])
            el_center = points.mean(axis=0)
            
            ax.plot(*zip(*points), color='black', linewidth=0.5, alpha=0.5)
            ax.scatter(*el_center, color='red', s=60, alpha=0.2)
            el_locs.append(el_center)

    el_locs = np.stack(el_locs)
    print(f"Found {len(el_locs)} electrodes")
    return el_locs

def calc_ionp_well_locs(
    el_locs: np.ndarray,
    pattern: np.ndarray,
    y_top: float,
    y_bottom: float,
    well_distance: float,
    first_well_distance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate IONP well locations based on electrode positions."""
    ionp_xlocs = []
    ionp_ylocs = []

    for el_x, el_y in el_locs:
        ionp_Y = np.arange(el_y + first_well_distance, y_top, well_distance)
        ionp_Y = filter_to_xlocs_in_pattern(ionp_Y, pattern, y_top, y_bottom)
        ionp_X = np.full_like(ionp_Y, el_x)
        
        plt.scatter(ionp_X, ionp_Y, color='blue', s=20, alpha=0.5)
        ionp_xlocs.append(ionp_X)
        ionp_ylocs.append(ionp_Y)

    return np.concatenate(ionp_xlocs), np.concatenate(ionp_ylocs)

def check_design_rules(
    pattern: np.ndarray,
    y_top: float,
    y_bottom: float,
    box_width: float,
    ionp_xlocs: np.ndarray, 
    ionp_ylocs: np.ndarray,
) -> None:
    yrange = y_top - y_bottom

    
    for i, (start, end) in enumerate(pattern):
        y_start = start * yrange + y_bottom
        y_end = end * yrange + y_bottom
        print(f"----\nProcessing strip (bottom2top) {i}:")
        
        if i > 0 and y_start - prev_y_end < MIN_STRIP_DISTANCE:
            print(f"WARNING: Strip {i} is too close to strip {i-1} (dist.: {int(y_start-prev_y_end )})um")
        
        # check how many wells are in this strip
        mask = (ionp_ylocs >= y_start) & (ionp_ylocs <= y_end)
        print(f"{np.sum(mask):05} IONP wells in this strip")
        
        if np.sum(mask) < MIN_N_WELLS or np.sum(mask) > MAX_N_WELLS:
            print(f"WARNING: Strip {i} has {np.sum(mask)} wells, which is outside the recommended range ({MIN_N_WELLS}-{MAX_N_WELLS})")        
        
        x_in_strip = ionp_xlocs[mask]
        y_in_strip = ionp_ylocs[mask]
        plt.scatter(x_in_strip, y_in_strip, s=20, alpha=0.5)
        prev_y_end = y_end
        print("----\n")
        
def add_ionp_to_dxf(
    input_fname: str,
    output_fname: str,
    ionp_xlocs: np.ndarray,
    ionp_ylocs: np.ndarray,
    well_diameter: float,
    layer_name: str = "pattern_ionp"
) -> None:
    """Add IONP wells and boxes to DXF file."""
    doc = ezdxf.readfile(input_fname)
    msp = doc.modelspace()
    
    if layer_name not in doc.layers:
        doc.layers.new(name=layer_name, dxfattribs={'color': 7})

    # Add wells
    for x, y in zip(ionp_xlocs, ionp_ylocs):
        circle = create_polygon_circle(x, y, well_diameter/2, resolution=12)
        msp.add_lwpolyline(circle, close=True, dxfattribs={'layer': layer_name})

    print(f"Added {len(ionp_xlocs)} IONP wells to layer '{layer_name}'")
    doc.saveas(output_fname)

# write the ionp pattern to a standalone dxf file
def write_ionp_pattern_to_dxf(
    ionp_xlocs: np.ndarray,
    ionp_ylocs: np.ndarray,
    well_diameter: float,
    output_fname: str,
    layer_name: str = "pattern_ionp",
    scale_factor: float = 1.0
) -> None:
    """Write IONP pattern to a DXF file.
    
    Args:
        ionp_xlocs: X coordinates of IONP wells
        ionp_ylocs: Y coordinates of IONP wells
        well_diameter: Diameter of wells
        output_fname: Output filename
        layer_name: Layer name for the wells
        scale_factor: Factor to scale all coordinates and dimensions (default: 1.0)
    """
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    if layer_name not in doc.layers:
        doc.layers.new(name=layer_name, dxfattribs={'color': 7})

    # Scale coordinates and diameter
    scaled_xlocs = ionp_xlocs * scale_factor
    scaled_ylocs = ionp_ylocs * scale_factor
    scaled_diameter = well_diameter * scale_factor

    # Add wells
    for x, y in zip(scaled_xlocs, scaled_ylocs):
        circle = create_polygon_circle(x, y, scaled_diameter/2, resolution=12)
        msp.add_lwpolyline(circle, close=True, dxfattribs={'layer': layer_name})

    print(f"Added {len(ionp_xlocs)} IONP wells to layer '{layer_name}' with scale factor {scale_factor}")
    doc.saveas(output_fname)

def write_pattern_summary_to_json(
    pattern: np.ndarray,
    y_top: float,
    y_bottom: float,
    el_locs: np.ndarray,
    ionp_xlocs: np.ndarray,
    ionp_ylocs: np.ndarray,
    output_fname: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write strip-level pattern information to a JSON file.

    The output includes, for each strip:
    - normalized coordinates
    - real Y coordinates
    - number of wells in the strip
    """
    yrange = y_top - y_bottom
    strips: Dict[str, Dict[str, Any]] = {}

    for i, (start, end) in enumerate(pattern):
        y_start = start * yrange + y_bottom
        y_end = end * yrange + y_bottom
        mask = (ionp_ylocs >= y_start) & (ionp_ylocs <= y_end)

        strips[str(i)] = {
            "name": f"ionp_pattern_{i}",
            "normalized": {
                "start": float(start),
                "end": float(end),
            },
            "real_y_um": {
                "start": float(y_start),
                "end": float(y_end),
                "height": float(y_end - y_start),
            },
            "n_wells": int(np.sum(mask)),
        }

    summary: Dict[str, Any] = {
        "electrodes": {
            "n_electrodes": int(len(el_locs)),
            "locations_um": [
                {"x": float(x), "y": float(y)}
                for x, y in el_locs
            ],
        },
        "ionp_pattern": {
            "name": f"ionp_pattern_{metadata['ionp_pattern_i']}" if metadata is not None and "ionp_pattern_i" in metadata else None,
            "n_strips": int(len(pattern)),
            "total_wells": int(len(ionp_xlocs)),
            "normalized_range": {
                "bottom": 0.0,
                "top": 1.0,
            },
            "real_y_range_um": {
                "bottom": float(y_bottom),
                "top": float(y_top),
                "height": float(yrange),
            },
            "strips": strips,
        },
    }

    if metadata is not None:
        summary["metadata"] = metadata

    output_path = Path(output_fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote pattern summary JSON: {output_path}")
    return summary
    
# Example usage
if __name__ == "__main__":
    # define patterns in normalized coordinates, from 0 budle bottom, to 1 
    # last electrode at top. In these intervals, IONP wells will be placed.
    
    MIN_N_WELLS = 700
    MAX_N_WELLS = 10_000
    MIN_STRIP_DISTANCE = 300
    
    PATTERNS = {
        0: np.array([[0, .15], [.25, .32], [.40, .45], [.57, .61], [.8, .82], [.97, 1]]),      # for 96 electrodes
        1: np.array([[0, 0.15], [.24, .4], [.48, .65], [.86,.87], [.94, 1]]),  # for 96 electrodes
        2: np.array([[0, .15], [.25+.1, .32+.06], [.40+.12, .45+.1], [.61, .65], [.72, .77], [.85,.9], [.97, 1]]),      # for 96 electrodes
        3: np.array([[0, 0.15], [.24, .29], [.36, .41], [.47, .51], [.57, .72], [.86,.87], [.94, 1]]),  # for 96 electrodes
        
        4: np.array([[0, 0.15], [.24, .29], [.36, .41], [.47, .51], [.57, .64], [.8,.815], [.885,.9], [.96, 1]]),  # for 128 electrodes, adopted 3
        5: np.array([[0, 0.15], [.24, .26], [.31, .34], [.41, .43], [.5, .52], [.58,.6], [.66,.68], [.75, .76], [.82,.85], [.96, 1]]),  # for 128 electrodes, adopted 3
        # 4: np.array([[0.1, 0.2], [.3, .4], [.6, .65], [.85, 1]]),  # for 96 electrodes and 128 
        # 5: np.array([[0, 0.1], [.2, .25], [.4, .45], [.65, .75], [.8, .9]]),  # for 96 electrodes and 128 
    }

    CONFIG = {
        # electrode top y coordinate in the DXF file
        # 'y_top': 5424, # 96 electrodes
        'y_top': 8320, # 128 electrodes
        # electrode bottom y coordinate in the DXF file
        'y_bottom': -85,
        # width of the IONP box around the wells
        # 'box_width': 2500, # for 96 electrodes
        'box_width': 3000, # for 128 electrodes
        # just used for defining the filename here
        # 'n_els': 96,
        'n_els': 128,
        # which pattern to use from definitions above
        'pattern_i': 5,
        # diameter of the IONP wells
        'ionp_well_d': 1.5,
        # distance between IONP wells
        'ionp_well_distance': 5,
        # distance of the first IONP well from the electrode
        'ionp_firstwell_distance': 15,
        # where are the electrodes in the DXF input file
        'el_layer': "Electrodes"
    }

    # file names
    path = '/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/devices/electrode_devices/S0844pad6shank/shanks/'
    # input_basename = '96.dxf' if CONFIG['n_els'] == 96 else f"{CONFIG['n_els']}ch_shank.dxf"
    input_basename = f"{CONFIG['n_els']}.dxf"
    input_fname = f"{path}{input_basename}"
    output_stem = f"{CONFIG['n_els']}ch_ionp_{CONFIG['pattern_i']}"
    output_fname = f"{path}{output_stem}.dxf"
    summary_fname = f"{path}{output_stem}_info.json"

    # Main workflow
    el_locs = extract_electrodes_from_dxf(input_fname, CONFIG['el_layer'])
    
    ionp_xlocs, ionp_ylocs = calc_ionp_well_locs(
        el_locs, 
        PATTERNS[CONFIG['pattern_i']], 
        CONFIG['y_top'],
        CONFIG['y_bottom'],
        CONFIG['ionp_well_distance'],
        CONFIG['ionp_firstwell_distance']
    )

    check_design_rules(
        PATTERNS[CONFIG['pattern_i']],
        CONFIG['y_top'],
        CONFIG['y_bottom'],
        CONFIG['box_width'],
        ionp_xlocs, 
        ionp_ylocs
    )

    # add_ionp_to_dxf(
    #     input_fname,
    #     output_fname,
    #     ionp_xlocs,
    #     ionp_ylocs,
    #     CONFIG['ionp_well_d'],
    #     layer_name=f"pattern{CONFIG['pattern_i']}_ionp"
    # )
    
    # write_ionp_pattern_to_dxf(
    #     ionp_xlocs,
    #     ionp_ylocs,
    #     CONFIG['ionp_well_d'],
    #     output_fname,
    #     layer_name=f"pattern{CONFIG['pattern_i']}_ionp",
    #     scale_factor=.001  # Default scaling, change as needed
    # )

    write_pattern_summary_to_json(
        PATTERNS[CONFIG['pattern_i']],
        CONFIG['y_top'],
        CONFIG['y_bottom'],
        el_locs,
        ionp_xlocs,
        ionp_ylocs,
        summary_fname,
        metadata={
            'ionp_pattern_i': int(CONFIG['pattern_i']),
            'ionp_well_d_um': float(CONFIG['ionp_well_d']),
            'ionp_well_distance_um': float(CONFIG['ionp_well_distance']),
            'ionp_firstwell_distance_um': float(CONFIG['ionp_firstwell_distance']),
        },
    )
    

    plt.axis('equal')
    plt.title("Pattern " + str(CONFIG['pattern_i']) + f", {CONFIG['n_els']} electrodes")
    # save the figure for documentation
    plt.savefig(f"{path}{output_stem}.png", dpi=300)
    plt.show()