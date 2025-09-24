import colorsys

def adjust_saturation(rgb_color, offset):
    # Convert RGB (0-1 scale) to HSV
    r, g, b = rgb_color
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    # Adjust saturation by offset, ensuring it stays within [0, 1]
    s = max(0.0, min(1.0, s - offset))
    # Convert back to RGB
    return colorsys.hsv_to_rgb(h, s, v)