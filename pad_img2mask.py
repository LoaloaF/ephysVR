import cv2
import numpy as np
import matplotlib.pyplot as plt
import napari
from napari.utils import DirectLabelColormap

import pandas as pd

def detect_columns(fname):
    image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    row_mask = image[0,:] != 255
    print("Found n columns: ", row_mask.sum())
    return np.where(row_mask)[0]

def detect_pads(fname_base):
    # Load the image
    path = './assets'
    image = cv2.imread(f"{path}/waferpic_{fname_base}.png")
    canvas = np.zeros((*image.shape[:2],4), dtype=np.uint8)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use the Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(blurred, 
                            cv2.HOUGH_GRADIENT, 
                            dp=1.4, 
                            minDist=65, 
                            param1=50, 
                            param2=20, 
                            minRadius=17, 
                            maxRadius=33)

    circles = np.round(circles[0, :]).astype("int")
    circles = np.array([(y, x) for (x, y, r) in circles])
    print("\n", len(circles), "circle centers detected")
    
    # load the circle centers from the CSV file, adjusted ones
    circles_fname = f"{path}/circle_centers_{fname_base}.csv"
    circles = np.loadtxt(circles_fname, delimiter=',', dtype=int)
    
    print('Update the circle centers using Napari\n\n')
    viewer = napari.Viewer()
    viewer.add_image(blurred, name='Original Image', rgb=False)
    points_layer = viewer.add_points(circles, size=25, border_color='green', 
                                face_color='white', name='Circle Centers')
    napari.run()
    
    # updated and save
    circles = points_layer.data  # This contains the updated coordinates
    np.savetxt(circles_fname.replace('.csv', '_new.csv'), circles, delimiter=',', fmt='%d')
    
    # sort the circles by column
    columns = detect_columns(f"{path}/columns_{fname_base}.png")
    columns = np.array([0, *columns, image.shape[1]])
    
    circles_sorted = []
    for from_x, to_x in zip(columns[:-1], columns[1:]):
        circles_in_col = circles[(circles[:, 1] > from_x) & (circles[:, 1] < to_x)]
        sorted_indices = np.argsort(circles_in_col[:, 0])
        circles_sorted.extend(circles_in_col[sorted_indices])
    circles = np.array(circles_sorted)
    
    # Create a colormap for the labeled pads
    colormap = plt.get_cmap('hsv', len(circles))
    colors = [list((col[0]*255, col[1]*255, col[2]*255, 255)) 
              for col in colormap(np.linspace(0, 1, len(circles)))]
    colors_df = pd.DataFrame(np.array(colors, dtype=int)[:,:3], columns=['r', 'g', 'b'], 
                             index=np.arange(len(circles)))
    colors_df.to_csv(f"{path}/padcolors_{fname_base}.csv", index=False)    
    
    for i, (y, x) in enumerate(circles):
        cv2.circle(canvas, (x, y), 26, colors[i], -1)
        # labelled version
        # cv2.circle(canvas, (x, y), 26, colors[i], 2)
        # offset = 14 if i < 1000 else 19
        # cv2.putText(canvas, f"{i}", (x-offset,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0,255), 1)

    # add vertical lines to separate columns
    for col_i in columns:
        blurred[:, col_i-1:col_i+1] = 0

    # Launch Napari viewer to visualize the image and the circle points
    viewer = napari.Viewer()
    viewer.add_image(blurred, name='Original Image', rgb=False, colormap='gray_r')
    viewer.add_image(canvas, name='Detected Circles', rgb=True)
    napari.run()
    
    cv2.imwrite(f"{path}/padlayout_{fname_base}.png", canvas)
    
    
fname_base = '46pads'
detect_pads(fname_base)
fname_base = '48pads'
detect_pads(fname_base)