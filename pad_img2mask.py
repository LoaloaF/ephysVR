import cv2
import numpy as np
import matplotlib.pyplot as plt
import napari
from napari.utils import DirectLabelColormap


def detect_pads():
    # Load the image
    image = cv2.imread('./waferpic2.png')
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use the Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(blurred, 
                            cv2.HOUGH_GRADIENT, 
                            dp=1.2, 
                            minDist=26, 
                            param1=50, 
                            param2=20, 
                            minRadius=9, 
                            maxRadius=15)

    # Convert circles to a list of (x, y) coordinates if any were found
    points = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        points = [(y, x) for (x, y, r) in circles]

    # Launch Napari viewer to visualize the image and the circle points
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(blurred, name='Original Image', rgb=False)
        points_layer = viewer.add_points(points, size=15, edge_color='green', 
                                        face_color='none', name='Circle Centers')

    # Once the GUI is closed, you can access the modified points using:
    adjusted_points = points_layer.data  # This contains the updated coordinates
    print("Adjusted points:", adjusted_points)

    # save the adjusted points to a CSV file
    np.savetxt('circle_centers.csv', adjusted_points, delimiter=',', fmt='%d')

def display_labeled_pads():
    # pad image
    image = cv2.imread('./waferpic2.png')
    # downscale by 10
    image = cv2.resize(image, (220, 120))    
    
    # read the circle centers from the CSV file
    centers = np.loadtxt('circle_centers.csv', delimiter=',', dtype=float)
    canvas = np.zeros((120,220), dtype=int)
    
    circle_centers = np.round(centers/10).astype(int)
    sorted_indices = np.lexsort((circle_centers[:, 1], circle_centers[:, 0]))
    circle_centers = circle_centers[sorted_indices]
    
    # create a colormap from 1 to the number of circles, use viridis and sample the colors
    colormap = plt.cm.get_cmap('hsv', len(circle_centers))
    colors = [list(col) for col in colormap(np.linspace(0, 1, len(circle_centers)))]
    cmap = DirectLabelColormap(color_dict=dict(zip(range(1, len(circle_centers)+1), colors)))
    
    for i, (x, y) in enumerate(circle_centers):
        # make a little cross around the center
        canvas[x-1:x+2, y-1:y+2] = i+1
        canvas[x-1:x+2, y-1:y+2] = i+1
        
    # canvas = np.load('labeled_pads_v1.npy')
    
    # Launch Napari viewer and load each point as a segmentaiton layer where the value encodes the pad number
    with napari.gui_qt():
        viewer = napari.Viewer()
        
        viewer.add_image(image, name='Original Image', rgb=True)
        
        viewer.add_labels(canvas, name='Labeled Pads', colormap=cmap)
        viewer.add_points(centers/10, size=1, edge_color='green', 
                          face_color='none', name='Circle Centers')
        viewer.add_points(np.round(centers/10), size=1, edge_color='red', 
                          face_color='none', name='Circle Centers')


    print("Saving labeled pads...")    
    # Once the GUI is closed, save the labeled pads as an image file /numpy array
    data = viewer.layers['Labeled Pads'].data
    print(data)
    np.save('labeled_pads.npy', data)
    cv2.imwrite('labeled_pads.png', data)
    

# detect_pads()
display_labeled_pads()