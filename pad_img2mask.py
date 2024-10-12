import cv2
import numpy as np
import matplotlib.pyplot as plt
import napari
from napari.utils import DirectLabelColormap

def detect_columns(fname):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the image
    image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    # Apply edge detection
    edges = cv2.Canny(image, threshold1=50, threshold2=150, apertureSize=3)

    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)

    # Draw the detected lines on the image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color drawing

    if lines is not None:
        for rho, theta in lines[:, 0]:
            # Filter out only vertical lines by checking the angle (theta)
            if abs(np.cos(theta)) < 0.1:  # near 90 degrees
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the result
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Vertical Lines')
    plt.axis('off')
    plt.show()

def detect_pads(fname):
    # Load the image
    image = cv2.imread(fname)
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
    print(len(circles), "circle centers detected")
    
    print('Update the circle centers using Napari\n\n')
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(blurred, name='Original Image', rgb=False)
        points_layer = viewer.add_points(circles, size=25, edge_color='green', 
                                    face_color='none', name='Circle Centers')
    circles = points_layer.data  # This contains the updated coordinates
    np.savetxt('circle_centers.csv', circles, delimiter=',', fmt='%d')
    
        
    sorted_indices = np.lexsort((circles[:, 0], circles[:, 1]))
    circles = circles[sorted_indices]
    
    # Create a colormap for the labeled pads
    colormap = plt.cm.get_cmap('hsv', len(circles))
    colors = [list((col[0]*255, col[1]*255, col[2]*255, 255)) 
              for col in colormap(np.linspace(0, 1, len(circles)))]
    
    for i, (y, x) in enumerate(circles):
        cv2.circle(canvas, (x, y), 21, colors[i], -1)
        # add text
        cv2.putText(canvas, f"{i}", (x-22,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0,255), 1)

    step = 56
    columns = np.array([-329.26126, -304.39027, -278.9902, -254.1193, -229.2484, -202.79005, -177.39001, -152.51914, -124.47328, -100.13146, -73.143943, -45.627259, -33.456418, -12.818905, 16.814435, 48.564455, 78.726974, 109.41871, 140.11042, 171.86044, 203.61046, 233.773, 265.52302, 296.74387, 333.25639, 995.73373, 970.86274, 945.46267, 920.59177, 895.72087, 869.26252, 843.86248, 818.99161, 790.94575, 766.60393, 739.61641, 712.09973, 699.92889, 679.29137, 649.65803, 617.90801, 587.7455, 557.05376, 526.36205, 494.61203, 462.86201, 432.69947, 400.94945, 369.7286])
    columns = np.sort(columns)
    columns += -columns.min()
    columns *= 3877/np.max(columns)
    columns *= .95
    columns = np.round(columns).astype(int)
    for col_i in columns:
        # print(col_i)
        canvas[:, col_i-1:col_i+1] = [0, 0, 0, 255]

    # Launch Napari viewer to visualize the image and the circle points
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(blurred, name='Original Image', rgb=False)
        viewer.add_image(canvas, name='Detected Circles', rgb=True)
        points_layer = viewer.add_points(circles, size=25, edge_color='green', 
                                        face_color='none', name='Circle Centers')

    cv2.imwrite('labeled_padlayout_um_res.png', canvas)

    # save the adjusted points to a CSV file
    # np.savetxt('circle_centers.csv', adjusted_points, delimiter=',', fmt='%d')

def display_labeled_pads(fname):
    # Load the image and downscale it by a factor of 9
    image = cv2.imread(fname)
    image = cv2.resize(image, (image.shape[1] // 9, image.shape[0] // 9))
    
    # Create a canvas to store the labeled pads
    canvas = np.zeros((image.shape[0], image.shape[1]), dtype=int)
    
    # Binarize the image
    _, image_bin = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Load the circle centers from the CSV file and downscale by 9
    centers = np.loadtxt('circle_centers_v2_um_res.csv', delimiter=',', dtype=float)
    circle_centers = np.round(centers / 9).astype(int)
    sorted_indices = np.lexsort((circle_centers[:, 1], circle_centers[:, 0]))
    circle_centers = circle_centers[sorted_indices]

    # Create a colormap for the labeled pads
    colormap = plt.cm.get_cmap('hsv', len(circle_centers))
    colors = [list((col[0]*255, col[1]*255, col[2]*255)) 
              for col in colormap(np.linspace(0, 1, len(circle_centers)))]
    cmap = DirectLabelColormap(color_dict=dict(zip(range(1, len(circle_centers) + 1), colors)))

    # Create the mask required by the floodFill operation
    mask = np.zeros((image_bin.shape[0] + 2, image_bin.shape[1] + 2), np.uint8)

    for i, (x, y) in enumerate(circle_centers):
        # Use flood fill to fill the circle on the binarized image
        cv2.floodFill(image_bin, mask, (y, x), newVal=colors[i])

        # # Add the label to the canvas for visualization purposes
        canvas[x, y] = i + 1

    # Launch Napari viewer and display the results
    with napari.gui_qt():
        viewer = napari.Viewer()

        viewer.add_image(image, name='Original Image', rgb=True)
        # viewer.add_image(image_bin, name='Binarized Image')
        viewer.add_labels(canvas, name='Labeled Pads', colormap=cmap)
        # viewer.add_points(centers / 9, size=1, edge_color='green', face_color='none', name='Circle Centers')
        viewer.add_points(circle_centers, size=1, edge_color='red', face_color='none', name='Circle Centers')

    # Save the labeled pads as a numpy array and image file
    print("Saving labeled pads...")
    # data = viewer.layers['Labeled Pads'].data
    # np.save('labeled_pads.npy', data)
    # cv2.imwrite('labeled_padlayout_el_res.png', image_bin)
    
    # append alpha dimension to the image using numpy only
    final_lbls = np.zeros((image_bin.shape[0], image_bin.shape[1], 4), dtype=np.uint8)
    final_lbls[..., :3] = image_bin
    final_lbls[..., 3] = 255
    
    # Replace white pixels with transparent
    # Create a mask for white pixels
    white_mask = (final_lbls[..., 0] == 255) & (final_lbls[..., 1] == 255) & (final_lbls[..., 2] == 255)
    # Set the alpha channel of the white pixels to 0 (transparent)
    final_lbls[white_mask] = [0, 0, 0, 0]  # Set RGBA values to transparent

    # save the image
    cv2.imwrite('labeled_padlayout_um_res.png', final_lbls)
    # numpy save
    np.save('labeled_pads_um_res.npy', canvas)

    
# fname = './waferpic2_el_res.png'
fname = './waferpic2_um_res.png'
# detect_pads(fname)
fname = 'columns.png'
detect_columns(fname)
# display_labeled_pads(fname)