import cv2
from object_detector import *
import numpy as np

#Load Aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)


#Load Object Detector
detector = HomogeneousBgDetector()

#Load Image
img = cv2.imread("items_aruco_marker.jpg")
if img is None:
    raise ValueError("Image not found or unable to read the image file.")

# Resize image to fit the screen while maintaining aspect ratio
max_width = 800
max_height = 600
(h, w) = img.shape[:2]
scaling_factor = min(max_width / w, max_height / h)
new_size = (int(w * scaling_factor), int(h * scaling_factor))
resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

#Get Aruco marker
corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

#Draw polygon around the marker
int_corners = np.int32(corners)
print(int_corners)
cv2.polylines(img, int_corners, True, (0, 255, 0), 5)


#Aruco Perimeter
aruco_perimeter = cv2.arcLength(corners[0], True)

#Pixel to cm ratio
pixel_cm_ratio = aruco_perimeter / 20

contours = detector.detect_objects(img)

#Draw objects boundaries
for cnt in contours:

    #Get rect
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect

    #Get Width and Height of the Objects by applying the Ration pixel to cm
    object_width = w / pixel_cm_ratio
    object_height = h / pixel_cm_ratio

    # Determine which dimension is greater
    if object_width > object_height:
        length = object_width
        width = object_height
    else:
        length = object_height
        width = object_width

    #Display rectangle
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(img, [box], True, (255, 0, 0), 3)
    cv2.putText(img, "Length {} cm".format(round(length, 2)), (int(x - 140), int(y-50)), cv2.FONT_HERSHEY_PLAIN, 2.2, (100, 200, 0), 4)
    cv2.putText(img, "Width {} cm".format(round(width, 2)), (int(x - 140), int(y+50)), cv2.FONT_HERSHEY_PLAIN, 2.2, (100, 200, 0), 4)

# Create a resizable window
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", min(new_size[0], max_width), min(new_size[1], max_height))

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()