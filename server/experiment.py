import numpy as np
import cv2
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
 
img = cv2.imread("labimage.jpeg")
img = cv2.resize(img, (640, 640))
cv2.imshow("myimage", img)
cv2.waitKey()