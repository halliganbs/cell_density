import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, color, io

# taken from this guide: https://github.com/bnsreenu/python_for_microscopists/blob/master/035-Cell%20Nuclei%20analysis%20using%20watershed.py
# use a test image first before automating

import tifffile
from PIL import Image

img = cv2.imread("test.tif", -1)
print(img.shape)
cells=img[:, :]

# threshold image to binary using OTSU, threshold set to 255
ret1, thresh = cv2.threshold(cells, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

from skimage.segmentation import clear_border
opening = clear_border(opening)

sure_bg = cv2.dilate(opening, kernel, iterations=10)

dist_transfrom = cv2.distanceTransform(opening, cv2.DIST_L2,5)

ret2, sure_fg = cv2.threshold(dist_transfrom, 0.3*dist_transfrom.max(), 255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

ret3, markers = cv2.connectedComponents(sure_fg)

markers = markers+10

markers[unknown==255] = 0

markers = cv2.watershed(img, markers)

img[markers == -1] = [0, 255, 255]

img = color.label2rgb(markers, bg_label=0)

cv2.imshow('Overlay on original image', img)
# cv2.imshow('Colored Grains', img2)
cv2.waitKey(0)