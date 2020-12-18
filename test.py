import cv2
import numpy as np

from PIL import Image
import tifffile

import matplotlib.pyplot as plt

path = 'out/backgrounds/all_backgrounds.tif'

# Open test

img = tifffile.imread('data/INS1_BF/2000/3765_A17_T0001F001L01A01Z01C01.tiff')
# print(type(img))
img = np.rint(img)
img = img.astype('uint16')
print(img.shape)
print(img.dtype)

img2 = cv2.imread('data/INS1_BF/2000/3765_A17_T0001F001L01A01Z01C01.tiff', -1)
# no param = color, 0 = grayscale, -1=uint16
print(img2.shape)
print(img2.dtype)
# cv2.imshow('words', img2)
# cv2.waitKey(0)

img3 = Image.open('data/INS1_BF/2000/3765_A17_T0001F001L01A01Z01C01.tiff')
x = np.array(img3)
print(x.shape)
print(x.dtype)

path = 'test/'

# note correct size is uint16 (unsigned 16-bit int)

tifffile.imsave(path+'tifffile.tif', img)
# maintains the histogram converted size correctly

cv2.imwrite(path+'cv2.tif', img2)
# needs -1 after file name when opening, then has historgram and correct size

out = Image.fromarray(x, 'I;16')
out.save(path+'pil.tif')
# maintains histogram and correct size