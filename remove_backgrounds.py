import os
import cv2
import numpy as np
import glob
import h5py
import re # REEEE
import tifffile
from progress.bar import Bar

DATA_PATH = 'data/INS1_BF/'
OUT_PATH = 'data/cleaned/'

reg = '(\d*\/3765_[A-Z]{1}[0-9]{2}_.*)'

back = cv2.imread('out/backgrounds/background.tif', -1)

data = glob.glob(DATA_PATH+'*/*.tiff')

# bar = Bar('Removing Background:',max=len(data))
# for file in data:
#     img = cv2.imread(file,-1)
#     n = img-back
#     new_file = re.search(reg, file)
#     new_path = new_file.group(0)
#     cv2.imwrite(OUT_PATH+new_path, n)
#     bar.next()
# bar.finish()

def test_background(path):
    '''
    params:
        path: path to file to test
    returns: img as numpy array without background
    '''
    img = cv2.imread(path, -1)
    new_img = back-img
    new_img = new_img.astype('uint16')
    cv2.imwrite('test.tif',new_img)
    return new_img

img = test_background('data/INS1_BF/2000/3765_A17_T0001F001L01A01Z01C01.tiff')
print(img.dtype)
print(img.shape)
print(img.max())
print(img.min())
print(img)