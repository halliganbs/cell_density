import glob
import tifffile
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn
from skimage.util.shape import view_as_blocks
from progress.bar import Bar

from image_spliter import sub_sample, get_images, split_images, filter_img, blurr

all_path = 'data/INS1_BF/*/*.tiff'
path_2000 = 'data/INS1_BF/2000/*.tiff'
split_path = 'data/splits/2000/3765_M20_T0001F001L01A01Z01C01_0_3.tiff'
test_path = 'data/INS1_BF/2000/3765_M20_T0001F001L01A01Z01C01.tiff'
zero_test = 'data/INS1_BF/0/3765_A21_T0001F001L01A01Z01C01.tiff'

if __name__=='__main__':
    print(f'CUDA AVAILABLE: {torch.cuda.is_available()}')
    zero = tifffile.imread('zeroed.tiff')
    test = tifffile.imread(test_path)
    
    out = test-zero

    test = filter_img(test, 'sobel')
    test = blurr(test)
    
    tifffile.imsave('blur_filter_M20_2000.tiff', out)