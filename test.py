import glob
import tifffile
import numpy as np
import torch

from torch import nn
from skimage.util.shape import view_as_blocks
from progress.bar import Bar

from image_spliter import sub_sample, get_images, split_images

all_path = 'data/INS1_BF/*/*.tiff'
path_2000 = 'data/INS1_BF/2000/*.tiff'
split_path = 'data/splits/2000/3765_M20_T0001F001L01A01Z01C01_0_3.tiff'
test_path = 'data/INS1_BF/2000/3765_M20_T0001F001L01A01Z01C01.tiff'


if __name__=='__main__':
    print(f'CUDA AVAILABLE: {torch.cuda.is_available()}')
    files = get_images(test_path, 1,2000,2000)
    img = sub_sample(files[0], 256)
    tifffile.imsave('test.tiff', img)