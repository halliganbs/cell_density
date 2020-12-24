import os
import glob
import tifffile
import re # REEEEE

import numpy as np
from skimage.util.shape import view_as_blocks
from progress.bar import Bar

path = 'data/INS1_BF/*/*.tiff'
pattern = r'(\/[0-9]*\/3765_[A-Z][0-9]{2}_T0001F001L01A01Z01C01)'

def split_images(files,x,y):
    """
    breaks image apart into size x,y
    args:
        files list of file names
    x:
        x size
    y:
        y size
    returns:
        numpy array of new images (x,y,img-x, img-y)
    """
    bar = Bar('SPLITING FILES',max=len(files))
    for f in files:
        img = tifffile.imread(f)
        splits = view_as_blocks(img, (x,y))
        x_axis = splits.shape[0]
        y_axis = splits.shape[1]
        filename = re.split(pattern,f)
        filename = filename[1]
        for x in range(x_axis):
            for y in range(y_axis):
                imgname = '{}{}_{}_{}.tiff'.format('data/splits',filename,x,y)
                tifffile.imsave(imgname,splits[x,y])
        bar.next()
    bar.finish()

def sub_sample(img, size):
    """
    Takes a 'random' sample of size size of larger image
    
    args:
        img - numpy array of image
        size - height and width of subsampled image
    return:
        numpy array sub sample
    TODO:
        make it randomly select area instead of top left corner
    """
    sub = img[:size,:size]
    return sub

def get_images(path, c,h,w):
    """
    loads images into numpy array
    args:
        path - path to files
        c,h,w - color channel, height, width
    """
    files = glob.glob(path)
    images = np.zeros(len(files), c, h, w)
    bar=Bar('Loading Images', max=len(files))
    for i, f in enumerate(files):
        images[i,0]=tifffile.imread(f)
        bar.next()
    bar.finish()
    return images

if __name__=='__main__':
    files = get_images(path, 1, 2000, 2000)
    split_images(files, 250,250)