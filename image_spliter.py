import os
import glob
import tifffile
import re # REEEEE

import numpy as np
from skimage.util.shape import view_as_blocks
from progress.bar import Bar

path = 'data/INS1_BF/*/*.tiff'
pattern = r'(\/[0-9]*\/3765_[A-Z][0-9]{2}_T0001F001L01A01Z01C01)'

def get_images(path):
    return glob.glob(path)

def split_images(files):
    bar = Bar('SPLITING FILES',max=len(files))
    for f in files:
        img = tifffile.imread(f)
        splits = view_as_blocks(img, (500,500))
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

if __name__=='__main__':
    files = get_images(path)
    split_images(files)