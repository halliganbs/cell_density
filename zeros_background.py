import numpy as np
import glob
import re # REEEEE
import cv2
from progress.bar import Bar
import tifffile as tiff

from image_spliter import get_images, blurr, filter_img

zeros = 'data/INS1_BF/0/*.tiff'

def zero_background(imgs):
    """
    averages all zero backgrounds, after applying
    guass blur and sobel filter
    args:
        imgs - numpy array of images (l, c, h, w)
    returns:
        bar - averaged image
    """
    zero = np.zeros((1, 2000, 2000))
    bar = Bar('Averaging Images',max=len(imgs))
    for img in imgs:
        i = blurr(img)
        # i = filter_img(i, 'sobel')
        zero += i
        bar.next()
    bar.finish()
    out = zero/len(imgs)
    return out

if __name__=='__main__':
    files = glob.glob(zeros)
    imgs = np.zeros((len(files), 1, 2000, 2000))
    for i, f in enumerate(files):
        imgs[i,0] = tiff.imread(f)
    z = zero_background(imgs)
    tiff.imsave('zeroed.tiff', z)