import numpy as np
import glob
import re # REEEEE
import cv2
from progress.bar import Bar
import tifffile as tiff

PATH = 'data/INS1_BF/*/*.tiff'
data = glob.glob(PATH)

shape = (2000,2000)
size = len(data)
print(size)
whole = np.zeros(shape)
bar = Bar('Merging Backgrounds', max=len(data))
for file in data:
    # print(file)
    img = tiff.imread(file) # read in grayscale uint16
    whole += img
    img = None
    bar.next()
bar.finish()

whole = whole/size
whole = np.rint(whole)
whole = whole.astype('uint32')
print(whole)
print(whole.shape)
print(whole.dtype)

tiff.imwrite('out/backgrounds/background.tif', whole)