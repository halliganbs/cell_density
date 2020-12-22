import glob
import tifffile
import numpy as np

from progress.bar import Bar


def get_files():
    path = 'data/INS1_BF/*/*.tiff'
    files = glob.glob(path)
    return files

def get_images(files):
    images = np.zeros((len(files),1, 2000, 2000))

    bar = Bar('Reading in files', max=len(files))
    for i, f in enumerate(files):
        images[i,0] = tifffile.imread(f)
        bar.next()
    bar.finish()
    print(images.shape)


if __name__=='__main__':
    files =get_files()
    get_images(files)