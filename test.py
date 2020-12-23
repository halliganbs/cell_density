import glob
import tifffile
import numpy as np

from skimage.util.shape import view_as_blocks
from progress.bar import Bar

all_path = 'data/INS1_BF/*/*.tiff'
path_2000 = 'data/INS1_BF/2000/*.tiff'
split_path = 'data/splits/2000/3765_M20_T0001F001L01A01Z01C01_0_3.tiff'
test_path = 'data/INS1_BF/2000/3765_M20_T0001F001L01A01Z01C01.tiff'

def get_files(path):
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

def split_image(files):
    print(files[0])
    img = tifffile.imread(files[0])
    splits = view_as_blocks(img, block_shape=(500,500))
    x_axis = splits.shape[0]
    y_axis = splits.shape[1]
    for x in range(x_axis):
        for y in range(y_axis):
            filename = f'data/test/test{x}_{y}.tiff'
            tifffile.imsave(filename,splits[x,y])

def sub_sample(file, x, y):
    """
    take a sub sample of an image and make a new tiff 

    args:
        file - path to image
        x - x-axis dimension of new image
        y - y-axis dimension of new image

    returns:
        numpy array of new image
    """
    img = tifffile.imread(file)
    sub_img = img[:x,:y]
    return sub_img

if __name__=='__main__':
    sub = sub_sample(test_path,256,256)
    tifffile.imsave('sub.tiff',sub)