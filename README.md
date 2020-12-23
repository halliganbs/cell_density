# cell_density
Prediction of cell density for the INS1_BF dataset

## Other Projects used

### Objects counting by estimating a density map with convolutional neural networks

- https://github.com/NeuroSYS-pl/objects_counting_dmap
- [Pytorch Implementation](https://github.com/WeidiXie/cell_counting_v2)
- [Blog Post](https://neurosys.com/article/objects-counting-by-estimating-a-density-map-with-convolutional-neural-networks/)

### H5 File packages

- https://docs.h5py.org/en/stable/quick.html
- https://stackoverflow.com/questions/28170623/how-to-read-hdf5-files-in-python
- https://www.geeksforgeeks.org/hdf5-files-in-python/

### scikit-image Packages

- CPU parallelization https://scikit-image.org/docs/stable/user_guide/tutorial_parallelization.html
- Backgroun Subtraction
    - https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_regional_maxima.html
    - https://flothesof.github.io/removing-background-scikit-image.html

## TODO:

1. ~~Seperated~~ Sub-sample 2000x2000 images into ~~200x200~~ 256x256 ✓
2. Apply either Sobel or Roberts filter, from `skimage.filters`
1. Feed filtered images using `validate.py` to get resulting dense maps
1. Get some form of prediction going to create a dense map similar to the `cell` data
2. Create list of options to increase predictions on `INS1_BF` data
3. Store data and meta data on NAS
4. ~~Rerun `validate.py` with smaller images to generate density maps~~ ✓