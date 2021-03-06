import os 
from typing import Union, Optional, List

import click # later if you're good
import torch
import torch.nn as nn
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from progress.bar import Bar

from model import UNet
from dataloaders import H5Dataset, INS1Dataset

def validate():
    '''
    TODO:
    fix cuda memory problem
    save the result image
    dump it into the NAS (manual)
    '''

    path ='models/cell_UNet.pth'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    weights = torch.load(path)
    out = None
    with torch.no_grad():

        model = UNet(input_filters=1,filters=64,N=2)
        model = model.to(device)
        print('made model')

        model.load_state_dict(weights, strict=False)
        print('loaded model weights')
    
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.05,
                                    momentum=0.9,
                                    weight_decay=1e-5)
        optimizer.zero_grad()
        print('loaded optimizer')
        # set to evaluate mode 
        model.eval()

        # data/splits/2000/3765_A17_T0001F001L01A01Z01C01_0_0.tiff
        dataset = INS1Dataset('sub.tiff')
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        bar = Bar('Finding cells...',max=len(loader))
        for  image in loader:
            image = image.to(device)
            result = model(image.float())
            out = result
            torch.cuda.empty_cache()
            bar.next()
        bar.finish()

        print(out.shape)
    out = out.cpu()
    return out.numpy()[0,0,:,:]

if __name__=='__main__':
    new = validate()
    tifffile.imsave('out.tiff', new)