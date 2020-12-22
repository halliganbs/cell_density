import os 
from typing import Union, Optional, List

import click # later if you're good
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar

from model import UNet
from dataloaders import H5Dataset, INS1Dataset

def validate():
    '''
    TODO:
    load in the weights of the model
    create that INS1_BF dataloader
    iterate over the dataset passing it to the model
    save the result image
    dump it into the NAS (manual)
    '''

    path ='models/cell_UNet.pth'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = UNet(input_filters=1,filters=64,N=2)
    model = model.to(device)
    print('made model')
    # model = nn.DataParallel(model, device_ids=[0], output_device=0)
    weights = torch.load(path)
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

    dataset = INS1Dataset('data/INS1_BF/2000/3765_A17_T0001F001L01A01Z01C01.tiff')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    bar = Bar('Finding cells...',max=len(loader))
    for  image in loader:
        image = image.to(device)
        result = model(image.float())
        bar.next()
    bar.finish()


    return True

if __name__=='__main__':
    validate()