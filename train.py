import os
from typing import Union, Optional, List

import click # might not use
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import UNet
from loop import Looper
from dataloaders import H5Dataset

learning_rate = 1e-2
epochs = 100
batch_size = 8
horizontal_flip = 0.0
vertical_flip = 0.0
unet_filters = 64
n = 2
plot = True

# stolen check README
def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = {}
    dataloader = {}

    dataset_path = 'data/cell/'

    for m in ['train', 'valid']:
        data_path = os.path.join(dataset_path, f"{m}.h5")
        dataset[m] = H5Dataset(data_path,
                               horizontal_flip=horizontal_flip,
                               vertical_flip=vertical_flip)
        dataloader[m] = torch.utils.data.DataLoader(dataset[m],
                                    batch_size=batch_size)

    print('dataset made')
    input_channels = 1
    
    # model TODO: get the sizes
    model = UNet(
        input_filters=input_channels, filters=unet_filters, N=n
    ).to(device)
    model = torch.nn.DataParallel(model)

    print('Model Made')

    # loss and optimizer
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    print('Loss and Optimizer')

    # plot if flag is to plot
    if plot:
        plt.ion()
        fig, plots = plt.subplots(nrows=2, ncols=2)
    else:
        plots = [None]*2

    print('Plot setup')

    # train and validate loopers for each epoch
    train_looper = Looper(model, device, loss, optimizer, 
                            dataloader['train'], len(dataset['train']), 
                            plots[0])

    valid_looper = Looper(model, device, loss, optimizer, 
                            dataloader['valid'], len(dataset['valid']), 
                            plots[0], validation=True)

    print('Loopers')
    
    # current best results (lowest mean absolute error on validation set)
    current_best = np.infty

    for epoch in range(epochs):
        print('Epoch: {}\n'.format(epoch+1))

        # TRAIN LOOP
        train_looper.run()
        lr_scheduler.step()

        # validation epoch
        with torch.no_grad():
            result = valid_looper.run()
        
        # upate checkpoint for new best
        if result < current_best:
            current_best = result
            torch.save(model.state_dict(),'cells_UNet.pth')
            print('\nNew Best Result: {}'.format(result))
        
        print('\n','-'*80,'\n',sep='')
    
    print('[Training Complete] Best Result: {}'.format(current_best))

if __name__ =='__main__':
    train()


