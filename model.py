"""U-Net Model"""
# Stolen from : https://github.com/NeuroSYS-pl/objects_counting_dmap

from typing import Tuple

import numpy as np
import torch
from torch import nn

def con_bloc(channels: Tuple[int, int], size: Tuple[int, int],stride: Tuple[int, int]=(1,1), N: int=1):
    """
    Block of N conv layers w/ RELU activation
    First layer IN x OUT, rest - OUT x OUT

    args:
        channels: (IN, OUT)
        size: Kernel size (fixed)
        stride: stride
        N: number of layers
    
    returns:
        sequential model of N conv layers
    """

    block = lambda in_channels: nn.Sequential(
        nn.Conv2d(in_channels=in_channels, 
                  out_channels=channels[1],
                  kernel_size=size, 
                  stride=stride, 
                  bias=False, 
                  padding=(size[0] // 2, size[1] // 2)),
        nn.BatchNorm2d(num_features=channels[1]),
        nn.ReLU()
    )

    return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])


class ConvUp(nn.Module):
    """ convolution layer with upsampling + concatenate block"""

    def __init__(self,
                channels: Tuple[int, int], 
                size: Tuple[int, int], 
                stride: Tuple[int, int]=(1,1), 
                N: int=1):

        """ sequential with conv block of N layers upsampling by 2 """
        super(ConvUp, self).__init__()
        self.conv = nn.Sequential(
            con_bloc(channels, size, stride, N),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, to_conv: torch.Tensor, to_cat: torch.Tensor):
        """ Wisconsin's Motto
        
        args:
            to_conv: input passed to conv block than upsampled
            to_cat: input concatenated with the output of a conv block
        """

        return torch.cat([self.conv(to_conv), to_cat], dim=1)

class UNet(nn.Module):
    """
    U-Net model - kernel sizes and stride may need to be changed due to larger images used
    """

    def __init__(self, filters: int=64, input_filters: int=3, **kwargs):
        """
        fixed kernel size = (3, 3)
        fixed  max pooling kernel size = (2,2)
        upsample factor = 2
        fixed num conv layers

        args:
            filters: num conv layers
            input_filters: num of input channels
        """
        super(UNet, self).__init__()
        initial_filters  = (input_filters, filters)
        down_filters = (filters, filters)
        up_filters = (2*filters, filters)

        # downsampling
        self.block1 = con_bloc(channels=initial_filters, size=(3,3), N=2)
        self.block2 = con_bloc(channels=down_filters, size=(3,3), N=2)
        self.block3 = con_bloc(channels=down_filters, size=(3,3), N=2)

        # upsampling
        self.block4 = ConvUp(channels=down_filters, size=(3,3), N=2)
        self.block5 = ConvUp(channels=up_filters, size=(3,3), N=2)
        self.block6 = ConvUp(channels=up_filters, size=(3,3), N=2)

        # density prediction
        self.block7 = con_bloc(channels=up_filters, size=(3,3), N=2)
        self.density_pred = nn.Conv2d(in_channels=filters, out_channels=1, kernel_size=(1,1), bias=False)


    def forward(self, in_tensor: torch.Tensor):
        """F O R W A R D"""
        pool = nn.MaxPool2d(2)

        # down
        block1 = self.block1(in_tensor)
        pool1 = pool(block1)
        block2 = self.block2(pool1)
        pool2 = pool(block2)
        block3 = self.block3(pool2)
        pool3 = pool(block3)

        # up
        block4 = self.block4(pool3, block3)
        block5 = self.block5(block4, block2)
        block6 = self.block6(block5, block1)

        # dense prediction
        block7 = self.block7(block6)
        return self.density_pred(block7)
