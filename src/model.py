import torch
import torch.nn as nn
from models.smc.src.network import Conv2d

class CSRNet(nn.Module):
    '''
    CSRNet CNN
        - Implementation of Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes (Li et al.)
    '''

    def __init__(self, bn=False):
        super(CSRNet, self).__init__()

        self.column = nn.Sequential(Conv2d(  3,  64, 3, padding='same', bn=bn), 
                                    Conv2d( 64,  64, 3, padding='same', bn=bn),
                                     
                                    nn.MaxPool2d(2, stride=2),
                                     
                                    Conv2d( 64, 128, 3, padding='same', bn=bn), 
                                    Conv2d(128, 128, 3, padding='same', bn=bn),
                                     
                                    nn.MaxPool2d(2, stride=2),
                                     
                                    Conv2d(128, 256, 3, padding='same', bn=bn), 
                                    Conv2d(256, 256, 3, padding='same', bn=bn),
                                    Conv2d(256, 256, 3, padding='same', bn=bn),
                                     
                                    nn.MaxPool2d(2, stride=2),
                                     
                                    Conv2d(256, 512, 3, padding='same', bn=bn),
                                    Conv2d(512, 512, 3, padding='same', bn=bn),
                                    Conv2d(512, 512, 3, padding='same', bn=bn),
                                     
                                    # backend - fully conv layers (CSRNet backend B configuration)
                                    Conv2d(512, 512, 3, padding=2, bn=bn, dilation=2),
                                    Conv2d(512, 512, 3, padding=2, bn=bn, dilation=2),
                                    Conv2d(512, 512, 3, padding=2, bn=bn, dilation=2),
                                    Conv2d(512, 256, 3, padding=2, bn=bn, dilation=2),
                                    Conv2d(256, 128, 3, padding=2, bn=bn, dilation=2),
                                    Conv2d(128,  64, 3, padding=2, bn=bn, dilation=2),
                                     
                                    # output layer
                                    Conv2d(64, 1, 1, padding='same', bn=bn))        

    def forward(self, im_data):
        x = self.column(im_data)
        return x