import torch
import torch.nn as nn
from network import Conv2d

class MCNN(nn.Module):
    '''
    Multi-column CNN 
        - Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        
        self.branch1 = nn.Sequential(Conv2d( 3, 16, 9, padding='same', bn=bn),
                                     Conv2d(16, 32, 7, padding='same', bn=bn),
                                     Conv2d(32, 16, 7, padding='same', bn=bn),
                                     Conv2d(16,  8, 7, padding='same', bn=bn))
        
        self.branch2 = nn.Sequential(Conv2d( 3, 20, 7, padding='same', bn=bn),
                                     Conv2d(20, 40, 5, padding='same', bn=bn),
                                     Conv2d(40, 20, 5, padding='same', bn=bn),
                                     Conv2d(20, 10, 5, padding='same', bn=bn))
        
        self.branch3 = nn.Sequential(Conv2d( 3, 24, 5, padding='same', bn=bn),
                                     Conv2d(24, 48, 3, padding='same', bn=bn),
                                     Conv2d(48, 24, 3, padding='same', bn=bn),
                                     Conv2d(24, 12, 3, padding='same', bn=bn))
        
        self.fuse    = nn.Sequential(Conv2d(30,  1, 1, padding='same', bn=bn))
        
    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3), 1)
        x = self.fuse(x)
        return x

class CSRNet(nn.Module):
    '''
    CSRNet CNN
        - Implementation of Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes (Li et al.)
    '''

    def __init__(self, bn=False):
        super(CSRNet, self).__init__()

        self.column = nn.Sequential(Conv2d(  3,  64, 3, padding='same', bn=bn), 
                                    Conv2d( 64,  64, 3, padding='same', bn=bn),
                                     
                                    #nn.MaxPool2d(2, stride=2),
                                     
                                    Conv2d( 64, 128, 3, padding='same', bn=bn), 
                                    Conv2d(128, 128, 3, padding='same', bn=bn),
                                     
                                    #nn.MaxPool2d(2, stride=2),
                                     
                                    Conv2d(128, 256, 3, padding='same', bn=bn), 
                                    Conv2d(256, 256, 3, padding='same', bn=bn),
                                    Conv2d(256, 256, 3, padding='same', bn=bn),
                                     
                                    #nn.MaxPool2d(2, stride=2),
                                     
                                    Conv2d(256, 512, 3, padding='same', bn=bn),
                                    Conv2d(512, 512, 3, padding='same', bn=bn),
                                    Conv2d(512, 512, 3, padding='same', bn=bn),
                                     
                                    # backend - fully conv layers
                                    # going ahead with CSRNet with backend B configuration
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

    # to delete?
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)