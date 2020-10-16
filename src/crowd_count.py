import torch.nn as nn
from models.csrnet_pytorch.src import network
from models.csrnet_pytorch.src.model import CSRNet
import cv2

class CrowdCounter(nn.Module):
    def __init__(self, is_cuda=False):
        super(CrowdCounter, self).__init__()        
        self.model = CSRNet()
        #self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()
        self.is_cuda=is_cuda
        
    @property
    def loss(self):
        return self.loss_mse

    def forward(self, im_data, gt_data=None):        
        im_data = network.np_to_variable(
            im_data, 
            is_cuda=self.is_cuda, 
            is_training=self.training
        )

        # generating density map + upsampling
        density_map = self.model(im_data)
        density_map = cv2.resize(density_map, (density_map.shape[0] * 8, density_map.shape[1] * 8), interpolation = cv2.INTER_LINEAR)
        
        if self.training:                        
            gt_data = network.np_to_variable(
                gt_data, 
                is_cuda=self.is_cuda, 
                is_training=self.training
            )
#             print(f'Ground Truth Map Size: {gt_data.shape}. Ground truth map type: {gt_data.dtype}')
            self.loss_mse = self.build_loss(density_map, gt_data)
            
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss = self.loss_fn(density_map, gt_data)
        return loss