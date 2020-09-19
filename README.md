# CSRNet using pytorch
## This repo will be used to design CSRNet with pytorch

Contains code adapted from https://github.com/CS3244-AY2021-SEM-1/mcnn-pytorch

### Changes made:

#### models.py
- added new class CSRNet

#### network.py
- updated Conv2d to accomodate dilated conv layers

#### data_loader.py
- updated ImageDataLoader to fit our usage of .h5 files

#### crowd_count.py
- updated CrowdCounter's constructor to allow for switching between CSRNet and MCNN architectures
