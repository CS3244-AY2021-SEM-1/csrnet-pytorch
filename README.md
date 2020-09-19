# CSRNet using pytorch
## This repo will be used to design CSRNet with pytorch

Contains code adapted from https://github.com/CS3244-AY2021-SEM-1/mcnn-pytorch

### Changes made:

#### models.py
- added new class CSRNet
- commented out the max pooling layers as we have yet to resize the data

#### network.py
- updated Conv2d to accomodate dilated conv layers

#### data_loader.py
- updated ImageDataLoader to fit our usage of .h5 files
- our .h5 files contain both the raw image and the gt density map, no need for a separate path to the gt files

#### crowd_count.py
- updated CrowdCounter's constructor to allow for switching between CSRNet and MCNN architectures
- to instantiate a new NN:
  - csrnet = CrowdCounter('csrnet')
  - mcnn = CrowdCounter('mcnn')
