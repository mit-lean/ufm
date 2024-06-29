import torch
import torch.nn as nn
from . import FCDenseNet
from uncertainty_from_motion.src import utils
import time # debugging

import sys
run_parameters = utils.initiate_parameters(sys.argv[1])
class FCDenseNetAutoencoder(nn.Module):
    def __init__(self, in_chs=3):
        super(FCDenseNetAutoencoder, self).__init__()
        self.autoencoder = FCDenseNet.FCDenseNet_make_autoencoder(in_channels=in_chs,
                                                pretrained=run_parameters['pretrained'])
    def forward(self, x):
        pred = self.autoencoder.forward(x)
        return pred