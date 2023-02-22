'''
file:           /agent/models/teamreg.py
author:         Michal Glos (xglosm01)
established:    20.1.2023
last modified:  20.1.2023

                        ##################
                        #@$%&        &%$@#
                        #!   <(o )___   !#
                        #!    ( ._> /   !#
                        #!     `---'    !#
                        #@$%&        &%$@#
                        ##################

This file defines the Team-regularized neural network for policy estimation
'''

import torch
from agent.models.encoder import Encoder

class TeamRegNN(torch.nn.Module):
    '''
    This is regular linear neural network providing actions (pi(s) = a) given observation
    '''
    def __init__(self):
        '''Initialize RegularNN'''
        # Initialize parent class
        super(TeamRegNN, self).__init__()

    def forward(self):
        '''Inference'''
        pass

    def calculate_loss(self, **kwargs):
        '''
        Calculate loss (is not part of MADDPG because each model would require
        different process for loss computing). Uses **kwargs, because each computation
        would require different data, so we pick data we need and ignore the rest.
        '''
        pass