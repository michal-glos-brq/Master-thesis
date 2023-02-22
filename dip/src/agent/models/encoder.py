'''
file:           /agent/models/encoder.py
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

This file defines Encoder class, which would be used to process input observations
'''

import torch
import torch.nn as nn

class Encoder(torch.nn.Module):
    '''
    Cascade of Conv2d or Linear layers
    Simple encoder capable of encoding simple vector observations for 
    extracted information with Linear layer or raw image with CNN encoder
    '''
    def __init__(self, obs_raw, device='cpu', layers=[(115, 256)], layer_opts=[{}],
                 act=None, out_act=True, out_act_opts={}):
        '''
        Initialize Encoder
        
        @params:
            obs_raw:            Is observation input raw image?
            layers:             Layer config (linear nn: (in, out), CNN: (in, out, (kernel_x, kernel_y)))
            layer_opts:         Other kwargs for layer to be passed on
            act:                Name of activation functions used til the last layer
            out_act:            The name of the last activation. True if the same as activation
            out_act_opts:       Parameters for the last activation function (it's only purpose so far is Softmax, hence only the last activation has this option)
        '''
        # Initialize parent class
        super(Encoder, self).__init__()

        # Safety assert
        assert len(layers) and ( len(layers) == len(layer_opts) or len(layer_opts) <= 1 ), 'Incorrect configuration of encoder layers!'
        assert act == None or hasattr(nn, act), f'Activation function "{act}" does not exist in torch.nn, use torch.nn activations only!'
        assert out_act == None or hasattr(nn, out_act), f'Output activation function "{out_act}" does not exist in torch.nn, use torch.nn activations only!'

        # Save the configuration
        self.obs_raw = obs_raw
        # Obtain the requested activation function
        act = getattr(nn, act) if act else act
        # Load out_act the same way, but in case of out_act == True, use the same as other act (default behavior)
        out_act =  getattr(nn, out_act) if isinstance(out_act, str) else (act if out_act else out_act)
        # Extend additional options if a list with single element provided
        if len(layer_opts) == 1:
            layer_opts *= len(layers)
        # Obtain the layer class, so we would not have to obtain it each time we want it
        layer_cls = nn.Conv2d if self.obs_raw else nn.Linear

        # Create the model
        self.layers = nn.ModuleList()
        for i, layer in enumerate(layers):
            # Create Conv2d or Linear layer with provided config
            self.layers.append(layer_cls(*layer, **layer_opts[i]))
            # If also activation is provided, add it into the flow
            if (act and (i + 1 < len(layers))) or (out_act and (i + 1 == len(layers))):
                self.layers.append(act() if i + 1 < len(layers) else out_act(**out_act_opts))

        # Move to device passed by parameter
        self.to(device)

    def forward(self, x):
        '''Encoder inference'''
        for function in self.layers:
            x = function(x)
        return x