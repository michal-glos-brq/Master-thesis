'''
file:           /agent/models/critic.py
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

This file defines Critic class, simple neural network with linear (or Convolution when raw input) layers
Learning critic does not require any regularization training - hence we could exploit the encoder class
'''
import sys
import torch

from agent.models.encoder import Encoder

class Critic(torch.nn.Module):
    '''
    Simple neural network as Critic, each instance composes of 2 instances pf Encoder class.
    1st. Encoder instance would be encoder identical to Actors encoder, the second one would be
    identical to the rest of the structure of Actor but the regularization structure is neglected.
    '''

    def __init__(self, env, agents_count, obs_raw, lr, device,
                 layers_enc, layer_opts_enc, act_enc, out_act_enc, out_act_opts_enc, 
                 layers_policy, layer_opts_policy, act_policy, out_act_policy, out_act_opts_policy,
                 **kwargs):
        '''
        Compose neural network out of 2 Encoders and one Linear network as for approximating the action score
        
        The naming gere is not quite descriptive as usefull - we could grab actor critic config dict and just reuse it here
        @params:
            env:                    Env object instance
            agents_count:           N agents system (MAX)
            lr:                     Beta parameter to Adam optimizer
            obs_raw:                Is observation RAW IMAGE or extracted information vector?
            layers_enc:             First encoder parameter `layers`
            layer_opts_enc:         First encoder parameter `layer_opts`
            act_enc:                First encoder parameter `act`
            out_act_enc:            First encoder parameter `out_act`
            out_act_opts_enc:       First encoder parameter `out_act_opts`
            layers_policy:          Second encoder parameter `layers`
            layer_opts_policy:      Second encoder parameter `layer_opts`
            act_policy:             Second encoder parameter `act`
            out_act_policy:         Second encoder parameter `out_act`
            out_act_opts_policy:    Second encoder parameter `out_act_opts`
        Note: naming variables with suffix policy ait'n straightforward, it's just to keep the actor settings
        '''
        # Initialize parent class
        super(Critic, self).__init__()
        # First encoder - would just encode the observation
        # Check the layers
        input_size = ((env.observation_space + env.action_space) * agents_count)
        if layers_enc[0][0] != input_size:
            print(f'Warning: The dimension of first layer of critic was in: {layers_enc[0][0]};' + 
                  f'out: {layers_enc[0][1]}. Changing it to in: {input_size}; out: {layers_enc[0][1]}.', file=sys.stderr)
            layers_enc[0] = (input_size, layers_enc[0][1])
            
        if layers_enc[-1][1] != layers_policy[0][0]:
            print(f'Error: The dimension of last layer of encoder part of critic was: {layers_enc[-1][1]};' + 
                  f' the policy network is: {layers_policy[0][0]}.', file=sys.stderr)
            sys.exit()
        
        if layers_policy[-1][1] != 1:
            print(f'Warning: Output of critic set to size {layers_policy[-1][1]}, setting to 1, as we approximate single value')


        self.encoder = Encoder(obs_raw, device, layers_enc, layer_opts_enc, act_enc, out_act_enc, out_act_opts_enc)
        # Second encoder - would take encoded observation as input
        self.linear = Encoder(False, device, layers_policy, layer_opts_policy, act_policy, out_act_policy, out_act_opts_policy)
        # Optimizer to learn the model
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # Transfer to CUDA if possible
        self.to(device)

    def forward(self, state, action):
        '''
        Neural network inference

        @params:    
            state:  State vector as critic input
            action: Action vector as critic input
        '''
        return self.linear(self.encoder(torch.cat([state, action], dim=1)))

    def calculate_additional_losses(self, **kwargs):
        '''Calculate additional losses from conditioning the neural network or regularization'''
        return torch.Tensor([0]).to(torch.device('cuda:0')) if torch.cuda.is_availible() else torch.Tensor([0])
