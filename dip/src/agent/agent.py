'''
file:           /agent/agent.py
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

This file defines Agent class - object wrapping the model, it's training and evaluation
'''

import torch
import numpy as np

from agent.models.critic import Critic
from agent.models.regular import RegularNN
from agent.models.teamreg import TeamRegNN
from agent.models.coachreg import CoachRegNN

MODELS = {
    'regular': RegularNN,
    'teamreg': TeamRegNN,
    'coachreg': CoachRegNN
}


class Agent:
    ''''''

    def __init__(self, actor_cnf, scnd_actor_cnf, critic_cnf, scnd_critic_cnf,
                 i, env, agents_count, scnd_model_enabled, device):
        '''
        Initialize Agent class
        '''
        # Store the ID for further manipulation Load/Store
        self.id = i
        self.device = device
        # Now let's create the models
        # The primary model would be always initialized
        model = MODELS[actor_cnf['model']]
        self.actor = model(env, device=device, **actor_cnf)
        self.actor_target = model(env, device=device, **actor_cnf)
        self.critic = Critic(env, agents_count=agents_count, device=device, **critic_cnf)
        self.critic_target = Critic(env, agents_count=agents_count, device=device, **critic_cnf)
        # If secondary model is also configured, initialize it (might not be used though)
        if scnd_model_enabled:
            model = MODELS[scnd_actor_cnf['model']]
            self.scnd_actor = model(env, agents_count=agents_count, device=device, **scnd_actor_cnf)
            self.scnd_actor_target = model(env, agents_count=agents_count, device=device, **scnd_actor_cnf)
            self.scnd_critic = Critic(env, device=device, **scnd_critic_cnf)
            self.scnd_critic_target = Critic(env, device=device, **scnd_critic_cnf)

    def active_actor(self, primary):
        '''Return active actor'''
        return self.actor if primary else self.scnd_actor

    def active_actor_t(self, primary):
        '''Return active target actor'''
        return self.actor_target if primary else self.scnd_actor_target

    def active_critic(self, primary):
        '''Return active actor'''
        return self.critic if primary else self.scnd_critic

    def active_critic_t(self, primary):
        '''Return active target actor'''
        return self.critic_target if primary else self.scnd_critic_target

    def get_current_ac_set(self, primary):
        '''
        Obtain actor, actor target, critic and critic target from primary or secondary model
        
        @params:
            primary:    Return primary set of neural networks
        '''
        if primary:
            return self.actor, self.actor_target, self.critic, self.critic_target
        return self.scnd_actor, self.scnd_actor_target, self.scnd_critic, self.scnd_critic_target

    def choose_action(self, observations, primary=True, noise=None):
        '''
        Choose action from model given observation
        
        @params:
            observations:   Single env observation vector of a single agent
            primary:        Use primary model
            noise:          Add noise to actions. None: none; <0,1>: multiplicative coefficient
        '''
        # What actor would be used?
        actor = self.actor if primary else self.scnd_actor
        # Load the observations as pytorch tensor, send it into the same device as the inferred NN
        obs_tensor = torch.Tensor(observations).to(self.device)
        # Here we expect even 2D arrays
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.view((1, -1))
        actions = actor.forward(obs_tensor)
        if noise:
            actions = actions + torch.rand(actions.shape).to(self.device)
        # Detach action from gradient computing
        action_id = np.argmax(actions.detach().cpu().numpy(), axis=1)
        action_vector = np.zeros(actions.shape)
        action_vector[np.arange(action_vector.shape[0]), action_id] = 1
        # For single env only - return only first elements of the list
        return action_id[0], action_vector[0]

    def morph(self, original_network, target_network, tau):
        '''
        With the coefficient of tau, partially morph the target network to the original network
        
        @params:
            original_network:   Source for morphing
            target_network:     Target for morphing
        '''
        for w, tw in zip(original_network.parameters(), target_network.parameters()):
            tw.data.copy_(tau * w.data + (1 - tau) * tw.data)

    def update_networks(self, primary=True, tau=None):
        '''Update the neural network weights'''
        # Start with choosing the model to train
        actor, actor_t, critic, critic_t = self.get_current_ac_set(primary)
        # Now perform the actual training
        self.morph(actor, actor_t, tau=tau)
        self.morph(critic, critic_t, tau=tau)

