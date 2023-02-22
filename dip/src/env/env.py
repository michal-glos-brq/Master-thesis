'''
file:           /env/env.py
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

This is wrapper around environment, provides classic OpenAI interface
Wrapper is here because different environments could be used, copmplying to different standards
'''

from pprint import pprint
import gfootball.env as football_env

class Env:
    '''
    OpenAI and other environment wrapper (envs for multi-agent systems support only)
    
    Provides 
    '''
     
    def __init__(self, env_config):
        '''
        Initialize environment wrapper
        
        @params:
            env_config:     Environment configuration loaded from config file
        '''
        self.cnf = env_config
        if self.cnf['name'] == 'gfootball':
            self.env = football_env.create_environment(**self.cnf['builder_params'])
            if self.cnf['builder_params']['render']:
                self.env.render()

    @property
    def action_space(self):
        '''Obtain number of actions'''
        # Multi agent settings
        try:
            return self.env.action_space[0].n
        # Single agent settings
        except:
            return self.env.action_space.n

    @property
    def observation_space(self):
        '''Obtain the shape of observation space'''
        # Multi agent settings
        try:
            return self.env.observation_space.shape[0]
        # Single agent settings
        except:
            return self.env.observation_space.shape

    def dump_config(self):
        '''Dump environment configuration and also env attribute shapes and sizes'''
        print('Configuration provided by config file + CLI arguments:')
        pprint(self.cnf)
        print('Configuration of the instantiazed environment:')
        print(f'Observation space {self.observation_space}')
        print(f'Action space {self.action_space}')

    @property
    def max_steps(self):
        '''Maximal number of steps in env (TODO: generate automatically)'''
        return 100

    @property
    def max_agents(self):
        '''Return maximal number of agents'''
        return self.cnf['max_agents']

    @property
    def actual_agents(self):
        '''Obtain the number of agents actually in the environment (agents controlled by model)'''
        if self.cnf['name'] == 'gfootball':
            # First dimension is dimension of agents
            return self.cnf['max_agents']

    @property
    def agent_abservation_shape(self):
        '''Obtain the shape of agent observation space'''
        # We assume the somehow coherent polymorphism, hence the wrapper, to provide coherent API for out agent
        return self.env.observation_space[0].shape

    @property
    def agent_action_space(self):
        '''Obtain action space of a single agent. Implicitly take only the space
        of the first actor - the same shape of actions is assumed for each agent'''
        return self.env.action_space[0]
    
    def safe_render(self):
        '''Render if and only if render parameter in cfg['eval_builder_params_update']['render'] se to True'''
        if self.cnf['builder_params']['render'] and self.cnf['name'] != 'gfootball':
            self.env.render()

    def reset(self):
        '''Render the invironment'''
        observations = self.env.reset()
        # If configured, env will be rendered
        self.safe_render()
        return observations
    
    def step(self, actions):
        '''Perform a single step in the environment'''
        # Crop the action space provided by the model
        actions = actions[:self.actual_agents]
        # If configured, env will be rendered
        self.safe_render()
        return self.env.step(actions)