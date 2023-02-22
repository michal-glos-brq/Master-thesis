'''
file:           /file_manager.py
author:         Michal Glos (xglosm01)
established:    22.1.2023
last modified:  22.1.2023

                        ##################
                        #@$%&        &%$@#
                        #!   <(o )___   !#
                        #!    ( ._> /   !#
                        #!     `---'    !#
                        #@$%&        &%$@#
                        ##################

This file provides methods to load and parse configuration and to load and save models
'''

import yaml

def load_config(config_path):
    '''Load and parse configuration YAML'''
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg
    
def load_models(agents, config):
    '''Load neural network parameters into Agent (Agent instances in Agents.agents list)'''
    raise NotImplemented

def save_models(agents, config):
    ''''''
    pass