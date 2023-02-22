'''
file:           /agent/maddpg.py
author:         Michal Glos (xglosm01)
established:    21.1.2023
last modified:  21.1.2023

                        ##################
                        #@$%&        &%$@#
                        #!   <(o )___   !#
                        #!    ( ._> /   !#
                        #!     `---'    !#
                        #@$%&        &%$@#
                        ##################

This file defines the MADDPG algorithm, other ways of calculating losses
are model-scpecific and rely only on model-provided methods
'''

import torch
import copy
from tqdm import tqdm
import numpy as np

from agent.agent import Agent


class Agents:
    '''Class wrapping multiple agent instancies and providing it's training and evaluating'''
    def __init__(self, primary_cnf, secondary_cnf, env):
        '''
        Initialize agents
        
        @params:
            env:            Env wrapper object instance
            primary_cnf:    Configuration dict for primary model
            secondary_cnf:  Configuration dict for primary, if bool(secondary_cnf) == False,
                            no secondary model would be configured
        '''
        # Store the env object, it's properties would be used to define actors and critics
        self.env = env
        self.scores = []
        # Parse the configuration
        self.primary_cnf, self.secondary_cnf = primary_cnf, secondary_cnf
        self.actor_cnf = primary_cnf['actor_cnf']
        self.scnd_actor_cnf = secondary_cnf['actor_cnf']
        self.critic_cnf = copy.deepcopy(self.actor_cnf)
        self.critic_cnf.update(primary_cnf['critic_cnf'])
        self.scnd_critic_cnf = copy.deepcopy(self.scnd_actor_cnf)
        self.scnd_critic_cnf.update(secondary_cnf['critic_cnf'])

        # Obtain the pytorch device to be used
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # Initialize agents
        self.init_agents()
        self.loss_fn = torch.nn.MSELoss()

    def create_agent(self, i):
        '''
        Create and add single agent
        
        @params:
            i:      Agent's order number (id)
        '''
        # Use goal agents - critic has to be trained on fixed shape of input, hence the largest possible number
        return Agent(actor_cnf=self.actor_cnf, scnd_actor_cnf=self.scnd_actor_cnf,
                     critic_cnf=self.critic_cnf, scnd_critic_cnf=self.scnd_critic_cnf,
                     i=i, env=self.env, agents_count=self.cnf()['goal_agents'],
                     scnd_model_enabled=self.cnf(primary=False)['enabled'], device=self.device)

    def cnf(self, primary=True):
        '''Return primary or secondary model config'''
        return self.primary_cnf if primary else self.secondary_cnf

    def init_agents(self):
        '''Initialize required number of Agent class instances'''
        self.agents = []
        for i in range(self.cnf(primary=True)['start_agents']):
            self.agents.append(self.create_agent(i))
    
    def choose_actions(self, observations, primary=True, noise=None):
        '''Choose actions from agent given observation'''
        obs = observations if len(self.agents) > 1 else [observations]
        action_tuples = [
            agent.choose_action(observation, primary=primary, noise=noise)
            for agent, observation in zip(self.agents, obs)
        ]
        return [a[0] for a in action_tuples], [a[1] for a in action_tuples]
         


    def learn(self, buffer, primary=True):
        '''
        Perform a single step of model traing
        
        @params:
            buffer:     Replay buffer with already set and configured settings
            primary:    Learn the primary model
        '''
        # Sample only experience made with chosen model or whatever? (Set the primary parameter for sampling)
        cnf = self.cnf(primary=primary)
        auth_exp_only = primary if cnf['authentic_experience'] else None
        sample = buffer.sample_buffer(cnf['batch'], with_return=False, primary=auth_exp_only)
        # Convert samples into tensors
        sample = {key: torch.Tensor(val).to(self.device) for key, val in sample.items()}

        # Now let's infere some actions from env states
        agents_actions_new_s = []
        agents_actions_old_s = []
    
        for idx, agent in enumerate(self.agents):
            # Get potential actions from upcoming states and save it
            new_policies = agent.active_actor_t(primary).forward(sample['actor_next_states'][:, idx])
            agents_actions_new_s.append(new_policies)
            # Now get the 'rethink' of actually executed actions/state
            old_policies = agent.active_actor(primary).forward(sample['actor_states'][:, idx])
            agents_actions_old_s.append(old_policies)
        

        agents_actions_new_s = torch.cat(agents_actions_new_s, dim=1)
        agents_actions_old_s = torch.cat(agents_actions_old_s, dim=1)
        sample['actions']    = sample['actions'].view((cnf['batch'], -1))
        

        for idx, agent in enumerate(self.agents):
            # Now compute the objective function
            # Obtain networks in use
            actor = agent.active_actor(primary)
            critic = agent.active_critic(primary)
            critic_t = agent.active_critic_t(primary)

            # Get the Qs
            new_states_Q = critic_t.forward(sample['global_next_states'], agents_actions_new_s).flatten()
            # We obviously obtained 0 reward from interacting with finished environment
            new_states_Q[sample['done'] == 1.] = 0.
            states_Q = critic.forward(sample['global_states'], sample['actions']).flatten()
            
            # Now compute the losses and update the weights (sum(dim=1) - we have reward for each agent)
            target = sample['reward'].sum(dim=1) + cnf['gamma'] * new_states_Q
            c_loss = self.loss_fn(target, states_Q)
            critic.optimizer.zero_grad()
            #make_dot(c_loss, params=dict(critic.named_parameters()), show_attrs=False, show_saved=True)
            #import pdb; pdb.set_trace()
            c_loss.backward(retain_graph=True)
            critic.optimizer.step()

            a_loss = critic.forward(sample['global_states'], agents_actions_old_s.detach()).flatten()
            a_loss = (-1) * torch.mean(a_loss)
            actor.optimizer.zero_grad()
            a_loss.backward(retain_graph=True)
            actor.optimizer.step()

            agent.update_networks(primary=primary, tau=cnf['tau'])



    def perform(self, games=100, steps=None, buffer=None, no_pbar=False, train=True):
        '''
        Evaluate agent on the environemt

        @params:
            games:      Number of played games, has priority to steps (pass None if steps to be used)
            steps:      Number of steps to execute (Is used for training for exact step count gained)
            buffer:     Replay buffer instance to log transitions, None if not to log
        '''
        # What metric to use for loop termination
        use_games = games is not None
        pbar = tqdm(range(games if use_games else steps), ncols=100, desc='Training ...' if train else 'Evaluating ...', disable=no_pbar)

        # Initialize the counters
        cnf = self.cnf(primary=True)
        _games, _game_steps, _total_steps, _score, learned = 0, 0, 0, 0, 0

        # Start interaction
        done = False
        current_observations = self.env.reset()

        # The main loop
        ### While condition depends on whether we requested N steps or N games 
        while (use_games and _games < games) or (not use_games and _total_steps < steps):
            # 1) Choose the action given observation
            action_ids, action_vectors = self.choose_actions(current_observations, primary=True, noise=1)
            # 2) Perform chosen actions
            next_observations, reward, done, _ = self.env.step(action_ids)
            # 3) Update the step counter and cumulative score
            _game_steps += 1
            _score += reward.sum()
            ### Training only steps (buffer filling)
            if buffer is not None:
                # 4) Convert observations to global states
                if self.env.actual_agents != 1:
                    global_state = np.concatenate(current_observations)
                    global_next_state = np.concatenate(next_observations)
                else:
                    global_state = current_observations
                    global_next_state = next_observations
                # 5) Store the transition
                buffer.log_new_step(global_state, global_next_state, action_vectors, reward, current_observations,
                                    next_observations, done, float('inf'))  # -inf is default value, should be unique
            # 6) Check if maximal number of steps-per-game was exceeded, consider it finished if so
            if _game_steps >= self.env.max_steps:
                done = True

            # 7) Move on with the observations
            current_observations = next_observations

            # Update pbar if per step
            if not use_games:
                pbar.update(1)

            # 'Log' the game
            if done:
                # Reset the env (prepore for another game)
                current_observations = self.env.reset()
                # Keep track of total steps and games
                _total_steps += _game_steps
                _game_steps = 0
                _games += 1
                self.scores.append(_score)
                _score = 0
                done = False
                # Perform training if requested
                if train:
                    # TODO: Ratio of eval and training
                    while learned < _total_steps * cnf['batch']:
                        self.learn(buffer, primary=True)
                        learned += cnf['batch']
                # Update pbar if per game
                if use_games:
                    pbar.update(1)
                # If per game or step, here we update it's description
                pbar.set_description(('Training ...' if train else 'Evaluating ... ') + f'{sum(self.scores[-100:])/len(self.scores[-100:])}')
