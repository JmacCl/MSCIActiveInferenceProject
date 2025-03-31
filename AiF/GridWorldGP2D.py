import random

import numpy as np
from pymdp.envs import GridWorldEnv
from AiF.helper_functions import *
#
# cue1_location = (2, 0)
#
# cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
# cue2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]
#
# # names of the reward conditions and their locations
# reward_conditions = ["TOP", "BOTTOM"]
# reward_locations = [(1, 5), (3, 5)]
#
# grid_dims = [5, 7] # dimensions of the grid (number of rows, number of columns)
# num_grid_points = np.prod(grid_dims) # total number of grid locations (rows X columns)
#
# # list of dimensionalities of the hidden states -- useful for creating generative model later on
# num_states = [num_grid_points, len(cue2_locations), len(reward_conditions)]
#
# # Names of the cue1 observation levels, the cue2 observation levels, and the reward observation levels
# cue1_names = ['Null'] + cue2_loc_names # signals for the possible Cue 2 locations, that only are seen when agent is visiting Cue 1
# cue2_names = ['Null', 'reward_on_top', 'reward_on_bottom']
# reward_names = ['Null', 'Cheese', 'Shock']

"""
Gen process is the means to which the agent must figure out, how it can act, how it can sense
its the actual environment of the hidden states, but also the actual actions and senses 

in a sense it is what is possible
"""


class GridWorldGP2D:

    def __init__(self, yaml_env: dict, trap: bool):
        # super().__init__()
        # self.cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
        # self.cue1_loc = cue1_loc
        # self.cue2_name = cue2
        # self.cue2_loc = cue2_locations[self.cue2_loc_names.index(self.cue2_name)]
        #
        # self.reward_condition = reward_condition
        grid_dims = yaml_env["environment"]['grid_dimensions']
        init_state = tuple(yaml_env["agent"]['initial_position'])
        self.current_location = init_state
        self.trap = trap

        self.starting_locations, _ = define_grid_space(grid_dims)

        self.boundary_locations = set_up_boundary_modalities(self.starting_locations)

        #
        # self.cue2_loc_names = yaml_env["cue2_loc_names"]
        # self.cue1_loc = yaml_env["cue1_loc"]
        # self.cue2_name = yaml_env["cue2"]
        # self.cue2_names = yaml_env["cue2_name"]
        # self.cue2_loc = yaml_env["cue2_name"][self.cue2_loc_names.index(self.cue2_name)]

        rewards = yaml_env["environment"]["terminal_states"]
        self.reward_conditions = list(rewards.keys())
        self.reward_locations = rewards
        self.start_state = init_state
        self.shape = grid_dims



    def step(self, action_label):

        (Y, X) = self.current_location

        # action movement
        if action_label == "UP":

            Y_new = Y - 1 if Y > 0 else Y
            X_new = X

        elif action_label == "DOWN":

            Y_new = Y + 1 if Y < (self.shape[0] - 1) else Y
            X_new = X

        elif action_label == "LEFT":
            Y_new = Y
            X_new = X - 1 if X > 0 else X

        elif action_label == "RIGHT":
            Y_new = Y
            X_new = X + 1 if X < (self.shape[1] - 1) else X

        elif action_label == "STAY":
            Y_new, X_new = Y, X

        loc_obs = (Y_new, X_new)

        self.current_location = loc_obs # store the new grid location

         # self.init_state  # agent always directly observes the grid location they're in

        # bounary stuff

        bound_obs = loc_obs

        ## Cue stuff
        # if self.init_state == self.cue1_loc:
        #     cue1_obs = self.cue2_name
        # else:
        #     cue1_obs = 'Null'
        #
        # if self.init_state == self.cue2_loc:
        #     cue2_obs = self.cue2_names[self.reward_conditions.index(self.reward) + 1]
        # else:
        #     cue2_obs = 'Null'

        # Reward Stuff
        if self.current_location == tuple(self.reward_locations["Goal"]):
            reward_obs = 'Goal'
        elif self.trap and self.current_location == tuple(self.reward_locations["Trap"]):
            reward_obs = 'Trap'
        else:
            reward_obs = "None"

        return loc_obs, bound_obs,  reward_obs



    def reset(self, start_state=None):
        if start_state:
            self.start_state = start_state
        else:
            self.start_state = random.choice(self.starting_locations)
        self.current_location = self.start_state
        #print(f'Re-initialized location to {self.current_location}')
        loc_obs = self.start_state
        # cue1_obs = 'Null'
        # cue2_obs = 'Null'
        bound_obs = loc_obs

        if self.current_location == tuple(self.reward_locations["Goal"]):
            reward_obs = 'Goal'
        elif self.trap and self.current_location == tuple(self.reward_locations["Trap"]):
            reward_obs = 'Trap'
        else:
            reward_obs = "None"

        return loc_obs, bound_obs, reward_obs
