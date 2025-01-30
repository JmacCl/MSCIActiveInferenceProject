import os
import sys
import pathlib
import numpy as np

from pymdp.agent import Agent
from pymdp import utils, maths

grid_dims = [5, 7] # dimensions of the grid (number of rows, number of columns)
num_grid_points = np.prod(grid_dims) # total number of grid locations (rows X columns)

# create a look-up table `loc_list` that maps linear indices to tuples of (y, x) coordinates
grid = np.arange(num_grid_points).reshape(grid_dims)
it = np.nditer(grid, flags=["multi_index"])

loc_list = []
while not it.finished:
    loc_list.append(it.multi_index)
    it.iternext()

# (y, x) coordinate of the first cue's location, and then a list of the (y, x) coordinates of the possible locations of the second cue, and their labels (`L1`, `L2`, ...)
cue1_location = (2, 0)

cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
cue2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]

# names of the reward conditions and their locations
reward_conditions = ["TOP", "BOTTOM"]
reward_locations = [(1, 5), (3, 5)]

# list of dimensionalities of the hidden states -- useful for creating generative model later on
num_states = [num_grid_points, len(cue2_locations), len(reward_conditions)]

# Names of the cue1 observation levels, the cue2 observation levels, and the reward observation levels
cue1_names = ['Null'] + cue2_loc_names # signals for the possible Cue 2 locations, that only are seen when agent is visiting Cue 1
cue2_names = ['Null', 'reward_on_top', 'reward_on_bottom']
reward_names = ['Null', 'Cheese', 'Shock']

num_obs = [num_grid_points, len(cue1_names), len(cue2_names), len(reward_names)]

# initialize `num_controls`
num_controls = [5, 1, 1]
actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

class GenerativeModel:

    def __init__(self, num_states, num_obs, num_controls):
        self.num_states = num_states
        self.num_obs = num_obs
        self.num_actions = num_controls
        self.A = utils.obj_array_zeros([ [o_dim] + num_states for o_dim in num_obs])
        self.B = utils.obj_array_zeros([ [ns, ns, num_controls[f]] for f, ns in enumerate(num_states)])
        self.C = utils.obj_array_zeros(num_obs)
        self.D = utils.obj_array_uniform(num_states)
        self.__construct_generative_model()

    def return_gen_model(self):
        return (self.A, self.B, self.C, self.D)

    def __set_up_observation_model(self):
        self.A[0] = np.tile(np.expand_dims(np.eye(num_grid_points), (-2, -1)), (1, 1, self.num_states[1], self.num_states[2]))

        # make the cue1 observation depend on the location (being at cue1_location) and the true location of cue2
        self.A[1][0, :, :, :] = 1.0  # default makes Null the most likely observation everywhere

        # Make the Cue 1 signal depend on 1) being at the Cue 1 location and 2) the location of Cue 2
        for i, cue_loc2_i in enumerate(cue2_locations):
            self.A[1][0, loc_list.index(cue1_location), i, :] = 0.0
            self.A[1][i + 1, loc_list.index(cue1_location), i, :] = 1.0

        # make the cue2 observation depend on the location (being at the correct cue2_location) and the reward condition
        self.A[2][0, :, :, :] = 1.0  # default makes Null the most likely observation everywhere

        for i, cue_loc2_i in enumerate(cue2_locations):
            # if the cue2-location is the one you're currently at, then you get a signal about where the reward is
            self.A[2][0, loc_list.index(cue_loc2_i), i, :] = 0.0
            self.A[2][1, loc_list.index(cue_loc2_i), i, 0] = 1.0
            self.A[2][2, loc_list.index(cue_loc2_i), i, 1] = 1.0

        # make the reward observation depend on the location (being at reward location) and the reward condition
        self.A[3][0, :, :, :] = 1.0  # default makes Null the most likely observation everywhere

        rew_top_idx = loc_list.index(reward_locations[0])  # linear index of the location of the "TOP" reward location
        rew_bott_idx = loc_list.index(
            reward_locations[1])  # linear index of the location of the "BOTTOM" reward location

        # fill out the contingencies when the agent is in the "TOP" reward location
        self.A[3][0, rew_top_idx, :, :] = 0.0
        self.A[3][1, rew_top_idx, :, 0] = 1.0
        self.A[3][2, rew_top_idx, :, 1] = 1.0

        # fill out the contingencies when the agent is in the "BOTTOM" reward location
        self.A[3][0, rew_bott_idx, :, :] = 0.0
        self.A[3][1, rew_bott_idx, :, 1] = 1.0
        self.A[3][2, rew_bott_idx, :, 0] = 1.0

    def __set_up_transition_model(self):
        # fill out `B[0]` using the
        for action_id, action_label in enumerate(actions):

            for curr_state, grid_location in enumerate(loc_list):

                y, x = grid_location

                if action_label == "UP":
                    next_y = y - 1 if y > 0 else y
                    next_x = x
                elif action_label == "DOWN":
                    next_y = y + 1 if y < (grid_dims[0] - 1) else y
                    next_x = x
                elif action_label == "LEFT":
                    next_x = x - 1 if x > 0 else x
                    next_y = y
                elif action_label == "RIGHT":
                    next_x = x + 1 if x < (grid_dims[1] - 1) else x
                    next_y = y
                elif action_label == "STAY":
                    next_x = x
                    next_y = y

                new_location = (next_y, next_x)
                next_state = loc_list.index(new_location)
                self.B[0][next_state, curr_state, action_id] = 1.0

        self.B[1][:, :, 0] = np.eye(num_states[1])
        self.B[2][:, :, 0] = np.eye(num_states[2])

    def __set_up_reward(self):
        self.C[3][1] = 2.0  # make the agent want to encounter the "Cheese" observation level
        self.C[3][2] = -4.0  # make the agent not want to encounter the "Shock" observation level

    def __set_up_prior(self):
        self.D[0] = utils.onehot(loc_list.index((0, 0)), num_grid_points)
        
    def __construct_generative_model(self):
        self.__set_up_observation_model()
        self.__set_up_transition_model()
        self.__set_up_reward()
        self.__set_up_prior()

