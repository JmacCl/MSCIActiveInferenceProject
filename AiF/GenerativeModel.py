import os
import sys
import pathlib
import numpy as np
from copy import deepcopy

from AiF.helper_functions import *
from pymdp.agent import Agent
from pymdp import utils, maths

"""
Gen model is the process for which the agent derives learning a given gen process, 
i.e how it learns to navigate the environment of th external given its means that it is offered

in a sense it represents the agent own model 90either by brain or some other mechanisms) 
to which it understands how things work,the development of the mathematical process
"""


# grid_dims = [5, 7]  # dimensions of the grid (number of rows, number of columns)
# num_grid_points = np.prod(grid_dims)  # total number of grid locations (rows X columns)
#
# # create a look-up table `loc_list` that maps linear indices to tuples of (y, x) coordinates
# grid = np.arange(num_grid_points).reshape(grid_dims)
# it = np.nditer(grid, flags=["multi_index"])
#
# loc_list = []
# while not it.finished:
#     loc_list.append(it.multi_index)
#     it.iternext()
#
# # (y, x) coordinate of the first cue's location, and then a list of the (y, x) coordinates of the possible locations of the second cue, and their labels (`L1`, `L2`, ...)
# cue1_location = (2, 0)
#
# cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
# cue2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]
#
# # names of the reward conditions and their locations
# reward_conditions = ["TOP", "BOTTOM"]
# reward_locations = [(1, 5), (3, 5)]
#
# # list of dimensionalities of the hidden states -- useful for creating generative model later on
# num_states = [num_grid_points, len(cue2_locations), len(reward_conditions)]
#
# # Names of the cue1 observation levels, the cue2 observation levels, and the reward observation levels
# cue1_names = [
#                  'Null'] + cue2_loc_names  # signals for the possible Cue 2 locations, that only are seen when agent is visiting Cue 1
# cue2_names = ['Null', 'reward_on_top', 'reward_on_bottom']
# reward_names = ['Null', 'Cheese', 'Shock']
#
# num_obs = [num_grid_points, len(cue1_names), len(cue2_names), len(reward_names)]
#
# # initialize `num_controls`
# num_controls = [5, 1, 1]
# actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class GenerativeModel(Agent):

    def __init__(self, yaml_env):

        # yaml_env = deepcopy(yaml_env)

        # Reward Information
        rewards = yaml_env["environment"]["rewards"]
        self.reward_conditions = rewards["name"]
        self.reward_locations = tuple(rewards["position"])
        self.reward_value = rewards["value"]

        # Factor Dimensions set up
        env = yaml_env["environment"]
        self.grid_dims = env['grid_dimensions']
        self.loc_list, self.num_grid_points = define_grid_space(self.grid_dims)
        self.factors = [self.num_grid_points]


        # factor specification


        # cues
        # self.cue2_loc_names = yaml_env["cue2_loc_names"]
        # self.cue1_loc = yaml_env["cue1_loc"]
        # self.cue2_name = yaml_env["cue2"]
        # self.cue2_names = yaml_env["cue2_name"]
        # self.cue2_loc = yaml_env["cue2_name"][self.cue2_loc_names.index(self.cue2_name)]
        # self.cue2_locations = yaml_env["cu2_locations"]

        # controls
        self.controls = deepcopy(yaml_env["agent"]["actions"]) + ["STAY"]
        self.num_controls = [len(self.controls), len(self.controls)]

        # Observations/modalities
        self.reward_conditions = ["None"] + self.reward_conditions
        self.boundaries = define_boundary(self.grid_dims)
        num_outer = len(self.boundaries)
        self.modalities = [self.num_grid_points, num_outer + 1]

        A = self.__set_up_observation_model()
        B = self.__set_up_transition_model()
        C = self.__set_up_reward()
        D = self.__set_up_prior()

        agent_params = yaml_env["experiment_parameters"]["AiF"]

        self.learning = True
        if self.learning:
            pA, pB, pD = self.__set_up_learning_models(A, B)
        else:
            pA, pB, pD = None, None, None

        super().__init__(A=A, B=B, C=C, D=D, policy_len=agent_params["policy_len"],
                         save_belief_hist=True,
                         gamma=agent_params["gamma"], alpha=agent_params["alpha"],
                         use_states_info_gain=bool(agent_params["use_states_info_gain"]),
                         use_utility=bool(agent_params["use_utility"]),
                         action_selection=agent_params["agent_selection"], pD=pD)


    def __set_up_learning_models(self, A, B):

        pA = utils.to_obj_array(np.ones_like(A))
        pB = None # utils.obj_array_ones([[ns, ns, self.num_controls[f]] for f, ns in enumerate(self.factors)])
        pD = utils.obj_array_ones(self.factors)

        return pA, pB, pD
    def return_gen_model(self):
        return self.A, self.B, self.C, self.D

    def __set_up_simple_observation_model(self):
        A = np.eye(self.modalities[0], self.factors[0])
        return A


    def __set_up_observation_model(self):

        A_m_shapes = [[o_dim] + self.factors for o_dim in self.modalities]  # list of shapes of modality-specific A[m] arrays
        A = utils.obj_array_zeros(A_m_shapes)

        # makde MDP

        # A[0] = np.eye(A_m_shapes[0])

        # add blur
        offset = 1

        distribution = 1 - offset
        if distribution == 0:
            dis_pint = offset / (A_m_shapes[0][1])
            arr = np.full(A_m_shapes[0], dis_pint)
        else:
            dis_pint = offset /( A_m_shapes[0][1] - 1)
            arr = np.full(A_m_shapes[0], dis_pint)
            np.fill_diagonal(arr, distribution)
        A[0] = arr

        # make completely blurtransition model



        # deal with ran dom sitribution
        # random_matrix = utils.norm_dist(np.random.rand(self.num_grid_points, self.num_grid_points))
        #
        # # Normalize each row to ensure they sum to 1
        #
        # # General observation
        # A[0] = random_matrix

        # gridbounary observation


        # Define boundary observations for each grid position
        for x in range(self.grid_dims[0]):
            for y in range(self.grid_dims[1]):
                coord = (x, y)
                state_index = self.loc_list.index((x, y))
                if coord in self.boundaries:
                    bound_index = self.boundaries.index((x, y))
                    A[1][bound_index + 1, state_index] = 1
                else:
                    A[1][0, state_index] = 1


        # A = np.eye(self.modalities[0], self.factors[0])

        # grid states
        # outer boundaries
        # obstacles/walls
        # cues
        #
        # rewards (optional false rewards)



        # self.A[0] = np.tile(np.expand_dims(np.eye(num_grid_points), (-2, -1)),(1, 1, self.num_states[1], self.num_states[2]))

        # A[0] = np.tile(
        #     np.expand_dims(np.eye(self.num_grid_points), -1),
        #     (1, 1, self.num_states[1])
        # )

        # # make the cue1 observation depend on the location (being at cue1_location) and the true location of cue2
        # A[1][0, :, :, :] = 1.0  # default makes Null the most likely observation everywhere
        #
        # # Make the Cue 1 signal depend on 1) being at the Cue 1 location and 2) the location of Cue 2
        # for i, cue_loc2_i in enumerate(self.cue2_loc):
        #     A[1][0, self.loc_list.index(self.cue1_loc), i, :] = 0.0
        #     A[1][i + 1, self.loc_list.index(self.cue1_loc), i, :] = 1.0
        #
        # # make the cue2 observation depend on the location (being at the correct cue2_location) and the reward condition
        # A[2][0, :, :, :] = 1.0  # default makes Null the most likely observation everywhere
        #
        # for i, cue_loc2_i in enumerate(self.cue2_locations):
        #     # if the cue2-location is the one you're currently at, then you get a signal about where the reward is
        #     [2][0, self.loc_list.index(cue_loc2_i), i, :] = 0.0
        #     A[2][1, self.loc_list.index(cue_loc2_i), i, 0] = 1.0
        #     A[2][2, self.loc_list.index(cue_loc2_i), i, 1] = 1.0

        # # make the reward observation depend on the location (being at reward location) and the reward condition
        # A[1][0, :, :] = 1.0  # default makes Null the most likely observation everywhere
        #
        # rew_idx = self.loc_list.index(
        #     self.reward_locations)  # linear index of the location of the "BOTTOM" reward location
        #
        # # # fill out the contingencies when the agent is in the "TOP" reward location
        # # A[1][0, rew_top_idx, :, :] = 0.0
        # # A[1][1, rew_top_idx, :, 0] = 1.0
        # # A[1][2, rew_top_idx, :, 1] = 1.0
        #
        # # fill out the contingencies when the agent is in the "BOTTOM" reward location
        # A[1][0, rew_idx, :] = 0.0
        # A[1][1, rew_idx, :] = 1

        return A

    def __set_up_transition_model(self):

        B = utils.obj_array_zeros([[ns, ns, self.num_controls[f]] for f, ns in enumerate(self.factors)])

        # fill out `B[0]` using the
        for action_id, action_label in enumerate(self.controls):

            for curr_state, grid_location in enumerate(self.loc_list):

                y, x = grid_location

                if action_label == "UP":
                    next_y = y - 1 if y > 0 else y
                    next_x = x
                elif action_label == "DOWN":
                    next_y = y + 1 if y < (self.grid_dims[0] - 1) else y
                    next_x = x
                elif action_label == "LEFT":
                    next_x = x - 1 if x > 0 else x
                    next_y = y
                elif action_label == "RIGHT":
                    next_x = x + 1 if x < (self.grid_dims[1] - 1) else x
                    next_y = y
                elif action_label == "STAY":
                    next_x = x
                    next_y = y

                new_location = (next_y, next_x)
                next_state = self.loc_list.index(new_location)
                B[0][next_state, curr_state, action_id] = 1.0

        # B[1][:, :, 0] = np.eye(self.num_states[1])

        return B

    def __set_up_reward(self):
        C = utils.obj_array_zeros(self.modalities)

        desired_loc_index = self.loc_list.index(self.reward_locations)

        C[0][desired_loc_index] = 1.0

        # print(C.shape)
        # print(C[0].shape)
        # print(C[1].shape)
        #
        # C[1][1] = self.reward_value  # make the agent want to encounter the "Cheese" observation level
        return C

    def __set_up_prior(self):

        D = utils.obj_array_uniform(self.factors)
        D[0] = utils.onehot(self.loc_list.index((0, 0)), self.num_grid_points)

        return D
