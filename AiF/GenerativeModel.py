import os
import sys
import pathlib
import numpy as np
import random
from copy import deepcopy

from AiF.helper_functions import *
from pymdp.agent import Agent
from pymdp import utils, maths

"""
Gen model is the process for which the agent derives learning a given gen process, 
i.e how it learns to navigate the environment of th external given its means that it is offered

in a sense it represents the agent own model 90either by brain or some other mechanisms) 
to which it understands how things work,the development of the mathematical process

boundary specificaiton
"""

BOUNDARY_MOD = {0: "None", 1: "Left_Wall", 2: "Right_Wall",
                3: "Up_Wall", 4: "Bottom_Wall", 5: "Up_Left_Corner",
                6: "Up_Right_Corner", 7: "Bottom_Left_Corner",
                8: "Bottom_Right_Corner"}

def extract_reward_information(yaml_env):
    """
    Given the taml configuration format, extract the reward infromatino to be used
    :param yaml_env:
    :return:
    """
    terminal_state_config = yaml_env["environment"]["terminal_states"]

    return terminal_state_config

def extract_cue_information():
    pass

def extract_boundary_information():
    pass

class GenerativeModel(Agent):

    def __base_setup__(self, yaml_env):
        # derive specific agent configurations
        agent_factor_config = yaml_env["agent"]

        # Set up base state factor information
        env = yaml_env["environment"]
        self.grid_dims = env['grid_dimensions']
        self.loc_list, self.num_grid_points = define_grid_space(self.grid_dims)
        self.factors = [self.num_grid_points]
        self.start_location = random.choice(self.loc_list)



        # controls
        self.controls = deepcopy(yaml_env["agent"]["actions"]) + ["STAY"]
        self.num_controls = [len(self.controls)]

        # Observations/modalities
        # update boundary modalities for grid boundaries
        self.boundaries = define_boundary(self.grid_dims)
        self.state_boundaries = set_up_boundary_modalities(self.loc_list)
        num_outer = len(self.boundaries)
        # update boundary modalities for maze/obstacles
        self.modalities = [self.num_grid_points, 8 + 1]



    def __init__(self, yaml_env):

        self.__base_setup__(yaml_env)

        self.extra_model_info = {}

        self.define_agent_complexity(yaml_env["agent"]["properties"])
        self.__set_terminal_state_information(yaml_env)
        # self.__cue_information()
        # self.__obstalce


        # rewards = terminal_states["Goal"]
        # reward_conditions = [rewards["name"]]
        # self.reward_locations = [tuple(rewards["position"])]
        # # if there is only one reward, do not make separate factor and modality
        # if len(terminal_states) > 1:
        #     traps = terminal_states["Trap"]
        #     reward_conditions += traps["name"]
        #     reward_conditions += None
        #     trap_position = terminal_states["Trap"]
        #     trap_value = traps["value"]

        # cues
        # self.cue2_loc_names = yaml_env["cue2_loc_names"]
        # self.cue1_loc = yaml_env["cue1_loc"]
        # self.cue2_name = yaml_env["cue2"]
        # self.cue2_names = yaml_env["cue2_name"]
        # self.cue2_loc = yaml_env["cue2_name"][self.cue2_loc_names.index(self.cue2_name)]
        # self.cue2_locations = yaml_env["cu2_locations"]

        A, pA = self.__set_up_observation_model()
        B, pB = self.__set_up_transition_model()
        C = self.__set_up_reward()
        D, pD = self.__set_up_prior()

        agent_params = yaml_env["experiment_parameters"]["AiF"]

        super().__init__(A=A, B=B, C=C, D=D, policy_len=agent_params["policy_len"],
                         save_belief_hist=True,
                         gamma=agent_params["gamma"], alpha=agent_params["alpha"],
                         use_states_info_gain=bool(agent_params["use_states_info_gain"]),
                         use_utility=bool(agent_params["use_utility"]),
                         action_selection=agent_params["agent_selection"],
                         sampling_mode=agent_params["sampling_mode"],
                         use_BMA=agent_params["use_BMA"],
                         policy_sep_prior=agent_params["policy_sep_prior"],
                         use_param_info_gain=agent_params["use_param_info_gain"],
                         pA=pA, pD=pD)

    def __set_terminal_state_information(self, yaml_env):
        """
        Given the yaml specification, update the factor and moaldities around what
        is specified in the rewards

        :return:
        """
        # Reward Information
        self.terminal_information = extract_reward_information(yaml_env)
        self.reward_value = 1

        self.extra_model_info["Terminal"] = {"modality": len(self.modalities),
                                                 "factor": len(self.factors),}
        reward_num = 0
        for k, v in self.terminal_information.items():
            if v != 'None':
                reward_num += 1
        self.factors.append(len(self.terminal_information))
        self.num_controls.append(1)
        self.modalities.append(len(self.terminal_information) + 1)

    def __set_cue_information(self):
        pass

    def define_agent_complexity(self, agent_properties):
        """
        For the given environment, this method implement the defined complexity
        for the Active Inference model
        :return:
        """

        self.observation_offset = agent_properties["observation_offset"]
        self.transition_offset = agent_properties["transition_offset"]

    def return_gen_model(self):
        return self.A, self.B, self.C, self.D

    def __set_up_simple_observation_model(self):
        A = np.eye(self.modalities[0], self.factors[0])
        return A


    def __set_up_observation_model(self):

        # Define Observation Model P(o|s) shape
        A_m_shapes = [[o_dim] + self.factors for o_dim in self.modalities]  # list of shapes of modality-specific A[m] arrays
        A = utils.obj_array_zeros(A_m_shapes)


        # Define state grid information modalities
        distribution = 1 - self.observation_offset
        if distribution == 0:
            dis_pint = self.observation_offset / (A_m_shapes[0][1])
            arr = np.full(A_m_shapes[0][:2], dis_pint)
        else:
            dis_pint = self.observation_offset/(A_m_shapes[0][1] - 1)
            arr = np.full(A_m_shapes[0][:2], dis_pint)
            np.fill_diagonal(arr, distribution)
        A[0][..., 0] = arr
        A[0][..., 1] = np.eye(self.modalities[0], self.factors[0])

        # Define boundary observations for each grid position

        for x in range(self.grid_dims[0]):
            for y in range(self.grid_dims[1]):
                coord = (y, x)
                state_index = self.loc_list.index((y, x))
                bound_index = self.state_boundaries[coord]
                A[1][bound_index, state_index, 0] = 1
                # if coord in self.state_boundaries:
                #     bound_index = self.boundaries.index((x, y))
                #     A[1][bound_index + 1, state_index, 0] = 1
                # else:
                #     A[1][0, state_index, 0] = 1

        goal_index = self.loc_list.index(tuple(self.terminal_information["Goal"]))
        trap_index = self.loc_list.index(tuple(self.terminal_information["Trap"]))

        A[1][0, :,  1] = 1
        A[1][0, goal_index,  1] = 1
        A[1][0, trap_index,  1] = 1



        A[2][0,] = 1.0  # default makes Null the most likely observation everywhere
        goal_index = self.loc_list.index(tuple(self.terminal_information["Goal"]))
        trap_index = self.loc_list.index(tuple(self.terminal_information["Trap"]))
        A[2][0, goal_index, :] = 0.0
        A[2][1, goal_index,  0] = 1.0
        A[2][2, goal_index, 1] = 1.0

        # fill out the contingencies when the agent is in the "BOTTOM" reward location
        # A[2][0, trap_index, :] = 0.0
        # A[2][1, trap_index, 0] = 1.0
        # A[2][2, trap_index, 1] = 1.0


        # grid states
        # outer boundaries
        # obstacles/walls
        # cues
        #
        # Define terminal state modalities

        # if a pomdp, agent must learn stuff
        if self.observation_offset:
            pA = utils.dirichlet_like(A, scale=1)
        else:
            pA = None




        return A, pA

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

        B[1][:, :, 0] = np.eye(self.factors[1], self.factors[1])

        pB = None  # utils.obj_array_ones([[ns, ns, self.num_controls[f]] for f, ns in enumerate(self.factors)])

        return B, pB

    def __set_up_reward(self):
        C = utils.obj_array_zeros(self.modalities)

        # desired_loc_index = self.loc_list.index(self.reward_locations)


        C[2][1] = 2
        C[2][2] = -4

        # print(C.shape)
        # print(C[0].shape)
        # print(C[1].shape)
        #
        # C[1][1] = self.reward_value  # make the agent want to encounter the "Cheese" observation level
        return C

    def __set_up_prior(self):

        D = utils.obj_array_uniform(self.factors)
        D[0] = utils.onehot(self.loc_list.index(self.start_location), self.num_grid_points)

        # random location is always renadmoly located
        pD = utils.dirichlet_like(D, scale=1)

        return D, pD
