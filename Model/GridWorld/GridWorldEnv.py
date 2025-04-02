from random import random
from typing import List, Tuple

import numpy as np

import random

from AiF.helper_functions import define_grid_space, set_up_boundary_modalities


def define_grid_space(grid_dims):
    """
    Defines the cartesian coordinates of the grid space
    :param grid_dims:
    :return:
    """
    num_grid_points = np.prod(grid_dims)
    grid = np.arange(num_grid_points).reshape(grid_dims)
    it = np.nditer(grid, flags=["multi_index"])
    loc_list = []
    while not it.finished:
        loc_list.append(it.multi_index)
        it.iternext()
    loc_list = loc_list

    return loc_list, num_grid_points


def define_boundary(grid_dims):
    """
    Given the grid spec, define the starting locations
    :param grid:
    :return:
    """
    outer_dims, _ = define_grid_space(grid_dims)

    inner_dims, _ = define_grid_space([grid_dims[0] - 1, grid_dims[1] - 1])

    new_inner_dims = []

    for tup in inner_dims:
        if not (tup[0] == 0 or tup[1] == 0):
            new_inner_dims.append(tup)

    output = list(set(outer_dims).symmetric_difference(set(new_inner_dims)))
    output.sort()

    return output


def set_up_boundary_modalities(free_space: List[Tuple[int, int]]):
    """
    Map each possible location with its boundary observation modality
    :param free_space: the list of grid locations where the agent can move
    :return: A dictionary that maps each location to its boundary observation
    """

    state_boundary_map = {}

    boundary_mod = {
        "None": 0,
        "Left_Wall": 1,
        "Right_Wall": 2,
        "Up_Wall": 3,
        "Bottom_Wall": 4,
        "Up_Left_Corner": 5,
        "Up_Right_Corner": 6,
        "Bottom_Left_Corner": 7,
        "Bottom_Right_Corner": 8,
        "Horizontal_Corider": 9,
        "Vertical_Corider": 10,
        "Top_Enclosure": 11,
        "Bottom_Enclosure": 12,
        "Left_Enclosure": 13,
        "Right_Enclosure": 14,
    }
    # Assumes for each environment that the outer boundaries are always free

    for y, x in free_space:
        up = (y - 1, x) in free_space
        down = (y + 1, x) in free_space
        left = (y, x - 1) in free_space
        right = (y, x + 1) in free_space

        if down and not (up or left or right):
            state_boundary_map[(y, x)] = boundary_mod["Top_Enclosure"]
        elif up and not (down or left or right):
            state_boundary_map[(y, x)] = boundary_mod["Bottom_Enclosure"]
        elif left and not (down or up or right):
            state_boundary_map[(y, x)] = boundary_mod["Right_Enclosure"]
        elif right and not (down or left or up):
            state_boundary_map[(y, x)] = boundary_mod["Left_Enclosure"]
        elif left and right and not (down or up):
            state_boundary_map[(y, x)] = boundary_mod["Horizontal_Corider"]
        elif up and down and not (left or right):
            state_boundary_map[(y, x)] = boundary_mod["Vertical_Corider"]
        elif not (up or left) and (down and right):
            state_boundary_map[(y, x)] = boundary_mod["Up_Left_Corner"]
        elif not (up or right) and (down and left):
            state_boundary_map[(y, x)] = boundary_mod["Up_Right_Corner"]
        elif not (down or right) and (up and left):
            state_boundary_map[(y, x)] = boundary_mod["Bottom_Right_Corner"]
        elif not (down or left) and (up and right):
            state_boundary_map[(y, x)] = boundary_mod["Bottom_Left_Corner"]
        elif not down and (up and left and right):
            state_boundary_map[(y, x)] = boundary_mod["Bottom_Wall"]
        elif not up and (down and left and right):
            state_boundary_map[(y, x)] = boundary_mod["Up_Wall"]
        elif not right and (down and left and up):
            state_boundary_map[(y, x)] = boundary_mod["Right_Wall"]
        elif not left and (down and right and up):
            state_boundary_map[(y, x)] = boundary_mod["Left_Wall"]
        else:
            state_boundary_map[(y, x)] = boundary_mod["None"]

    return state_boundary_map


ACTION_MAP = {
    "UP": lambda Y, X, shape: (max(Y - 1, 0), X),
    "DOWN": lambda Y, X, shape: (min(Y + 1, shape[0] - 1), X),
    "LEFT": lambda Y, X, shape: (Y, max(X - 1, 0)),
    "RIGHT": lambda Y, X, shape: (Y, min(X + 1, shape[1] - 1)),
    "STAY": lambda Y, X, shape: (Y, X)
}

def derive_coords(coordinates):
    """
    Given a list of lists, ensure that the input is a coordinate list
    :param coordinates:
    :return:
    """
    return [tuple(coord) for coord in coordinates]


class GridWorldEnv:

    def __init__(self, yaml_env: dict, init_state, verbosity=False):

        grid_dims = yaml_env['grid_dimensions']
        self.current_location = init_state

        free_space, _ = define_grid_space(grid_dims)

        obstacles = derive_coords(yaml_env["complexities"]["obstacles"])
        cues = yaml_env["complexities"]["cues"]

        if obstacles:
            self.loc_list = list(set(free_space).difference(set(obstacles)))
        else:
            self.loc_list = free_space
        self.loc_list.sort()

        self.boundary_locations = set_up_boundary_modalities(self.loc_list)

        #
        # self.cue2_loc_names = yaml_env["cue2_loc_names"]
        # self.cue1_loc = yaml_env["cue1_loc"]
        # self.cue2_name = yaml_env["cue2"]
        # self.cue2_names = yaml_env["cue2_name"]
        # self.cue2_loc = yaml_env["cue2_name"][self.cue2_loc_names.index(self.cue2_name)]

        rewards = yaml_env["terminal_states"]
        self.reward_conditions = ["None"] + list(rewards.keys())
        self.reward_locations = rewards
        self.goal_location = tuple(self.reward_locations["Goal"])

        if self.goal_location in obstacles:
            raise IOError("Goal Location is in defined Obstacles")

        if self.reward_locations["Trap"] != None:
            self.trap_location = tuple(self.reward_locations["Trap"])
        else:
            self.trap_location = None

        if verbosity:
            self.verbosity = True
        else:
            self.verbosity = False



        self.shape = grid_dims
        if init_state != None:
            init_state = tuple(init_state)
            self.start_state = init_state
        else:
            self.start_state = random.choice(self.loc_list)

    def step(self, action_label):

        (Y, X) = self.current_location

        # # action movement
        # if action_label == "UP":
        #
        #     Y_new = Y - 1 if Y > 0 else Y
        #     X_new = X
        #
        # elif action_label == "DOWN":
        #
        #     Y_new = Y + 1 if Y < (self.shape[0] - 1) else Y
        #     X_new = X
        #
        # elif action_label == "LEFT":
        #     Y_new = Y
        #     X_new = X - 1 if X > 0 else X
        #
        # elif action_label == "RIGHT":
        #     Y_new = Y
        #     X_new = X + 1 if X < (self.shape[1] - 1) else X
        #
        # elif action_label == "STAY":
        #     Y_new, X_new = Y, X

        Y_new, X_new = ACTION_MAP[action_label](Y, X, self.shape)

        loc_obs = (Y_new, X_new)
        if loc_obs not in self.boundary_locations:
            loc_obs = (Y, X)
            if self.verbosity:
                print(f"Hit boundary location at {(Y_new, X_new)}!, lets move back to {loc_obs}")


        self.current_location = loc_obs  # store the new grid location

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
        elif self.trap_location and self.current_location == self.trap_location:
            reward_obs = 'Trap'
        else:
            reward_obs = "None"

        return loc_obs, bound_obs, reward_obs

    def reset(self, start_state=None):
        if start_state:
            self.start_state = start_state
        else:
            self.start_state = random.choice(self.loc_list)
        self.current_location = self.start_state
        if self.verbosity:
            print(f'Re-initialized location to {self.current_location}')
            print("\n")
        loc_obs = self.start_state
        # cue1_obs = 'Null'
        # cue2_obs = 'Null'
        bound_obs = loc_obs


        if self.current_location == tuple(self.reward_locations["Goal"]):
            reward_obs = 'Goal'
        elif self.trap_location != None:
            if self.current_location == tuple(self.reward_locations["Trap"]):
                reward_obs = 'Trap'
        else:
            reward_obs = "None"

        self.obs = [loc_obs, bound_obs, reward_obs]

        return loc_obs, bound_obs, reward_obs
