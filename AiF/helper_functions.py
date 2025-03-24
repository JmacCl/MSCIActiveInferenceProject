import os

import numpy as np

from typing import List, Tuple, Dict


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


def dtmc_construction(locs, grid_dims):
    """
    This function will construct the dictionary that will be used to construct hte possible transitions that can be
    made per state, such that
    :return:
    """

    actions = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1), "STAY": (0, 0)}
    # locs, _ = define_grid_space(grid_dims)
    transition_count = {}

    for state in locs:
        transition_count[tuple(state)] = {}
        for action, val in actions.items():
            if grid_dims[0] > state[0] + val[0] >= 0 and grid_dims[1] > state[1] + val[1] >= 0:
                transition_count[state][(state[0] + val[0], state[1] + val[1])] = 0

    return transition_count


def formulate_dtmc(trajectory_count, locs):
    """
    Given the transition counts, determine the DTMC transitions
    :param trajectory_count:
    :return:
    """
    prism_dtmc = {}
    for state in trajectory_count:
        transitions_dict = trajectory_count[state]
        k = locs.index(state)
        prism_dtmc[k] = {}

        total_transition = sum(transitions_dict.values())
        for tup in transitions_dict:
            prism_dtmc[k][locs.index(tup)] = transitions_dict[tup] / total_transition

    return prism_dtmc

def create_prism_file(unique_states, dtmc):
    # Create PRISM file
    prism_filename = "dtmc_model.pm"

    cwd = os.getcwd()

    p = os.path.join(cwd, prism_filename)

    with open(p, "w") as f:
        f.write("dtmc\n\nmodule agent\n")
        f.write(f"    s : [0..{len(unique_states) - 1}] init 0;\n\n")

        # Convert transitions to PRISM format
        for from_state, to_states in  dtmc.items():
            from_idx = from_state
            transitions_str = " + ".join(
                [f"{prob:.6f} : (s'={to_state})" for to_state, prob in to_states.items()]
            )
            f.write(f"    [] s={from_idx} -> {transitions_str};\n")

        f.write("\nendmodule\n")

