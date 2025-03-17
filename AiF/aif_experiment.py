import os
from typing import Dict

import numpy as np
import pickle
import gym
import statistics
import time
import tracemalloc

from measurement.measurement import ExperimentResults

from copy import deepcopy

import pymdp.utils

from AiF.GenerativeModel import GenerativeModel
from AiF.GridWorldGP2D import GridWorldGP2D

def dependent_variable_set_up():



    pass

def aif_experiment_run(config_data: Dict):
    """
    Creates, and simulates active inference experiment
    :param config_data:
    :return: desired results.
    """

    my_agent = GenerativeModel(config_data)
    my_env = GridWorldGP2D(config_data)

    # experimental variables
    actions = deepcopy(config_data["agent"]["actions"]) + ["STAY"]

    rewards = config_data["environment"]["rewards"]
    # rewards = rewards["rewards"]
    reward_conditions = ["None"] + rewards["name"]

    reward_location = config_data["environment"]["rewards"]["position"]


    # Grid set up, states
    grid_dims = config_data["environment"]['grid_dimensions']
    num_grid_points = np.prod(grid_dims)
    grid = np.arange(num_grid_points).reshape(grid_dims)
    it = np.nditer(grid, flags=["multi_index"])
    loc_list = []
    while not it.finished:
        loc_list.append(it.multi_index)
        it.iternext()
    # loc_list = loc_list
    num_states = [num_grid_points, len(reward_conditions)]

    loc_obs, reward_obs = my_env.reset()
    history_of_locs = [loc_obs]
    obs = [loc_list.index(loc_obs), reward_conditions.index(reward_obs)]



    # experimental parameters
    experimental_params = config_data["experiment_parameters"]["global"]
    T = experimental_params["time_steps"]
    # gamma = experimental_params["gamma"]
    # alpha = experimental_params["alpha"]

    qs = None
    qs_prev = None

    # episode_count = 0

    start_time = time.time()
    tracemalloc.start()

    exp_agent = {"time_step_per_episode": [],
                 "episode_count": 0,
                 "peak_memory_per_episode": [],
                 "time_per_episode": [],
                 "policy_length_per_episode": [],
                 "state_space_coverage": []}

    for t in range(T):
        qs = my_agent.infer_states(obs)

        # Apply learning

        q_pi, G = my_agent.infer_policies()
        chosen_action_id = my_agent.sample_action()

        movement_id = int(chosen_action_id[0])

        choice_action = actions[movement_id]

        print(f'Action at time {t}: {choice_action}')

        loc_obs, reward_obs = my_env.step(choice_action)

        obs = [loc_list.index(loc_obs), reward_conditions.index(reward_obs)]

        history_of_locs.append(loc_obs)

        qs_prev = qs

        # if qs_prev is not None:
        #     obj_arr_obs = obs
        #     my_agent.update_A(obj_arr_obs)
        #     my_agent.update_B(qs_prev)
        #     # my_agent.update_D(qs_t0=None)

        print(f'Grid location at time {t}: {loc_obs}')


        print(f'Reward at time {t}: {reward_obs}')
        if reward_obs == "GOAL":
            # record results
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            exp_agent["episode_count"] += 1
            exp_agent["time_per_episode"].append((end_time - start_time))
            exp_agent["peak_memory_per_episode"].append(peak/1024)
            exp_agent["time_step_per_episode"].append((my_agent.curr_timestep))
            exp_agent["policy_length_per_episode"].append(len(my_agent.action_selection))
            space_coverage = len(set(history_of_locs))/len(loc_list)
            exp_agent["state_space_coverage"].append(space_coverage)

            # rest agent
            my_agent.reset()
            loc_obs, reward_obs = my_env.reset()
            history_of_locs = [loc_obs]
            obs = [loc_list.index(loc_obs), reward_conditions.index(reward_obs)]

            start_time = time.time()
            tracemalloc.reset_peak()

    # exp_agent["episode_count"] = statistics.mean(exp_agent["episode_count"])
    exp_agent["time_per_episode"] = statistics.mean(exp_agent["time_per_episode"])
    exp_agent["peak_memory_per_episode"] = statistics.mean(exp_agent["peak_memory_per_episode"])
    exp_agent["time_step_per_episode"] = statistics.mean(exp_agent["time_step_per_episode"])
    exp_agent["policy_length_per_episode"] = statistics.mean(exp_agent["policy_length_per_episode"])
    exp_agent["state_space_coverage"] = statistics.mean(exp_agent["state_space_coverage"])

    print(exp_agent)
    filename = os.path.join(os.getcwd(), "results", "exp_agent_date.pkl")
    with open(filename, "wb") as f:
        pickle.dump(exp_agent, f)



    print(my_agent)

    print(qs)
    print(my_agent.qs)
    my_env.render("title")

