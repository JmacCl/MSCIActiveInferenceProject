import os
from typing import Dict

import numpy as np

import statistics
import time
import tracemalloc

import yaml


from copy import deepcopy


from AiF.GenerativeModel import GenerativeModel
from AiF.GridWorldGP2D import GridWorldGP2D
from AiF.helper_functions import (define_grid_space, define_boundary, dtmc_construction, formulate_dtmc,
                                  create_prism_file, set_up_boundary_modalities)

def start_space_policy(loc_list):
    return_dict = {}
    for loc in loc_list:
        return_dict[loc] = {}
    return return_dict


def aif_experiment_run(config_data: Dict, experiment_config):
    """
    Creates, and simulates active inference experiment
    :param config_data:
    :return: desired results.
    """


    # Define the agent
    my_agent = GenerativeModel(config_data, experiment_config)
    my_env = GridWorldGP2D(config_data)

    # experimental variables
    actions = deepcopy(config_data["agent"]["actions"]) + ["STAY"]

    rewards = config_data["environment"]["terminal_states"]
    # rewards = rewards["rewards"]
    reward_conditions = ["None"] + list(rewards.keys())

    reward_location = rewards["Goal"]
    # trap = rewards["Trap"]


    # Grid set up, states
    grid_dims = config_data["environment"]['grid_dimensions']
    loc_list, num_grid_points = define_grid_space(grid_dims)
    bound_list = set_up_boundary_modalities(loc_list)

    # loc_list = loc_list

    loc_obs, bound_obs, reward_obs = my_env.reset(start_state=my_agent.start_location)
    history_of_locs = [loc_obs]
    obs = [loc_list.index(loc_obs), bound_list[bound_obs], reward_conditions.index(reward_obs)]



    # experimental parameters
    experimental_params = config_data["experiment_parameters"]["global"]
    T = experiment_config["time_steps"]

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
                 "state_space_coverage": [],
                 "trap_arrival": 0,
                 "goal_arrival": 0,
                 "belief_in_target": [],

                 }

    # "difference_in_goal_trap": [],
    # "policy_use_per_start": start_space_policy(loc_list)

    # construct the

    verbosity = bool(config_data['verbosity'])

    trans_count = dtmc_construction(loc_list, grid_dims)
    transition_i = 1
    for t in range(T):
        qs = my_agent.infer_states(obs)

        # Apply learning

        q_pi, G = my_agent.infer_policies()
        chosen_action_id = my_agent.sample_action()

        movement_id = int(chosen_action_id[0])

        choice_action = actions[movement_id]


        loc_obs, bound_obs, reward_obs = my_env.step(choice_action)

        obs = [loc_list.index(loc_obs), bound_list[bound_obs],
               reward_conditions.index(reward_obs)]

        history_of_locs.append(loc_obs)

        next, prev = history_of_locs[my_agent.curr_timestep], history_of_locs[my_agent.curr_timestep - 1]
        trans_count[prev][next] += 1
        # transition_i += 1

        if verbosity:
            print(f'Action at time {t}: {choice_action}')
            print(f'Grid location at time {t}: {loc_obs}')
            print(f'Reward at time {t}: {reward_obs}')

        # Implement Learning
        obj_arr_obs = obs
        my_agent.update_A(obj_arr_obs)
        # my_agent.update_B(qs_prev)
        my_agent.update_D(qs_t0=None)


        if reward_obs == "Goal":
            # record results
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            exp_agent["episode_count"] += 1
            exp_agent["time_per_episode"].append((end_time - start_time))
            exp_agent["peak_memory_per_episode"].append(peak/1024)
            exp_agent["time_step_per_episode"].append(my_agent.curr_timestep)
            exp_agent["policy_length_per_episode"].append(len(history_of_locs) - 1)
            space_coverage = len(set(history_of_locs))/len(loc_list)
            exp_agent["state_space_coverage"].append(space_coverage)
            exp_agent["goal_arrival"] += 1
            exp_agent["belief_in_target"].append(float(qs[0][loc_list.index(tuple(reward_location))]))

            # rest agent
            my_agent.curr_timestep = 0
            loc_obs, bound_obs, reward_obs = my_env.reset()
            history_of_locs = [loc_obs]
            obs = [loc_list.index(loc_obs), bound_list[bound_obs], reward_conditions.index(reward_obs)]

            start_time = time.time()
            tracemalloc.reset_peak()
            # print("\n")
        if reward_obs == "Trap":
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            exp_agent["episode_count"] += 1
            exp_agent["time_per_episode"].append((end_time - start_time))
            exp_agent["peak_memory_per_episode"].append(peak / 1024)
            exp_agent["time_step_per_episode"].append(my_agent.curr_timestep)
            exp_agent["trap_arrival"] += 1
            exp_agent["policy_length_per_episode"].append(len(history_of_locs) - 1)
            space_coverage = len(set(history_of_locs)) / len(loc_list)
            exp_agent["state_space_coverage"].append(space_coverage)
            # exp_agent["belief_in_target"] = float(qs[0][loc_list.index(tuple(reward_location))])

            # rest agent
            my_agent.curr_timestep = 0
            loc_obs, bound_obs, reward_obs = my_env.reset()
            history_of_locs = [loc_obs]
            obs = [loc_list.index(loc_obs), bound_list[bound_obs], reward_conditions.index(reward_obs)]

            start_time = time.time()
            tracemalloc.reset_peak()
            # print("\n")

    # exp_agent["episode_count"] = statistics.mean(exp_agent["episode_count"])
    if exp_agent["episode_count"] > 0:
        exp_agent["time_per_episode"] = statistics.mean(exp_agent["time_per_episode"])
        exp_agent["peak_memory_per_episode"] = statistics.mean(exp_agent["peak_memory_per_episode"])
        exp_agent["time_step_per_episode"] = statistics.mean(exp_agent["time_step_per_episode"])
        exp_agent["policy_length_per_episode"] = statistics.mean(exp_agent["policy_length_per_episode"])
        exp_agent["state_space_coverage"] = statistics.mean(exp_agent["state_space_coverage"])

    dtmc = formulate_dtmc(trans_count, loc_list)

    exp_agent["dtmc"] = dtmc
    exp_agent["parameters"] = experiment_config

    # exp_name = experiment_config["name"] + ".pkl"
    # filename = os.path.join(os.getcwd(), "results", exp_name)
    # with open(filename, "wb") as f:
    #     pickle.dump(exp_agent, f)
    exp_class = experiment_config["name"]
    yaml_file = f"{exp_class}_g={experiment_config['gamma']}_a={experiment_config['alpha']}_t={experiment_config['time_steps']}.yaml"


    yaml_filename = os.path.join(os.getcwd(), "results", exp_class, yaml_file)

    if not os.path.exists(os.path.join(os.getcwd(), "results", exp_class)):
        os.mkdir(os.path.join(os.getcwd(), "results", exp_class))


    with open(yaml_filename, "w") as f:
        yaml.dump(exp_agent, f, default_flow_style=True)



    # create_prism_file(loc_list, dtmc)


    #
    #
    #
    #
    # print(my_agent)
    #
    # print(qs)
    # print(my_agent.qs)
    # my_env.render("title")

