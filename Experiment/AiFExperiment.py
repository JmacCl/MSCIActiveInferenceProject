import os
import statistics
import time
import tracemalloc
import pickle
from copy import deepcopy

import time

from typing import Dict

import yaml

from Experiment.Experiment import Experiment


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
            new_state = (state[0] + val[0], state[1] + val[1])
            if grid_dims[0] > new_state[0] >= 0 and grid_dims[1] > new_state[1] >= 0 and new_state in locs:
                transition_count[state][(state[0] + val[0], state[1] + val[1])] = 0

    return transition_count


def set_up_aif_params(aif_params: Dict, timesteps, trap, verbosity):
    """
    given a specified format of active inference parameters,
    set them up to record each experiment
    :param aif_params:
    :return:
    """
    experiments = []
    gammas = aif_params["gamma"]
    alphas = aif_params["alpha"]
    use_states_info_gain = True
    use_utility = True
    policy_sep_prior = False  # cant be true with use_BMA
    use_BMA = True  # cant be true with policy_sep_prior
    use_param_info_gain = True

    for t in timesteps:
        for g in gammas:
            for alpha in alphas:
                for name, agent_class in aif_params["agent_class"].items():
                    version = deepcopy(agent_class)
                    version["time_steps"] = t
                    version["alpha"] = alpha
                    version["gamma"] = g
                    version["use_states_info_gain"] = use_states_info_gain
                    version["use_utility"] = use_utility
                    version["use_BMA"] = use_BMA
                    version["policy_sep_prior"] = policy_sep_prior
                    version["use_param_info_gain"] = use_param_info_gain
                    version["name"] = f"{name}"
                    version["Trap"] = trap
                    version["verbosity"] = verbosity
                    experiments.append(version)
    return experiments


class AiFExperiment(Experiment):

    def __start_space_policy(self, loc_list):
        return_dict = {}
        for loc in loc_list:
            return_dict[loc] = {}
        return return_dict

    def derive_performance_metrics(self, loc_list):
        performance_metrics = {"time_step_per_episode": [],
                               "episode_count": 0,
                               "peak_memory_per_episode": [],
                               "time_per_episode": [],
                               "policy_length_per_episode": [],
                               "state_space_coverage": [],
                               "trap_arrival": 0,
                               "goal_arrival": 0,
                               "belief_in_target": [],
                               "policy_per_start": self.__start_space_policy(loc_list)
                               }

        return performance_metrics

    def derive_agent_metrics(self, agent):
        agent_information = {}

        self.G_values = [None]
        self.pA_values = [agent.pA]
        self.pB_values = [agent.pB]
        self.pD_values = [agent.pD]
        self.qs_vals = [agent.qs]
        self.q_pi_vals = [agent.E]

        return agent_information

    def __init__(self, agent, env, experiment_config):
        self.environment = env
        self.aif_agent = agent

        loc_obs, bound_obs, reward_obs = self.environment.reset(start_state=self.environment.start_state)
        self.history_of_locs = [loc_obs]
        self.curr_obs = [self.environment.loc_list.index(loc_obs),
                         self.environment.boundary_locations[bound_obs],
                         self.environment.reward_conditions.index(reward_obs)]

        self.T = experiment_config["time_steps"]

        self.performance_metrics = self.derive_performance_metrics(self.environment.loc_list)
        self.agent_information = self.derive_agent_metrics(agent)

        self.verbosity = bool(experiment_config['verbosity'])

        self.curr_policy = []
        self.trans_count = dtmc_construction(self.environment.loc_list, self.environment.shape)
        self.curr_episode = 0

    def __process_trap_terminal(self):
        self.performance_metrics["trap_arrival"] += 1

    def __process_goal_terminal(self):
        self.performance_metrics["goal_arrival"] += 1
        goal_loc = self.environment.goal_location
        goal_belief = float(self.aif_agent.qs[0][self.environment.loc_list.index(goal_loc)])
        self.performance_metrics["belief_in_target"].append(goal_belief)

        proc_policy = tuple(self.curr_policy)
        # now get the policy index from all possible policies
        self.performance_metrics["policy_per_start"][self.history_of_locs[0]][proc_policy] = \
        self.performance_metrics["policy_per_start"][
            self.history_of_locs[0]].get(proc_policy, 0) + 1

    def __transition_change(self):
        """
        From the specified parameters, the tranistion model is changed
        :return:
        """

        pass

    def __observation_change(self):
        pass

    def formulate_dtmc(self, trajectory_count, locs):
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
                if total_transition != 0:
                    prism_dtmc[k][locs.index(tup)] = transitions_dict[tup] / total_transition
                else:
                    prism_dtmc[k][locs.index(tup)] = 0

        return prism_dtmc

    def process_results(self, experiment_config):

        if self.performance_metrics["episode_count"] > 0:
            self.performance_metrics["time_per_episode"] = statistics.mean(self.performance_metrics["time_per_episode"])
            self.performance_metrics["peak_memory_per_episode"] = statistics.mean(
                self.performance_metrics["peak_memory_per_episode"])
            self.performance_metrics["time_step_per_episode"] = statistics.mean(
                self.performance_metrics["time_step_per_episode"])
            self.performance_metrics["policy_length_per_episode"] = statistics.mean(
                self.performance_metrics["policy_length_per_episode"])
            self.performance_metrics["state_space_coverage"] = statistics.mean(
                self.performance_metrics["state_space_coverage"])

        dtmc = self.formulate_dtmc(self.trans_count, self.environment.loc_list)

        self.performance_metrics["dtmc"] = dtmc
        self.performance_metrics["parameters"] = experiment_config

        exp_class = experiment_config["name"]
        exp_case_name = f"{exp_class}_g={experiment_config['gamma']}_a={experiment_config['alpha']}_t={experiment_config['time_steps']}"

        yaml_file = exp_case_name + ".yml"
        pickle_file = exp_case_name + ".pkl"
        yaml_filename = os.path.join(os.getcwd(), "results", exp_class, exp_case_name, yaml_file)
        pkl_filename = os.path.join(os.getcwd(), "results", exp_class, exp_case_name, pickle_file)

        if not os.path.exists(os.path.join(os.getcwd(), "results", exp_class)):
            os.mkdir(os.path.join(os.getcwd(), "results", exp_class))

        # create experiment_folder
        if not os.path.exists(os.path.join(os.getcwd(), "results", exp_class, exp_case_name)):
            os.mkdir(os.path.join(os.getcwd(), "results", exp_class, exp_case_name))

        with open(yaml_filename, "w") as f:
            yaml.dump(self.performance_metrics, f, default_flow_style=True)
        with open(pkl_filename, "wb") as f:
            pickle.dump(self.agent_information, f)

    def __process_performance_metrics(self, peak, end_time, start_time):
        self.performance_metrics["episode_count"] = self.curr_episode
        self.performance_metrics["time_per_episode"].append((end_time - start_time))
        self.performance_metrics["peak_memory_per_episode"].append(peak / 1024)
        self.performance_metrics["time_step_per_episode"].append(self.aif_agent.curr_timestep)
        self.performance_metrics["policy_length_per_episode"].append(len(self.history_of_locs) - 1)
        space_coverage = len(set(self.history_of_locs)) / len(self.environment.loc_list)
        self.performance_metrics["state_space_coverage"].append(space_coverage)

    def __process_agent_metrics(self, t):
        self.agent_information[self.curr_episode] = {
            "policy": tuple(self.curr_policy),
            "G_values": self.G_values[t + 1 - self.aif_agent.curr_timestep: t + 1],
            "pA_values": self.pA_values[t + 1 - self.aif_agent.curr_timestep: t + 1],
            "pB_values": self.pB_values[t + 1 - self.aif_agent.curr_timestep: t + 1],
            "pD_values": self.pD_values[t + 1 - self.aif_agent.curr_timestep: t + 1],
            "qs_vals": self.qs_vals[t + 1 - self.aif_agent.curr_timestep: t + 1],
            "q_pi_vals": self.q_pi_vals[t + 1 - self.aif_agent.curr_timestep: t + 1],
        }

    def __reset_agent(self):
        self.aif_agent.curr_timestep = 0
        loc_obs, bound_obs, reward_obs = self.environment.reset()
        self.history_of_locs = [loc_obs]
        self.curr_obs = [self.environment.loc_list.index(loc_obs),
                         self.environment.boundary_locations[bound_obs],
                         self.environment.reward_conditions.index(reward_obs)]
        self.curr_policy = []

    def run(self):
        start_time = time.time()
        tracemalloc.start()

        for t in range(self.T):

            # If obseravtion of tarnsition effect
            # call and cahneg the observation model via POMDP function of GRIDWOrld.

            self.aif_agent.infer_states(self.curr_obs)

            self.aif_agent.infer_policies()
            chosen_action_id = self.aif_agent.sample_action()
            movement_id = int(chosen_action_id[0])
            self.curr_policy.append(movement_id)

            choice_action = list(self.aif_agent.actions)[movement_id]

            loc_obs, bound_obs, reward_obs = self.environment.step(choice_action)

            self.curr_obs = [self.environment.loc_list.index(loc_obs),
                             self.environment.boundary_locations[bound_obs],
                             self.environment.reward_conditions.index(reward_obs)]

            self.history_of_locs.append(loc_obs)

            next, prev = self.history_of_locs[self.aif_agent.curr_timestep], self.history_of_locs[
                self.aif_agent.curr_timestep - 1]
            self.trans_count[prev][next] += 1

            if self.verbosity:
                print(f'Action at time {t}: {choice_action}')
                print(f'Grid location at time {t}: {loc_obs}')
                print(f'Reward at time {t}: {reward_obs}')

            # Implement Learning
            self.aif_agent.update_A(self.curr_obs)
            # my_agent.update_B(qs_prev)
            self.aif_agent.update_D(qs_t0=None)

            # after each timestep_record_exp_infro
            self.G_values.append(self.aif_agent.G)
            self.pA_values.append(self.aif_agent.pA)
            self.pB_values.append(self.aif_agent.pB)
            self.pD_values.append(self.aif_agent.pD)
            self.qs_vals.append(self.aif_agent.qs)
            self.q_pi_vals.append(self.aif_agent.q_pi)

            if reward_obs != "None":
                self.curr_episode += 1
                # record results
                end_time = time.time()
                current, peak = tracemalloc.get_traced_memory()
                self.__process_performance_metrics(
                    peak, end_time, start_time
                )
                self.__process_agent_metrics(t)
                if reward_obs == "Goal":
                    self.__process_goal_terminal()
                elif reward_obs == "Trap":
                    self.__process_trap_terminal()

                # rest agent
                self.__reset_agent()
                start_time = time.time()
                tracemalloc.reset_peak()
                # self.aif_agent.curr_timestep = 0
                # loc_obs, bound_obs, reward_obs = self.environment.reset()
                # history_of_locs = [loc_obs]
                # obs = [self.environment.loc_list.index(loc_obs), self.environment.bound_list[bound_obs],
                #        self.environment.reward_conditions.index(reward_obs)]
                # self.curr_policy = []

                # print("\n")
