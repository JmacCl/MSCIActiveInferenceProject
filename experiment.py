from copy import deepcopy
from typing import Dict

from AiF.aif_experiment import aif_experiment_run

def set_up_aif_params(aif_params: Dict, timesteps, trap):
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
                    experiments.append(version)
    return experiments







def experiment_set_up(config_data: Dict):

    # Experiment Comparison
    experiment_parameters = config_data["experiment_parameters"]
    global_parameters = experiment_parameters["global"]

    timesteps = global_parameters["time_steps"]
    ## AIF
    aif_params = experiment_parameters["AiF"]
    aif_configs = set_up_aif_params(aif_params, timesteps=timesteps, trap=global_parameters["trap"])

    for aif in aif_configs:
        aif_experiment_run(config_data, aif)

    # ## RF
    #
    # my_agent = GenerativeModel(config_data)
    # my_env = GridWorldGP2D(config_data)
    #
    #
    # # experimental variables
    # actions = config_data["agent"]["actions"]
    #
    # loc_list = config_data["environment"]["loc_list"]
    # cue1_names = config_data["environment"]["cue1_name"]
    # cue2_names = config_data["environment"]["cue2_name"]
    # reward_names = config_data["environment"]["reward_name"]
    # loc_obs, cue1_obs, cue2_obs, reward_obs = my_env.reset()
    # history_of_locs = [loc_obs]
    # obs = [loc_list.index(loc_obs), cue1_names.index(cue1_obs), cue2_names.index(cue2_obs),
    #        reward_names.index(reward_obs)]
    #
    # # experimental parameters
    # experimental_params = config_data["experiment_parameters"]
    # T = experimental_params["time_steps"]
    # gamma = experimental_params["gamma"]
    # alpha = experimental_params["alpha"]
    #
    # for t in range(T):
    #     my_agent.infer_policies()
    #     chosen_action_id = my_agent.sample_action()
    #
    #     movement_id = int(chosen_action_id[0])
    #
    #     choice_action = actions[movement_id]
    #
    #     print(f'Action at time {t}: {choice_action}')
    #
    #     loc_obs, cue1_obs, cue2_obs, reward_obs = my_env.step(choice_action)
    #
    #     obs = [loc_list.index(loc_obs), cue1_names.index(cue1_obs), cue2_names.index(cue2_obs),
    #            reward_names.index(reward_obs)]
    #
    #     history_of_locs.append(loc_obs)
    #
    #     print(f'Grid location at time {t}: {loc_obs}')
    #
    #     print(f'Reward at time {t}: {reward_obs}')
    #
    # print(my_agent.qs)