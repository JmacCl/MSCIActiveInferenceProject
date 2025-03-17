from typing import Dict

from AiF.aif_experiment import aif_experiment_run


def experiment_set_up(config_data: Dict):

    # Experiment Comparison

    ## AIF

    aif_res = aif_experiment_run(config_data)

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