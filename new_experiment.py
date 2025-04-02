import os
from typing import Dict

from Experiment.AiFExperiment import set_up_aif_params, AiFExperiment
from Model.GridWorld.GridWorldEnv import GridWorldEnv
from Model.GridWorld.GridWorldModel import GridWorldModel
from Agents.AiF import AiFAgent



def experiment_set_up(config_data: Dict):

    # Experiment Comparison
    experiment_parameters = config_data["experiment_parameters"]
    global_parameters = experiment_parameters["global"]
    timesteps = global_parameters["time_steps"]
    verbosity = global_parameters["verbosity"]
    ## AIF
    aif_params = experiment_parameters["AiF"]
    aif_configs = set_up_aif_params(aif_params, timesteps=timesteps,
                                    trap=global_parameters["trap"],
                                    verbosity=verbosity,
                                    test_name=global_parameters["test_name"])

    if not os.path.exists(os.path.join(os.getcwd(), "results")):
        os.mkdir(os.path.join(os.getcwd(), "results"))

    test_name = global_parameters["test_name"]

    if not os.path.exists(os.path.join(os.getcwd(), "results", test_name)):
        os.mkdir(os.path.join(os.getcwd(), "results", test_name))

    agent_config = config_data["agent"]
    env_config = config_data["environment"]

    init_state = agent_config["init_state"]


    env = GridWorldEnv(env_config, init_state, verbosity=verbosity)
    gen_model = GridWorldModel(env, agent_config)

    for aif in aif_configs:
        agent = AiFAgent(gen_model, aif)
        exper = AiFExperiment(agent, env, aif)
        exper.run()
        exper.process_results(aif)

