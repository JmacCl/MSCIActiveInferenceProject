"""
Given a Model agent strucutre, this file implements the active infrence deicison model
"""

from Model.POMDP.POMDP import POMDP

from pymdp.agent import Agent


ACTION_MAP = {
    "UP": lambda Y, X, shape: (max(Y - 1, 0), X),
    "DOWN": lambda Y, X, shape: (min(Y + 1, shape[0] - 1), X),
    "LEFT": lambda Y, X, shape: (Y, max(X - 1, 0)),
    "RIGHT": lambda Y, X, shape: (Y, min(X + 1, shape[1] - 1)),
    "STAY": lambda Y, X, shape: (Y, X)
}

class AiFAgent(Agent):

    def __init__(self, gen_model: POMDP, aif_config):

        self.actions = ACTION_MAP.keys()

        A = gen_model.observation_model
        pA = gen_model.learning_observation

        B = gen_model.transition_model
        pB = gen_model.learning_transition

        C = gen_model.reward_model

        D = gen_model.prior_model
        pD = gen_model.learning_prior

        super().__init__(A=A, B=B, C=C, D=D, policy_len=aif_config["policy_len"],
                         save_belief_hist=True,
                         gamma=aif_config["gamma"], alpha=aif_config["alpha"],
                         use_states_info_gain=bool(aif_config["use_states_info_gain"]),
                         use_utility=bool(aif_config["use_utility"]),
                         action_selection=aif_config["agent_selection"],
                         sampling_mode=aif_config["sampling_mode"],
                         use_BMA=aif_config["use_BMA"],
                         policy_sep_prior=aif_config["policy_sep_prior"],
                         use_param_info_gain=aif_config["use_param_info_gain"],
                         pA=pA, pD=pD)

