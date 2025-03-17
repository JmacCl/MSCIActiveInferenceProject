import pickle
import time
import tracemalloc

defined_agents= ["AiF", "RF"]

class ExperimentResults:

    def __init__(self, agent):
        self.agent = agent
        self.objective_experiments = self.__derive_global_metrics

    def __derive_global_metrics(self):
        return {"time_step_per_episode": [], "episode_count": 0, "peak_memory_per_episode": [],
                "time_per_episode": [], "policy_length_per_episode": []}








