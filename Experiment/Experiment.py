
from abc import ABC, abstractmethod

class Experiment(ABC):


    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def derive_performance_metrics(self, loc_list):
        pass

    @abstractmethod
    def derive_agent_metrics(self, agent):
        pass