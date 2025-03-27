"""
This file uses the pymdp python fiel to consutruct an agents view of an evnrionment in the ocntext of a 
Partial Observable Markov Decision Process, this can then be used by a given Decision Model class 
that can then be edited,

"""
from abc import ABC, abstractmethod

from Model.POMDP.Observation import Observation
from Model.POMDP.Prior import Prior
from Model.POMDP.Reward import Reward
from Model.POMDP.Transition import Transition


class POMDP(ABC):

    @abstractmethod
    def observation_model(self) -> Observation:
        pass

    @abstractmethod
    def transition_model(self) -> Transition:
        pass

    @abstractmethod
    def reward_model(self) -> Reward:
        pass

    @abstractmethod
    def prior_model(self) -> Prior:
        pass


