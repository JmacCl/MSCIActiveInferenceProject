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

import numpy as np


class POMDP(ABC):

    observation_model: np.ndarray
    learning_observation: np.ndarray
    transition_model: np.ndarray
    learning_transition: np.ndarray

    reward_model: np.ndarray

    prior_model: np.ndarray
    learning_prior: np.ndarray

    @abstractmethod
    def observation_model(self, observation_offset) -> np.ndarray:
        pass

    @abstractmethod
    def dir_observation_model(self, observation_offset) -> np.ndarray:
        pass

    @abstractmethod
    def transition_model(self, transition_offset) -> np.ndarray:
        pass

    @abstractmethod
    def dir_transition_model(self, transition_modality) -> np.ndarray:
        pass

    @abstractmethod
    def reward_model(self) -> np.ndarray:
        pass

    @abstractmethod
    def prior_model(self, start_location) -> np.ndarray:
        pass

    @abstractmethod
    def dir_prior_model(self) -> np.ndarray:
        pass


