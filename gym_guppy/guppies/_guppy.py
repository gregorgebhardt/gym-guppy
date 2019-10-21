import abc
from typing import List

import numpy as np
from scipy.spatial.ckdtree import cKDTree

from gym_guppy.guppies._base_agents import Agent, TorqueThrustAgent, TurnBoostAgent, TurnSpeedAgent


class Guppy(Agent, abc.ABC):
    # TODO Guppy decides when to call compute next action, this allows to run at different frequencies
    # def __init__(self, *, world_bounds, **kwargs):
    #     super().__init__(world_bounds=world_bounds, **kwargs)
    #     self._time_step = 0.1
    #     self._time = .0
    #
    # def step(self, *, time_step, state, kd_tree, **kwargs):
    #     if self._time % self._time_step:
    #         self.compute_next_action(state, kd_tree)
    #
    #     super(Guppy, self).step(time_step=time_step, **kwargs)
    #
    #     self._time += time_step

    @abc.abstractmethod
    def compute_next_action(self, state: List[Agent], kd_tree: cKDTree = None):
        raise NotImplementedError


class TorqueGuppy(Guppy, TorqueThrustAgent):
    def compute_next_action(self, **kwargs):
        self._torque = 250


class BoostGuppy(Guppy, TorqueThrustAgent):
    def compute_next_action(self, state: List['Agent'], kd_tree: cKDTree = None):
        print(str(self._body.mass) + " " + str(self._body.linearVelocity))
        if np.linalg.norm(self._body.linearVelocity) < .05:
            self._thrust = 10.


class RandomTurnBoostGuppy(Guppy, TurnBoostAgent):
    def compute_next_action(self, state: List['Agent'], kd_tree: cKDTree = None):
        self._turn = np.random.randn() * .75
        self._boost = np.abs(np.random.randn() * .3)


class TurnBoostGuppy(Guppy, TurnSpeedAgent, abc.ABC):
    pass
