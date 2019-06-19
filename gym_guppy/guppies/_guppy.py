import abc
from typing import List

import numpy as np
from Box2D import b2Vec2
from scipy.spatial.ckdtree import cKDTree

from gym_guppy.guppies._base_agents import Agent, TorqueThrustAgent, TurnBoostAgent


class Guppy(Agent, abc.ABC):
    @abc.abstractmethod
    def compute_next_action(self, state: List[Agent], kd_tree: cKDTree = None):
        raise NotImplementedError


class TorqueGuppy(Guppy, TorqueThrustAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_next_action(self, **kwargs):
        self._torque = 250


class BoostGuppy(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_next_action(self, state: List['Agent'], time_step):
        print(str(self._body.mass) + " " + str(self._body.linearVelocity))
        if np.linalg.norm(self._body.linearVelocity) < .05:
            self._body.ApplyForceToCenter(self._body.GetLocalVector(b2Vec2(10., .0)), wake=True)


class RandomTurnBoostGuppy(TurnBoostAgent):
    def compute_next_action(self, state: List['Agent'], time_step):
        self._turn = np.random.randn() * .75
        self._boost = np.abs(np.random.randn() * .3)