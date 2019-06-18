from typing import List

import numpy as np
from Box2D import b2Vec2

from gym_guppy.guppies._base_guppies import TorqueThrustGuppy, Guppy, TurnBoostGuppy


class TorqueGuppy(TorqueThrustGuppy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_next_action(self, **kwargs):
        self._torque = 250


class BoostGuppy(Guppy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_next_action(self, state: List['Guppy'], time_step):
        print(str(self._body.mass) + " " + str(self._body.linearVelocity))
        if np.linalg.norm(self._body.linearVelocity) < .05:
            self._body.ApplyForceToCenter(self._body.GetLocalVector(b2Vec2(10., .0)), wake=True)


class RandomTurnBoostGuppy(TurnBoostGuppy):
    def compute_next_action(self, state: List['Guppy'], time_step):
        self._turn = np.random.randn() * .75
        self._boost = np.abs(np.random.randn() * .3)
