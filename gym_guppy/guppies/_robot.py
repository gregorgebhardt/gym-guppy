import abc

import numpy as np

from gym_guppy.guppies import Agent, TurnBoostAgent


class Robot(Agent, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._color = np.array((165, 98, 98))

    @abc.abstractmethod
    def set_action(self, action):
        pass


class TurnBoostRobot(Robot, TurnBoostAgent):
    def set_action(self, action):
        self._turn = action[0]
        self._boost = action[1]


class GoToRobot(Robot, TurnBoostAgent):
    def __init__(self, **kwargs):
        super().__init__(** kwargs)

        self._target = None

        self._turn_eps = np.pi / 16
        self._boost_gain = 1.

    def set_action(self, action):
        self._target = action

    def step(self, time_step):
        # get local polar coordinates
        local_target = self.get_local_point(self._target)
        d, phi = np.arctan2(local_target[1], local_target[0])

        # compute turn and boost
        self._turn = phi - self.get_orientation()
        if self._turn < self._turn_eps:
            self._turn = .0
        self._boost = d * self._boost_gain

        super(GoToRobot, self).step(time_step)
