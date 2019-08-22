import abc

import gym
import numpy as np

from gym_guppy.guppies import Agent, TurnBoostAgent


class Robot(Agent, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._color = np.array((165, 98, 98))

    @property
    @abc.abstractmethod
    def action_space(self) -> gym.spaces.Box:
        pass

    @abc.abstractmethod
    def set_action(self, action):
        pass


class TurnBoostRobot(Robot, TurnBoostAgent):
    def set_action(self, action):
        self.turn = action[0]
        self.boost = action[1]
    
    @property
    def action_space(self):
        return gym.spaces.Box(low=np.array([-self._max_turn, 0.0]), 
                              high=np.array([self._max_turn, self._max_boost]))


class GoToRobot(Robot, TurnBoostAgent):
    def __init__(self, **kwargs):
        super().__init__(** kwargs)

        self.__target = None

        self._turn_eps = np.pi / 32
        self._boost_gain = .1
        self._max_target_distance = .2

    @property
    def _target(self):
        return self.__target

    @_target.setter
    def _target(self, new_target):
        self.__target = new_target

        # get local polar coordinates
        local_target = self.get_local_point(self.__target)
        target_distance = np.linalg.norm(local_target)
        if target_distance > self._max_target_distance:
            local_target *= self._max_target_distance / target_distance

        if target_distance >= .005:
            d = np.linalg.norm(local_target)
            phi = np.arctan2(local_target[1], local_target[0])

            # compute turn and boost
            self.turn = phi
            if abs(self.turn) < self._turn_eps:
                self.turn = .0
            self.boost = d * self._boost_gain
        else:
            self.turn = .0
            self.boost = .0

    def set_action(self, action):
        self._target = action

    # def step(self, time_step):
    #     super(GoToRobot, self).step(time_step)
