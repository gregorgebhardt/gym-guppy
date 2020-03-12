import abc

import numpy as np

from ._guppy_env import GuppyEnv


class GoalGuppyEnv(GuppyEnv, abc.ABC):
    @property
    @abc.abstractmethod
    def desired_goal(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def achieved_goal(self):
        raise NotImplementedError

    @abc.abstractmethod
    def goal_reached(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_goal(self):
        raise NotImplementedError

    def step(self, action: np.ndarray):
        ret_val = super(GoalGuppyEnv, self).step(action)
        if self.goal_reached():
            self._update_goal()
        return ret_val
