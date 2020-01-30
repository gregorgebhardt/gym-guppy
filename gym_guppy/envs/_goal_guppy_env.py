import abc

import numpy as np
from gym import spaces

from ._guppy_env import GuppyEnv


class GoalGuppyEnv(GuppyEnv, abc.ABC):
    # def _get_observation_space(self):
    #     return spaces.Dict(dict(
    #         desired_goal=spaces.Box(*self.world_bounds, dtype='float32'),
    #         achieved_goal=spaces.Box(*self.world_bounds, dtype='float32'),
    #         observation=super(GoalGuppyEnv, self)._get_observation_space(),
    #     ))

    @property
    @abc.abstractmethod
    def desired_goal(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def achieved_goal(self):
        raise NotImplementedError

    @abc.abstractmethod
    def maybe_update_goal(self):
        raise NotImplementedError

    def step(self, action: np.ndarray):
        ret_val = super(GoalGuppyEnv, self).step(action)
        self.maybe_update_goal()
        return ret_val
