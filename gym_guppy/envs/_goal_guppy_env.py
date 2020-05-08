import abc

import numpy as np

from ._guppy_env import GuppyEnv


class GoalGuppyEnv(GuppyEnv, abc.ABC):
    def __init__(self, *, change_goal_threshold=.05, **kwargs):
        super(GoalGuppyEnv, self).__init__(**kwargs)
        self.change_goal_threshold = change_goal_threshold

    @property
    @abc.abstractmethod
    def desired_goal(self):
        raise NotImplementedError

    @property
    def achieved_goal(self):
        if self.num_guppies:
            return np.mean([g.get_position() for g in self.guppies], axis=0)
        else:
            return self.robot.get_position()

    def goal_reached(self) -> bool:
        return np.linalg.norm(self.desired_goal - self.achieved_goal) <= self.change_goal_threshold

    @abc.abstractmethod
    def _update_goal(self):
        raise NotImplementedError

    def step(self, action: np.ndarray):
        ret_val = super(GoalGuppyEnv, self).step(action)
        if self.goal_reached():
            self._update_goal()
        return ret_val

    def get_done(self, state, action):
        return self.goal_reached()
