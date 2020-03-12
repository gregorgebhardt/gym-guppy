import abc

import numpy as np

from ._goal_guppy_env import GoalGuppyEnv
from ._guppy_env import GuppyEnv


class VariableStepGuppyEnv(GuppyEnv, abc.ABC):
    def __init__(self, *, min_steps_per_action=0, max_steps_per_action=None, **kwargs):
        super().__init__(**kwargs)

        self._min_steps_per_action = min_steps_per_action
        self._max_steps_per_action = max_steps_per_action
        self._step_logger = []

    @property
    def _max_steps_per_action_reached(self):
        if self._max_steps_per_action is None:
            return False
        return self._action_steps >= self._max_steps_per_action

    @property
    def _min_steps_per_action_performed(self):
        return self._action_steps >= self._min_steps_per_action

    @property
    def _internal_sim_loop_condition(self):
        if self._min_steps_per_action_performed and \
                (self.robot.action_completed() or self._max_steps_per_action_reached):
            return False
        else:
            time = self.sim_steps * self.step_time
            self._step_logger.append((time,) + tuple(self.get_state().flat) +
                                     (*self.robot.get_linear_velocity(), self.robot.get_angular_velocity()))
            return True

    def get_info(self, state, action):
        steps = np.array(self._step_logger)
        self._step_logger = []
        return dict(steps=steps)


class VariableStepGoalGuppyEnv(VariableStepGuppyEnv, GoalGuppyEnv, abc.ABC):
    @property
    def _internal_sim_loop_condition(self):
        return (not self.goal_reached()) and super(VariableStepGoalGuppyEnv, self)._internal_sim_loop_condition
