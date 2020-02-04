import abc

import numpy as np

from ._guppy_env import GuppyEnv


class VariableStepGuppyEnv(GuppyEnv, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._step_logger = []

    def _internal_sim_loop_condition(self):
        if self.robot.action_completed():
            return False
        else:
            time = self._sim_steps * self.sim_step
            self._step_logger.append((time,) + self.robot.get_state() +
                                     (*self.robot.get_linear_velocity(), self.robot.get_angular_velocity()))
            return True

    def get_info(self, state, action):
        steps = np.array(self._step_logger)
        self._step_logger = []
        return dict(steps=steps)
