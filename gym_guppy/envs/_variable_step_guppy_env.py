import abc

import numpy as np
from gym import spaces

from ._guppy_env import GuppyEnv


class VariableStepGuppyEnv(GuppyEnv, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._step_logger = []

    def _internal_sim_loop_condition(self):
        self._step_logger.append(self.robot.get_state())
        # print('.', end='')
        return not self.robot.action_completed()

    def get_info(self, state, action):
        steps = np.array(self._step_logger)
        self._step_logger = []
        return dict(steps=steps)
