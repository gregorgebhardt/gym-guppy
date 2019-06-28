import os

from gym_guppy.guppies import BoostCouzinGuppy
from gym_guppy.guppies import AdaptiveAgent

# os.environ['NUMBA_DISABLE_JIT'] = "1"

import numpy as np
from gym_guppy.envs import GuppyEnv


class TestEnv(GuppyEnv):
    def _configure_environment(self):
        self._add_guppy(AdaptiveAgent(world=self.world, world_bounds=self.world_bounds))

        num_guppies = 10
        # random initialization
        positions = np.random.normal(loc=.0, scale=.05, size=(num_guppies, 2))
        orientations = np.random.rand(num_guppies) * 2 * np.pi - np.pi

        for p, o in zip(positions, orientations):
            self._add_guppy(BoostCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
                                             position=p, orientation=o))


env = TestEnv()
env.reset()
# env.video_path = 'video_out'

for t in range(500):
    env.render()

    state_t, reward_t, done, info = env.step(np.array([]))
