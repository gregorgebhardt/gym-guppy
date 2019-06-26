import os
os.environ['NUMBA_DISABLE_JIT'] = "1"

import numpy as np
from gym_guppy.envs import GuppyEnv


env = GuppyEnv()
env.reset()
env.video_path = 'video_out'

for t in range(500):
    env.render()

    state_t, reward_t, done, info = env.step(np.array([]))
