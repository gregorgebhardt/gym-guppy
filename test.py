import numpy as np
from gym_guppy.envs import GuppyEnv

env = GuppyEnv()
env.reset()
# env.video_path = 'video_out'

for t in range(2000):
    env.render()

    state_t, reward_t, done, info = env.step(np.array([]))
