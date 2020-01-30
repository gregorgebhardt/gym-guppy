import numpy as np
from numba import njit

from gym_guppy.tools.reward_function import reward_wrapper

__all__ = ['clipped_reward']

@reward_wrapper
@njit
def clipped_reward(reward, min_reward=-np.inf, max_reward=np.inf):
    return min(max(reward, min_reward), max_reward)
