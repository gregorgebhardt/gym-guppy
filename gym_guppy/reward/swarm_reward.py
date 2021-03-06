import gym
import numpy as np
from numba import njit

from gym_guppy.tools.reward_function import reward_function, reward_function_with_args
from gym_guppy.tools.math import row_norm

__all__ = ['negative_distance_to_center']

@reward_function
@njit(fastmath=True)
def negative_distance_to_center(_state, _action, next_state):
    # reward is high, if fish are in the center of the tank
    return -np.sum(row_norm(next_state))
