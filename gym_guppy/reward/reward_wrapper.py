import numpy as np

from gym_guppy.tools.reward_function import reward_wrapper


@reward_wrapper
def clipped_reward(reward, min_reward=-np.inf, max_reward=np.inf):
    return min(max(reward, min_reward), max_reward)