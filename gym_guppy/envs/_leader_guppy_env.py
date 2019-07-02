import numpy as np
from numba import njit

from gym_guppy.guppies import BoostCouzinGuppy, TurnBoostRobot
from . import GuppyEnv


@njit
def compute_reward(new_state, state):
    swim_directions = new_state[1:, :2] - state[1:, :2]
    directions_to_actor = state[0, :2] - state[1:, :2]
    inner = np.sum(swim_directions * directions_to_actor, axis=1)
    norm_a = np.sqrt(np.sum(swim_directions ** 2, axis=1))
    norm_b = np.sqrt(np.sum(directions_to_actor ** 2, axis=1))
    cosine_sim = inner / (norm_a * norm_b)
    cosine_sim[np.isnan(cosine_sim)] = 0
    reward = np.mean(cosine_sim)
    return reward


@njit
def compute_leadership_bonus(new_state, state):
    env_agents_before = state[1:, :2]
    env_agents_now = new_state[1:, :2]
    actor_before = state[0, :2]
    norm_before = np.sqrt(np.sum(actor_before - env_agents_before, axis=1) ** 2)
    norm_after = np.sqrt(np.sum(actor_before - env_agents_now, axis=1) ** 2)
    difference = norm_before - norm_after
    reward = np.mean(difference)
    return reward


class LeaderGuppyEnv(GuppyEnv):

    def __init__(self, **kwargs):
        super().__init__()
        self.leadership_bonus = kwargs['leadership_bonus'] if 'leadership_bonus' in kwargs.keys() else None
        self.ignore_robots = kwargs['ignore_robots'] if 'ignore_robots' in kwargs.keys() else False
            
    # overrides parent method
    def _configure_environment(self):
        num_guppies = 1
        # random initialization
        positions = np.random.normal(loc=.0, scale=.05, size=(num_guppies + 1, 2))
        orientations = np.random.rand(num_guppies + 1) * 2 * np.pi - np.pi

        self._add_robot(TurnBoostRobot(world=self.world, world_bounds=self.world_bounds,
                                       position=positions[0], orientation=orientations[0]))

        for p, o in zip(positions[1:], orientations[1:]):
            self._add_guppy(BoostCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
                                             position=p, orientation=o))

    # overrides parent method
    def _get_reward(self, state, action, new_state):
        reward = compute_reward(new_state, state)
        if self.leadership_bonus is not None:
            bonus = compute_leadership_bonus(new_state, state)
            reward += bonus * self.leadership_bonus
        if np.isnan(reward):
            raise ValueError('Got NaN-Reward with inputs state {} and past_state {}'.format(state, past_state))
        return reward
    
    # overrides parent method
    def get_state(self):
        if self.ignore_robots:
            return np.array([g.get_state() for g in self.guppies])
        return np.array([a.get_state() for a in self.__agents])
