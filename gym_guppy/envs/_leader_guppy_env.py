import numpy as np
from numba import njit
import warnings

from gym_guppy.guppies import BoostCouzinGuppy, TurnBoostRobot, AdaptiveCouzinGuppy
from . import GuppyEnv


@njit
def cosine_similarity_reward(new_state, state):
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
def _compute_leadership_bonus(new_state, state):
    env_agents_before = state[1:, :2]
    env_agents_now = new_state[1:, :2]
    actor_before = state[0, :2]
    norm_before = np.sqrt(np.sum(actor_before - env_agents_before, axis=1) ** 2)
    norm_after = np.sqrt(np.sum(actor_before - env_agents_now, axis=1) ** 2)
    difference = norm_before - norm_after
    reward = np.mean(difference)
    return reward


def cosine_similarity_with_leadership_bonus_reward(new_state, state, bonus_factor):
    if not bonus_factor:
        warnings.warn('Using cosine similarity with leadership bonus but bonus factor is 0.')
        return cosine_similarity_reward(new_state, state)
    return cosine_similarity_reward(new_state, state) + bonus_factor * _compute_leadership_bonus(new_state, state)


@njit
def proximity_to_center_reward(new_state, half_diagonal):
    env_agents_coordinates = new_state[1:, :2]
    norm_to_center = np.sqrt(np.sum(env_agents_coordinates, axis=1) ** 2)
    reward = (half_diagonal - np.mean(norm_to_center)) / half_diagonal
    return reward
    

class LeadGuppiesEnv(GuppyEnv):

    def __init__(self, **kwargs):
        self.ignore_robots = kwargs['ignore_robots'] if 'ignore_robots' in kwargs.keys() else False
        self.adaptive = kwargs['adaptive'] if 'adaptive' in kwargs.keys() else False
        self.leadership_bonus = kwargs['leadership_bonus'] if 'leadership_bonus' in kwargs.keys() else 0
        self._num_guppies = kwargs['num_guppies'] if 'num_guppies' in kwargs.keys() else 1
        super().__init__()
        
    # overrides parent method
    def _configure_environment(self):
        # random initialization
        positions = np.random.normal(loc=.0, scale=.05, size=(self._num_guppies + 1, 2))
        orientations = np.random.rand(self._num_guppies + 1) * 2 * np.pi - np.pi
        robot = TurnBoostRobot(world=self.world, world_bounds=self.world_bounds,
                                       position=positions[0], orientation=orientations[0])
        self._add_robot(robot)

        for p, o in zip(positions[1:], orientations[1:]):
            if self.adaptive:
                self._add_guppy(AdaptiveCouzinGuppy(unknown_agents=[robot], world=self.world, world_bounds=self.world_bounds,
                                             position=p, orientation=o))
            else:
                self._add_guppy(BoostCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
                                             position=p, orientation=o))

    # overrides parent method
    def get_reward(self, state, action, new_state):
        reward = cosine_similarity_with_leadership_bonus_reward(new_state, state, self.leadership_bonus)
        if np.isnan(reward):
            raise ValueError('Got NaN-Reward with inputs state {} and past_state {}'.format(state, past_state))
        return reward

    # overrides parent method
    def get_state(self):
        if self.ignore_robots:
            return np.array([g.get_state() for g in self.guppies])
        return super().get_state()


class LeadGuppiesToCenterEnv(LeadGuppiesEnv):    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.half_diagonal = np.linalg.norm(self.world_bounds[0] - self.world_bounds[1]) / 2.
    
    # overrides parent method
    def get_reward(self, state, action, new_state):
        reward = proximity_to_center_reward(new_state, self.half_diagonal)
        if np.isnan(reward):
            raise ValueError('Got NaN-Reward with inputs state {} and past_state {}'.format(state, past_state))
        return reward