import numpy as np
from numba import njit
import warnings
import random

from gym_guppy.guppies import BoostCouzinGuppy, TurnBoostRobot, AdaptiveCouzinGuppy, BiasedAdaptiveCouzinGuppy
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

# @njit
def negative_distance_to_center(new_state, diagonal, social_bonus_ratio=0.5):    
    fish_coordinates = new_state[1:, :2]
    robot_coordinates = new_state[0, :2]
    # print(new_state)
    distance_to_robot = np.linalg.norm(robot_coordinates - fish_coordinates)
    distance_to_center = np.mean(np.sqrt(np.sum(fish_coordinates, axis=1) ** 2))
    reward = diagonal - ((1. - social_bonus_ratio) * distance_to_center + social_bonus_ratio * distance_to_robot)
    # print(reward , distance_to_robot)
    return reward


def proximity_to_robot_reward(new_state, half_diagonal):
    fish_coordinates = new_state[1:, :2]
    robot_coordinates = new_state[0, :2]
    distance_to_robot = np.mean(np.sqrt(np.sum(robot_coordinates - fish_coordinates, axis=1) ** 2))
    reward = (half_diagonal - distance_to_robot) / half_diagonal
    return reward





class LeaderGuppyEnv(GuppyEnv):

    def __init__(self, **kwargs):
        self.ignore_robots = kwargs['ignore_robots'] if 'ignore_robots' in kwargs.keys() else False
        self.guppy_type = kwargs['guppy_type'] if 'guppy_type' in kwargs.keys() else 'BoostCouzin'
        self.leadership_bonus = kwargs['leadership_bonus'] if 'leadership_bonus' in kwargs.keys() else 0
        self._num_guppies = kwargs['num_guppies'] if 'num_guppies' in kwargs.keys() else 1
        self.render_zones = kwargs['render_zones'] if 'render_zones' in kwargs.keys() else True
        self.diagonal = np.linalg.norm(self.world_bounds[0] - self.world_bounds[1])
        super().__init__()
        
    # overrides parent method
    def _configure_environment(self):
        # random initialization
        positions = np.random.normal(loc=.0, scale=.05, size=(self._num_guppies, 2))
        orientations = np.random.rand(self._num_guppies + 1) * 2 * np.pi - np.pi
        robot_position = np.concatenate([random.choices(self.world_x_range), random.choices(self.world_y_range)]) / 2.
        robot = TurnBoostRobot(world=self.world, world_bounds=self.world_bounds,
                                       position=robot_position, orientation=orientations[0])
        self._add_robot(robot)

        for p, o in zip(positions, orientations[1:]):
            if self.guppy_type == 'AdaptiveCouzin':
                self._add_guppy(AdaptiveCouzinGuppy(unknown_agents=[robot], world=self.world, world_bounds=self.world_bounds,
                                             position=p, orientation=o))
            elif self.guppy_type == 'BoostCouzin':
                self._add_guppy(BoostCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
                                             position=p, orientation=o))
            elif self.guppy_type == 'BiasedAdaptiveCouzin':
                self._add_guppy(BiasedAdaptiveCouzinGuppy(unknown_agents=[robot], world=self.world, world_bounds=self.world_bounds, 
                                                  position=p, orientation=o, repulsion_points=[[.0, .0]]))
            else:
                raise ValueError('Guppy type does not exist.')
            
            if self._GuppyEnv__reset_counter <= 1:
                print('Added {}Guppy'.format(self.guppy_type))
                
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
    
    def avoid_walls(self, new_state, amount, eps):
        left_right_wall = new_state[0, 0] <= self.world_x_range[0] + eps or new_state[0, 0] >= self.world_x_range[1] - eps
        if left_right_wall:
            return - amount
        up_down_wall = new_state[0, 1] <= self.world_y_range[0] + eps or new_state[0, 1] >= self.world_y_range[1] - eps
        if up_down_wall:
            return - amount
        return 0.0
        

        
    def _draw_on_table(self, screen):
        if not self.render_zones:
            return
        for g in self.guppies:
            if isinstance(g, AdaptiveCouzinGuppy):
                zors, zoos, zoas = g.adaptive_couzin_zones()

                width = .002
                for zor, zoo, zoa in zip(zors, zoos, zoas):
                    screen.draw_circle(g.get_position(), zor + zoo + zoa, color=(0, 100, 0), filled=False, width=width)
                    if zoo + zor > width:
                        screen.draw_circle(g.get_position(), zor + zoo, color=(50, 100, 100), filled=False, width=width)
                    if zor > width:
                        screen.draw_circle(g.get_position(), zor, color=(100, 0, 0), filled=False, width=width)


class LeaderGuppyCenterEnv(LeaderGuppyEnv):    
    
    def __init__(self, **kwargs):
        kwargs['guppy_type'] = 'BiasedAdaptiveCouzin'
        super().__init__(**kwargs)
        print(self.diagonal)
        
    # overrides parent method
    def get_reward(self, state, action, new_state):
        # reward = proximity_to_center_reward(new_state, self.half_diagonal)
        reward = negative_distance_to_center(new_state, self.diagonal, social_bonus_ratio=0.5)
        if np.isnan(reward):
            raise ValueError('Got NaN-Reward with inputs state {} and past_state {}'.format(state, past_state))
        return reward
    
    
class CurriculumEnvironment(LeaderGuppyEnv):
    def __init__(self, **kwargs):
        kwargs['guppy_type'] = 'BiasedAdaptiveCouzin'
        super().__init__(**kwargs)
        self.diagonal = np.linalg.norm(self.world_bounds[0] - self.world_bounds[1])
        self.level = 1
        
    # overrides parent method
    def get_reward(self, state, action, new_state):
        if level == 1:
            return
        reward = proximity_to_center_reward(new_state, self.half_diagonal)
        if np.isnan(reward):
            raise ValueError('Got NaN-Reward with inputs state {} and past_state {}'.format(state, past_state))
        return reward
    
    def zone_reward(self, state, action, new_state):
        for g in self.guppies:
            if np.sqrt(np.sum(robot_coordinates - fish_coordinates)):
                return