import numpy as np

from numba import njit, jit

from gym_guppy import GuppyEnv
from gym_guppy.tools.reward_function import reward_function, reward_function_with_args
from gym_guppy.tools.math import row_norm


__all__ = ['negative_distance_to_swarm',
           'follow_reward',
           'approach_swarm_reward',
           'proximity_to_center_reward',
           'wall_avoidance_gate',
           'in_zor_gate',
           'in_zoi_gate',
           'in_zoi_reward']


@reward_function
# @jit
def negative_distance_to_swarm(env: GuppyEnv, _state, _action, next_state):
    """
    TODO: documentation
    :param env
    :param _state:
    :param _action:
    :param next_state:
    :return:
    """
    robot_id = env.robots_idx[0]
    robot_state = next_state[robot_id, :]
    swarm_state = np.array([s for i, s in enumerate(next_state) if i != robot_id])
    return [-1 * np.sum(row_norm(swarm_state - robot_state))]


@njit(fastmath=True)
def extract_swarm_state(state, robot_id):
    if robot_id:
        return np.concatenate((state[:robot_id, :], state[robot_id + 1:, :]))
    else:
        return state[robot_id + 1:, :]


@njit(fastmath=True)
def _robot_swarm_dist(state, robot_id):
    swarm_state = extract_swarm_state(state, robot_id)
    robot_state = state[robot_id, :]
    return row_norm(swarm_state[:, :2] - robot_state[:2])


@reward_function
# @jit
def follow_reward(env: GuppyEnv, state, _action, next_state):
    """
    TODO: documentation
    :param env:
    :param state:
    :param _action:
    :param next_state:
    :return:
    """
    robot_id = env.robots_idx[0]
    env_agents_before = extract_swarm_state(state, robot_id)[:, :2]
    env_agents_now = extract_swarm_state(next_state, robot_id)[:, :2]
    fish_swim_vec = env_agents_now - env_agents_before
    rob_fish_vec = state[robot_id, :2] - env_agents_before
    inner = fish_swim_vec.dot(rob_fish_vec.T)
    norm_b = row_norm(rob_fish_vec)
    follow_metric = inner / norm_b
    # follow_metric[np.isnan(follow_metric)] = 0
    follow_metric = np.where(np.isnan(follow_metric), .0, follow_metric)
    reward = np.mean(follow_metric)
    return reward


@reward_function
# @jit
def approach_swarm_reward(env: GuppyEnv, state, _action, next_state):
    """
    TODO: documentation
    :param env:
    :param state:
    :param _action:
    :param next_state:
    :return:
    """
    robot_id = env.robots_idx[0]
    norm_before = _robot_swarm_dist(state, robot_id)
    norm_after = _robot_swarm_dist(next_state, robot_id)
    difference = norm_before - norm_after
    return np.mean(difference)


@reward_function
# @jit
def neg_robot_swarm_dist(env: GuppyEnv, _state, _action, next_state):
    robot_id = env.robots_idx[0]
    robot_swarm_dist = _robot_swarm_dist(next_state, robot_id)
    return -np.mean(robot_swarm_dist)


@reward_function_with_args
# @jit
def proximity_to_center_reward(env: GuppyEnv, _state, _action, next_state, half_diagonal):
    robot_id = env.robots_idx[0]
    env_agents_coordinates = np.array([s[:2] for i, s in enumerate(next_state) if i != robot_id])
    norm_to_center = row_norm(env_agents_coordinates)
    return (half_diagonal - np.mean(norm_to_center)) / half_diagonal


@reward_function_with_args
# @jit
def wall_avoidance_gate(env: GuppyEnv, _state, _action, next_state, epsilon):
    min_wall_dist = np.min(np.abs(np.asarray(env.world_bounds) - next_state[env.robots_idx[0], :2]))
    if min_wall_dist > epsilon:
        return 1.
    return .0


@reward_function
def in_zor_gate(env: GuppyEnv, _state, _action, next_state):
    zor = np.array([g.zor for g in env.guppies])
    robot_id = env.robots_idx[0]
    robot_swarm_dist = _robot_swarm_dist(next_state, robot_id)
    if np.any(robot_swarm_dist <= zor):
        return 0.
    return 1.


@reward_function
def in_zoi_gate(env: GuppyEnv, _state, _action, next_state):
    zoi = np.array([g.zoi for g in env.guppies])
    robot_id = env.robots_idx[0]
    robot_swarm_dist = _robot_swarm_dist(next_state, robot_id)
    return 1. + np.mean(zoi - np.maximum(robot_swarm_dist, zoi))


@reward_function
def in_zoi_reward(env:GuppyEnv, _state, _action, next_state):
    zoi = np.array([g.zoi for g in env.guppies])
    robot_id = env.robots_idx[0]
    robot_swarm_dist = _robot_swarm_dist(next_state, robot_id)
    return np.sum(robot_swarm_dist <= zoi)
