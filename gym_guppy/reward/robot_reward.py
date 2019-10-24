import numpy as np

from numba import njit

from gym_guppy.tools.reward_function import reward_function, reward_function_with_args, reward_wrapper
from gym_guppy.tools.math import row_norm


@reward_function_with_args
@njit
def negative_distance_to_swarm(_state, _action, next_state, robot_id):
    robot_state = next_state[robot_id, :]
    swarm_state = np.array([s for i, s in enumerate(next_state) if i != robot_id])
    return [-1 * np.sum(row_norm(swarm_state - robot_state))]
    

@reward_function_with_args
@njit
def follow_reward(state, _action, next_state, robot_id):
    env_agents_before = np.array([s[:2] for i, s in enumerate(state) if i != robot_id])
    env_agents_now = np.array([s[:2] for i, s in enumerate(next_state) if i != robot_id])
    swim_directions = env_agents_now - env_agents_before
    directions_to_actor = state[robot_id, :2] - env_agents_before
    inner = swim_directions.dot(directions_to_actor)
    # inner = np.sum(swim_directions * directions_to_actor, axis=1)
    norm_b = row_norm(directions_to_actor)
    # norm_b = np.sqrt(np.sum(directions_to_actor ** 2, axis=1))
    follow_metric = inner / norm_b
    follow_metric[np.isnan(follow_metric)] = 0
    reward = np.mean(follow_metric)
    return reward


@reward_function_with_args
@njit
def fish_to_robot_reward(state, _action, next_state, robot_id):
    env_agents_before = np.array([s[:2] for i, s in enumerate(state) if i != robot_id])
    env_agents_now = np.array([s[:2] for i, s in enumerate(next_state) if i != robot_id])
    actor_before = state[robot_id, :2]
    norm_before = row_norm(actor_before - env_agents_before)
    norm_after = row_norm(actor_before - env_agents_now)
    difference = norm_before - norm_after
    return np.mean(difference)

@reward_function_with_args
def clipped_proximity_bonus(state, clip_value, robot_id):
    env_agents_now = np.array([s[:2] for i, s in enumerate(state) if i != robot_id])
    robot_coordinates = state[robot_id, :2]
    distance_to_robot = np.linalg.norm(robot_coordinates - fish_coordinates)
    clipped_distance = max(clip_value, distance_to_robot)
    return 1.0 + clip_value - clipped_distance


@reward_function_with_args
@njit
def proximity_to_center_reward(_state, _action, next_state, half_diagonal, robot_id):
    env_agents_coordinates = np.array([s[:2] for i, s in enumerate(new_state) if i != robot_id])
    norm_to_center = row_norm(env_agents_coordinates)
    return (half_diagonal - np.mean(norm_to_center)) / half_diagonal


def distance_to_fish(new_state):
    # TODO: this function assumes, the robot position is always in the first row
    #  reuse code from guppy_gym, e.g., tools.math.get_local_poses
    fish_coordinates = new_state[1:, :2]
    robot_coordinates = new_state[0, :2]
    return np.linalg.norm(robot_coordinates - fish_coordinates)
