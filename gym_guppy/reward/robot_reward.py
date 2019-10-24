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
def cosine_similarity_reward(state, _action, next_state, robot_id):
    env_agents_before = np.array([s[:2] for i, s in enumerate(state) if i != robot_id])
    env_agents_now = np.array([s[:2] for i, s in enumerate(next_state) if i != robot_id])
    swim_directions = env_agents_now - env_agents_before
    directions_to_actor = state[robot_id, :2] - env_agents_before
    # inner = np.sum(swim_directions * directions_to_actor, axis=1)
    inner = swim_directions.dot(directions_to_actor)
    norm_a = row_norm(swim_directions)
    norm_b = row_norm(directions_to_actor)
    # norm_a = np.sqrt(np.sum(swim_directions ** 2, axis=1))
    # norm_b = np.sqrt(np.sum(directions_to_actor ** 2, axis=1))
    cosine_sim = inner / (norm_a * norm_b)
    cosine_sim[np.isnan(cosine_sim)] = 0
    return np.mean(cosine_sim)


@reward_function_with_args
@njit
def leadership_bonus(state, _action, next_state, robot_id):
    actor_before = state[robot_id, :2]
    env_agents_before = np.array([s[:2] for i, s in enumerate(state) if i != robot_id])
    env_agents_now = np.array([s[:2] for i, s in enumerate(next_state) if i != robot_id])
    # env_agents_before = state[1:, :2]
    # env_agents_now = next_state[1:, :2]
    # actor_before = state[0, :2]
    norm_before = row_norm(actor_before - env_agents_before)
    norm_after = row_norm(actor_before - env_agents_now)
    # norm_before = np.sqrt(np.sum(actor_before - env_agents_before, axis=1) ** 2)
    # norm_after = np.sqrt(np.sum(actor_before - env_agents_now, axis=1) ** 2)
    difference = norm_before - norm_after
    return np.mean(difference)


@reward_function_with_args
def cosine_similarity_with_leadership_bonus_reward(state, _action, next_state, bonus_factor):
    return cosine_similarity_reward(next_state, state) + bonus_factor * leadership_bonus(next_state, state)


@reward_function
@njit
def follow_reward(state, _action, next_state):
    swim_directions = next_state[1:, :2] - state[1:, :2]
    directions_to_actor = state[0, :2] - state[1:, :2]
    inner = swim_directions.dot(directions_to_actor)
    # inner = np.sum(swim_directions * directions_to_actor, axis=1)
    norm_b = row_norm(directions_to_actor)
    # norm_b = np.sqrt(np.sum(directions_to_actor ** 2, axis=1))
    follow_metric = inner / norm_b
    follow_metric[np.isnan(follow_metric)] = 0
    return np.mean(follow_metric)


@reward_function
@njit
def fish_to_robot_reward(state, _action, next_state):
    env_agents_before = state[1:, :2]
    env_agents_now = next_state[1:, :2]
    actor_before = state[0, :2]
    norm_before = np.sqrt(np.sum((actor_before - env_agents_before) ** 2, axis=1))
    norm_after = np.sqrt(np.sum((actor_before - env_agents_now) ** 2, axis=1))
    difference = norm_before - norm_after
    return np.mean(difference)


def clipped_proximity_bonus(new_state, clip_value):
    fish_coordinates = new_state[1:, :2]
    robot_coordinates = new_state[0, :2]
    distance_to_robot = np.linalg.norm(robot_coordinates - fish_coordinates)
    clipped_distance = max(clip_value, distance_to_robot)
    return 1.0 + clip_value - clipped_distance


@reward_function_with_args
@njit
def proximity_to_center_reward(_state, _action, next_state, half_diagonal):
    env_agents_coordinates = next_state[1:, :2]
    norm_to_center = np.sqrt(np.sum(env_agents_coordinates ** 2, axis=1))
    return (half_diagonal - np.mean(norm_to_center)) / half_diagonal


def distance_to_fish(new_state):
    # TODO: this function assumes, the robot position is always in the first row
    #  reuse code from guppy_gym, e.g., tools.math.get_local_poses
    fish_coordinates = new_state[1:, :2]
    robot_coordinates = new_state[0, :2]
    return np.linalg.norm(robot_coordinates - fish_coordinates)
