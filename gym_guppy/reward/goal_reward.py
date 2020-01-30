import numpy as np

from gym_guppy import GuppyEnv
from gym_guppy.reward.robot_reward import extract_swarm_state
from gym_guppy.tools.reward_function import reward_function

__all__ = [
    'goal_reward',
    'relative_goal_reward'
]


@reward_function
def goal_reward(env: GuppyEnv, _state, _action, next_state):
    goal = env.desired_goal
    robot_id = env.robots_idx[0]
    mean_state = np.mean(extract_swarm_state(next_state, robot_id)[:, :2])
    return -np.linalg.norm(goal - mean_state)


@reward_function
def relative_goal_reward(env: GuppyEnv, state, _action, next_state):
    goal = env.desired_goal
    robot_id = env.robots_idx[0]
    mean_state = np.mean(extract_swarm_state(state, robot_id)[:, :2])
    mean_next_state = np.mean(extract_swarm_state(next_state, robot_id)[:, :2])
    distance_before = np.linalg.norm(goal - mean_state)
    distance_after = np.linalg.norm(goal - mean_next_state)
    return distance_before - distance_after
