import numpy as np

from gym_guppy import GuppyEnv, GoalGuppyEnv
from gym_guppy.reward.robot_reward import extract_swarm_state
from gym_guppy.tools.reward_function import reward_function, reward_function_with_args

__all__ = [
    'goal_reward',
    'relative_goal_reward'
]


@reward_function
def goal_reward(env: GuppyEnv, _state, _action, _next_state):
    assert isinstance(env, GoalGuppyEnv)
    return -np.linalg.norm(env.desired_goal - env.achieved_goal)


@reward_function
def relative_goal_reward(env: GuppyEnv, state, _action, next_state):
    assert isinstance(env, GoalGuppyEnv)
    goal = env.desired_goal
    robot_id = env.robots_idx[0]
    mean_state = np.mean(extract_swarm_state(state, robot_id)[:, :2])
    mean_next_state = np.mean(extract_swarm_state(next_state, robot_id)[:, :2])
    distance_before = np.linalg.norm(goal - mean_state)
    distance_after = np.linalg.norm(goal - mean_next_state)
    return distance_before - distance_after


@reward_function_with_args
def goal_reached_reward(env: GuppyEnv, _state, _action, _next_state, reward=100.):
    assert isinstance(env, GoalGuppyEnv)
    if env.goal_reached():
        return reward
    return .0