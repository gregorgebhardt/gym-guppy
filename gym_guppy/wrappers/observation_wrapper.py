from collections import deque

import gym
import numpy as np

from gym_guppy import AdaptiveCouzinGuppy
from gym_guppy.tools.math import get_local_poses, transform_sin_cos, ray_casting_walls, ray_casting_agents
from gym_guppy.tools.datastructures import LazyFrames


class LocalObservationsWrapper(gym.ObservationWrapper):
    def __init__(self, env, robot_id=0):
        super(LocalObservationsWrapper, self).__init__(env)
        self.robot_id = robot_id
        # assert action space is n x 3, where n > 1
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        assert self.env.observation_space.shape[1] == 3
        assert self.env.observation_space.shape[0] > 1

    def observation(self, observation):
        robot_pose = observation[self.robot_id, :]
        guppy_pose = np.array([r for i, r in enumerate(observation) if i != self.robot_id])

        local_poses = get_local_poses(guppy_pose, robot_pose)
        return np.c_[local_poses[:, :2], transform_sin_cos(local_poses[:, [2]])]


class RayCastingWrapper(gym.ObservationWrapper):

    def __init__(self, env, degrees=360, num_bins=36 * 2):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2, num_bins), dtype=np.float32)
        self.world_bounds = np.float32((self.env.world_bounds[0], self.env.world_bounds[1]))
        self.diagonal = np.linalg.norm(self.world_bounds[0] - self.world_bounds[1])
        self.cutoff = np.radians(degrees) / 2.0
        self.vo_agents, self.vo_walls = self._prepare_view_bins((degrees, num_bins), (degrees, num_bins))
        self.obs_placeholder = np.empty(self.observation_space.shape)
        self.env.vo_goal = self.vo_agents

    def observation(self, state):
        self.obs_placeholder[0] = ray_casting_agents(state[0], state[1:], self.vo_agents, self.diagonal)
        self.obs_placeholder[1] = ray_casting_walls(state[0], self.world_bounds, self.vo_walls, self.diagonal)
        return self.obs_placeholder

    def _prepare_view_bins(self, view_of_agents, view_of_walls):
        fop, view_of_agents_size = view_of_agents
        view_of_agents_sectors = np.linspace(-self.cutoff,
                                             self.cutoff,
                                             view_of_agents_size + 1,
                                             dtype=np.float32)
        fop, view_of_walls_size = view_of_walls
        view_of_walls_rays = np.linspace(-self.cutoff,
                                         self.cutoff,
                                         view_of_walls_size,
                                         dtype=np.float32)
        return view_of_agents_sectors, view_of_walls_rays


class TrackAdaptiveZones(gym.ObservationWrapper):
    def __init__(self, env, include_in_observation, record=False):
        super(TrackAdaptiveZones, self).__init__(env)
        if not (include_in_observation or record):
            print('Warning: Using TrackAdaptiveZones-Wrapper without effect.')
        if record:
            print('Warning: Recording not implemented yet!')
        self.include_in_observation = include_in_observation
        if include_in_observation:
            num_guppies = env.num_guppies
            self.observation_space = gym.spaces.Tuple(
                (env.observation_space, gym.spaces.Box(low=0.0, high=np.inf, shape=(num_guppies * 3,))))

    def observation(self, observation):
        if self.include_in_observation:
            return observation, self.zones()
        return observation

    def zones(self):
        return np.array([g.adaptive_couzin_zones() for g in self.env.guppies
                         if isinstance(g, AdaptiveCouzinGuppy)]).flatten()


class TimeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TimeWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Tuple((self.observation_space,
                                                   gym.spaces.Box(low=0.0, high=1.0, shape=(1,))))
        self.max_time_steps = self.env.spec.max_episode_steps

    def reset(self):
        self.t = 0.0
        return super(TimeWrapper, self).reset()

    def observation(self, observation):
        time = self.t / self.max_time_steps
        return observation, time


class TimeWrapper2(gym.ObservationWrapper):

    def __init__(self, env):
        super(TimeWrapper2, self).__init__(env)
        shape = list(env.observation_space.shape)
        shape[0] = shape[0] + 1
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=shape, dtype=env.observation_space.dtype)
        self.placeholder = np.empty(shape)
        self.max_time_steps = self.env.spec.max_episode_steps
        self.t = 0

    def reset(self):
        obs = self.env.reset()
        self.placeholder[0] = self.t / self.max_time_steps
        self.placeholder[1:] = obs
        return self.placeholder

    def observation(self, observation):
        self.t += 1
        time = self.t / self.max_time_steps
        self.placeholder[0] = time
        self.placeholder[1:] = observation
        return self.placeholder


class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=([k] + list(shape)),
                                                dtype=env.observation_space.dtype)

    def reset(self):
        obs = super(FrameStack, self).reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def observation(self, observation):
        self.frames.append(observation)
        return self._get_obs()

    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.array(LazyFrames(list(self.frames)), dtype=np.float32)


class IgnorePastWallsWrapper(gym.ObservationWrapper):
    def __init__(self, env, k):
        gym.ObservationWrapper.__init__(self, env)
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(k + 1, shape[-1]),
                                                dtype=env.observation_space.dtype)
        self.placeholder = np.empty(self.observation_space.shape)

    def observation(self, state):
        self.placeholder[0:2] = state[0]
        self.placeholder[2:] = state[1:, 0]  # Take only view of agents
        return self.placeholder
