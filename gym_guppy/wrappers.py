from collections import deque

import gym
import numpy as np
# import pdb
from gym import spaces
from numpy import float32, linspace

from gym_guppy.tools import LazyFrames, ray_casting_agents, ray_casting_walls


class TimeWrapper(gym.Wrapper):
    
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = spaces.Tuple((self.observation_space, spaces.Box(low=0.0, high=1.0, shape=(1,))))
        self.max_time_steps = self.env.spec.max_episode_steps
        
    def reset(self):
        obs = self.env.reset()
        return (obs, 0.0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        time = self.t/self.max_time_steps
        return (obs, time), reward, done, info
    
    
class TimeWrapper2(gym.Wrapper):
    
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        shp = list(env.observation_space.shape)
        shp[0] = shp[0] + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(shp), dtype=env.observation_space.dtype)
        self.placeholder = np.empty(shp)
        self.max_time_steps = self.env.spec.max_episode_steps
        self.t = 0
        
    def reset(self):
        obs = self.env.reset()
        self.placeholder[0] = self.t/self.max_time_steps
        self.placeholder[1:] = obs
        return self.placeholder

    def step(self, action):
        self.t += 1
        obs, reward, done, info = self.env.step(action)
        time = self.t/self.max_time_steps
        self.placeholder[0] = time
        self.placeholder[1:] = obs
        return self.placeholder, reward, done, info


class FlatActionWrapper(gym.ActionWrapper):
    
    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = gym.spaces.Box(low=self.action_space.low.flatten(), 
                              high=self.action_space.high.flatten())
    def action(self, action):
        return np.expand_dims(action, axis=0)

class DiscreteActionWrapper(gym.ActionWrapper):
    
    def __init__(self, env, num_bins_turn_rate=20, num_bins_speed=20):
        gym.ActionWrapper.__init__(self, env)
        max_turn_rate = self.action_space.high[0]
        self.turn_rate_bins = linspace(-max_turn_rate, max_turn_rate, num_bins_turn_rate + 1)
        max_speed = self.action_space.high[1]
        self.speed_bins = linspace(0, max_speed, num_bins_speed + 1)
        self.action_space = spaces.MultiDiscrete([num_bins_turn_rate, num_bins_speed])

    def action(self, action):
        turn_rate = action[0]
        speed = action[1]
        sample_from_bins = np.random.uniform([self.turn_rate_bins[turn_rate], self.speed_bins[speed]],
                                             [self.turn_rate_bins[turn_rate+1], self.speed_bins[speed+1]]).astype('float32')
        return sample_from_bins
    
    
class RayCastingWrapper(gym.ObservationWrapper):
    
    def __init__(self, env, degrees=360, num_bins=36*2):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2, num_bins), dtype=np.float32)
        self.world_bounds = float32((self.env.world_bounds[0], self.env.world_bounds[1]))
        self.diagonal = np.linalg.norm(self.world_bounds[0] - self.world_bounds[1])
        self.cutoff = np.radians(degrees) / 2.0
        self.vo_agents, self.vo_walls = self._prepare_view_bins((degrees, num_bins), (degrees, num_bins))
        self.obs_placeholder = np.empty(self.observation_space.shape)
     
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

class FrameStack(gym.Wrapper):
    
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=([k] + list(shp)), dtype=env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return np.array(self._get_obs(), dtype=float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.array(self._get_obs(), dtype=float32), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class IgnorePastWallsWrapper(gym.ObservationWrapper):
    def __init__(self, env, k):
        gym.ObservationWrapper.__init__(self, env)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(k+1, shp[-1]), dtype=env.observation_space.dtype)
        self.placeholder = np.empty(self.observation_space.shape)
    
    def observation(self, state):
        self.placeholder[0:2] = state[0]
        self.placeholder[2:] = state[1:,0] # Take only view of agents
        return self.placeholder

    
class TrackAdaptiveZones(gym.Wrapper):
    
    def __init__(self, env, include_in_observation, record=False):
        gym.Wrapper.__init__(self, env)
        if not (include_in_observation or record):
            print('Warning: Using TrackAdaptiveZones-Wrapper without effect.')
        if record:
            print('Warning: Recording not implemented yet!')
        self.include_in_observation = include_in_observation
        if include_in_observation:
            num_guppies = env.num_guppies
            self.observation_space = spaces.Tuple((env.observation_space, spaces.Box(low=0.0, high=np.inf, shape=(num_guppies*3,))))

    def reset(self):
        state = self.env.reset()
        if self.include_in_observation:
            return (state, self.zones()) 
        return state
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.include_in_observation:
            return (state, self.zones()), reward, done, info 
        return state, reward, done, info
    
    def zones(self):
        return np.array([g.couzin_zones() for g in self.env.guppies if isinstance(g, AdaptiveCouzinGuppy)]).flatten()


# class TrackingWrapper(gym.Wrapper):
#
#     def __init__(self, env, output_dir=None, tags=[], verbose=False):
#
#         global h5py
#
#         super().__init__(env)
#         self.env_config = self.env.env_config
#         self.output_dir = output_dir
#         self.tags = tags
#         self.verbose = verbose
#         self.env = env
#
#         print('initializing tracking wrapper') # for debugging
#         print(verbose)
#
#     def reset(self):
#         state = self.env.reset()
#
#         path = '{}/{}_{:03d}.hdf5'.format(self.output_dir, self.tags, self.env.rollout_id)
#         track = h5py.File(path, 'w')
#
#         if self.verbose:
#             print('Saving under path \'{}\''.format(path))
#
#         track.attrs['seed'] = self.env.robofish_seed
#         track.attrs['world'] = self.env.env_config['world']
#         track.attrs['margins'] = self.env.env_config['margins']
#         track.attrs['num_frames'] = self.env.spec.max_episode_steps
#         track.attrs['time_step'] = self.env.step_size
#         if self.tags:
#             track.attrs['tags'] = ','.join(self.tags)
#
#
#         datasets = {}
#         agents = [self.env.actor] + self.env.env_agents
#         for agent in agents:
#             dataset = track.create_dataset(f'{agent.uid}',
#                                            shape=(self.env.spec.max_episode_steps, 4),
#                                            dtype=float32,
#                                            chunks=True)
#             dataset[0] = agent.pose
#             dataset.attrs['agent_type'] = encode_agent_type(agent)
#             if hasattr(agent, 'config_file'):
#                 dataset.attrs['config_file'] = agent.config_file
#             datasets[agent] = dataset
#
#         self.datasets = datasets
#         self.track = track
#
#         agents = [self.env.actor] + self.env.env_agents
#         for agent in agents:
#             self.datasets[agent][self.env.t] = agent.pose
#
#         return state
#
#     def step(self, action):
#         # print('tracking wrapper step') # debugging
#         state, reward, done, info = self.env.step(action)
#         agents = [self.env.actor] + self.env.env_agents
#         for agent in agents:
#             self.datasets[agent][self.env.t] = agent.pose
#         if done:
#             self.track.close()
#         return state, reward, done, info
