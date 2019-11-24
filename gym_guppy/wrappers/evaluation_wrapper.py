import os

import gym
import h5py

import numpy as np

from gym_guppy import AdaptiveCouzinGuppy
from gym_guppy.guppies._randomized_guppies import RandomizedCouzinGuppy
from gym_guppy.reward.robot_reward import follow_reward


class EvaluationWrapper(gym.Wrapper):
    def __init__(self, env, saving_dir, name, num_episodes):
        gym.Wrapper.__init__(self, env)
        self.f = h5py.File(os.path.join(saving_dir, "{}.hdf5".format(name)), "w")
        self.episode_counter = 0
        self.num_episodes = num_episodes

    def reset(self):
        if self.episode_counter == self.num_episodes:
            print('closing')
            self.f.close()
            return
        self.episode_counter += 1

        return_value = self.env.reset()
        self.step_counter = 0

        self.last_state = self.env.get_state()

        self.group = self.f.create_group("episode_{}".format(self.episode_counter))
        self.group.create_dataset(
            'rewards',
            (self.env.spec.max_episode_steps,),
            dtype=np.float32)
        self.group.create_dataset(
            'follow_metric',
            (self.env.spec.max_episode_steps,),
            dtype=np.float32)
        self.group.create_dataset(
            'distance_to_fish',
            (self.env.spec.max_episode_steps,),
            dtype=np.float32)
        self.group.create_dataset(
            'zones_size',
            (self.env.spec.max_episode_steps, 3),
            dtype=np.float32)
        guppy = next(self.env.guppies)
        self.group.attrs['adaptive_zone_factor'] = guppy._adaptive_zone_factors[0]
        self.group.attrs['zone_radius'] = guppy._zone_radius
        self.group.attrs['grow_factor'] = guppy._adaptive_zone_grow_factor
        self.group.attrs['shrink_factor'] = guppy._adaptive_zone_shrink_factor
        self.group.attrs['bias_gain'] = guppy.bias_gain
        self.group.attrs['zoo_factor'] = guppy._zoo_factor

        return return_value

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        self.group['rewards'][self.step_counter] = reward

        current_state = self.env.get_state()
        self.group['follow_metric'][self.step_counter] = follow_reward(current_state, self.last_state)
        self.last_state = current_state

        # TODO: use kd_tree instead
        self.group['distance_to_fish'][self.step_counter] = distance_to_fish(current_state)

        guppy = next(self.env.guppies)
        self.group['zones_size'][self.step_counter] = np.array(guppy.adaptive_couzin_zones(),
                                                               dtype=np.float32).flatten()
        self.step_counter += 1

        return state, reward, done, info


class OmniscienceWrapper(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        num_guppies = env.num_guppies
        # TODO: Time left, DR Parameters
        self.observation_space = gym.spaces.Dict(dict(
                visible_for_all=env.observation_space, 
                only_for_critic=gym.spaces.Box(low=0.0, high=np.inf, shape=(num_guppies *(3 + 6) + 1,)))
                )
        self.max_time_steps = self.env.spec.max_episode_steps

    def reset(self):
        self.t = 0
        state = self.env.reset()
        return dict(visible_for_all=state, only_for_critic=self._get_information())

    def step(self, action):
        self.t += 1
        state, reward, done, info = self.env.step(action)
        return dict(visible_for_all=state, only_for_critic=self._get_information()), reward, done, info

    def _get_information(self):
        dr_parameters = np.array \
            ([g.dr_parameter_list for g in self.env.guppies if isinstance(g, RandomizedCouzinGuppy)]).flatten()
        zones = np.array \
            ([g.adaptive_couzin_zones() for g in self.env.guppies if isinstance(g, AdaptiveCouzinGuppy)]).flatten()
        time_left = np.array([(self.max_time_steps - self.t) / self.max_time_steps]).flatten()
        info = np.concatenate((dr_parameters, zones, time_left), axis=0)
        print(info.shape)
        return info