import abc
import warnings
from itertools import cycle
from typing import Callable, List, Tuple, Type, Union

import numpy as np

from gym_guppy.envs import GuppyEnv
# from gym_guppy.guppies import Guppy, AdaptiveCouzinGuppy, BiasedAdaptiveCouzinGuppy, Robot, PolarCoordinateTargetRobot
from gym_guppy.guppies import *


def robot_pose_sample_normal(bounds):
    position = np.random.normal(scale=.15, size=(2,))
    position = np.minimum(np.maximum(position, bounds[0]), bounds[1])

    orientation = np.random.random_sample() * 2 * np.pi - np.pi

    return np.r_[position, orientation]

def robot_pose_sample_uniform(bounds):
    size = bounds[1] - bounds[0]
    position = (np.random.random_sample(2) * size + bounds[0]) * .9

    orientation = np.random.random_sample() * 2 * np.pi - np.pi

    return np.r_[position, orientation]


def swarm_pose_sample_normal(num_guppies: int, bounds):
    size = bounds[1] - bounds[0]
    swarm_position = (np.random.random_sample(2) * size + bounds[0]) * .95

    fish_positions = np.random.normal(loc=swarm_position, scale=0.05, size=(num_guppies, 2))
    fish_positions = np.minimum(np.maximum(fish_positions, bounds[0]), bounds[1])

    fish_orientations = np.random.random_sample((num_guppies, 1)) * 2 * np.pi - np.pi

    return np.c_[fish_positions, fish_orientations]


class ConfigurableGuppyEnv(GuppyEnv, abc.ABC):
    def __init__(self,
                 robot_type: Union[str, Type[Robot]] = PolarCoordinateTargetRobot,
                 robot_args: dict = None,
                 robot_pose_rng: Union[str, Callable[[Tuple[np.ndarray]], np.ndarray]] = robot_pose_sample_normal,
                 guppy_type: Union[str, Type[Guppy]] = AdaptiveCouzinGuppy,
                 guppy_args: Union[dict, List[dict]] = None,
                 guppy_pose_rng: Union[str, Callable[[int, Tuple[np.ndarray]], np.ndarray]] = swarm_pose_sample_normal,
                 num_guppies=1,
                 controller_params=None,
                 **kwargs):

        if isinstance(robot_type, str):
            robot_type = eval(robot_type)
        self.robot_type = robot_type
        self.robot_args = robot_args or {}

        if controller_params:
            # warnings.warn('use of controller_params keyword in ConfigurableGuppyEnv is deprecated, add to robot_args '
            #               'instead!', DeprecationWarning)
            self.robot_args['ctrl_params'] = controller_params

        if isinstance(guppy_type, str):
            guppy_type = eval(guppy_type)
        self.guppy_type = guppy_type
        self.guppy_args = guppy_args or {}
        if not isinstance(self.guppy_args, list):
            self.guppy_args = [self.guppy_args]
        self._num_guppies = num_guppies

        if isinstance(robot_pose_rng, str):
            robot_pose_rng = eval(robot_pose_rng)
        self._robot_pose_rng = robot_pose_rng
        if isinstance(guppy_pose_rng, str):
            guppy_pose_rng = eval(guppy_pose_rng)
        self._swarm_pose_rng = guppy_pose_rng

        super(ConfigurableGuppyEnv, self).__init__(**kwargs)

    def _reset(self):
        super(ConfigurableGuppyEnv, self)._reset()

        robot_pose = self._robot_pose_rng(self.world_bounds)
        self._add_robot(self.robot_type(world=self.world,
                                        world_bounds=self.world_bounds,
                                        position=robot_pose[:2],
                                        orientation=robot_pose[2],
                                        **self.robot_args))

        fish_poses = self._swarm_pose_rng(self._num_guppies, self.world_bounds)
        guppy_args_cycle = cycle(self.guppy_args)
        for pose in fish_poses:
            guppy_args = dict(world=self.world, world_bounds=self.world_bounds, position=pose[:2], orientation=pose[2])
            guppy_args.update(next(guppy_args_cycle))
            if issubclass(self.guppy_type, AdaptiveCouzinGuppy):
                guppy_args['unknown_agents'] = [self.robot]
            if issubclass(self.guppy_type, BiasedAdaptiveCouzinGuppy):
                guppy_args['repulsion_points'] = [[.0, .0]]

            self._add_guppy(self.guppy_type(**guppy_args))

            # TODO: add parameters for randomization
            # elif self.guppy_type == 'RandomizedCouzin':
            #     if not self.randomization_params:
            #         warnings.warn('Specified RandomizedCouzin but no randomizations!')
            #     self._add_guppy(RandomizedCouzinGuppy(
            #         unknown_agents=[self.robot], world=self.world, world_bounds=self.world_bounds,
            #         position=pos, orientation=ori, repulsion_points=[[.0, .0]], rng=dr_rng,
            #         **self.randomization_params))
