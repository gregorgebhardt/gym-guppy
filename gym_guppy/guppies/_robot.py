import abc

import gym
import numpy as np

from gym_guppy.guppies import Agent, TurnBoostAgent
from gym_guppy.guppies._base_agents import VelocityControlledAgent


class Robot(Agent, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._color = np.array((165, 98, 98))

    @property
    @abc.abstractmethod
    def action_space(self) -> gym.spaces.Box:
        pass

    @abc.abstractmethod
    def set_action(self, action):
        pass

    @abc.abstractmethod
    def action_completed(self) -> bool:
        pass


class TurnBoostRobot(Robot, TurnBoostAgent):
    def set_action(self, action):
        self.turn = action[0]
        self.boost = action[1]
    
    @property
    def action_space(self):
        return gym.spaces.Box(low=np.array([-self._max_turn, 0.0]),
                              high=np.array([self._max_turn, self._max_boost]))

    def action_completed(self) -> bool:
        return True


class GoToRobot(Robot, VelocityControlledAgent):
    _linear_damping = .0  # * _world_scale
    _angular_damping = .0  # * _world_scale

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=np.array([-.05, -.05]),
                              high=np.array([.05, .05]))

    def set_action(self, action):
        self._target = self.get_world_point(action)

    # def step(self, time_step):
    #     super(GoToRobot, self).step(time_step)


class ToTargetRobot(Robot, VelocityControlledAgent):
    _angular_damping = .0
    _linear_damping = .0

    def __init__(self, **kwargs):
        super(ToTargetRobot, self).__init__(**kwargs)

        # set position epsilon to 1cm
        self._pos_eps = .01
        # set rotation epsilon to ??rad
        self._ori_eps = np.deg2rad(5)

        world_width = self._world_bounds[1, 0] - self._world_bounds[0, 0]
        world_height = self._world_bounds[1, 1] - self._world_bounds[0, 1]
        world_diag = np.sqrt(world_width**2 + world_height**2)
        # TODO: test action space
        self._action_space = gym.spaces.Box(low=np.array((-world_diag, -world_diag)),
                                            high=np.array((world_diag, world_diag)))

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def set_action(self, action):
        self._target = action

    def action_completed(self) -> bool:
        pos_error = self._target - self.get_position()
        if np.linalg.norm(pos_error) < self._pos_eps:
            return True
