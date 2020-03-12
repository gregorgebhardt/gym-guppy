import abc

import gym
import numpy as np

from gym_guppy.guppies import Agent, TurnBoostAgent
from gym_guppy.guppies._base_agents import VelocityControlledAgent


class Robot(Agent, abc.ABC):
    _linear_damping = .0
    _angular_damping = .0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._env_state = None
        self._env_kd_tree = None

        self._color = np.array((165, 98, 98))

    def set_env_state(self, env_state, kd_tree):
        self._env_state = env_state
        self._env_kd_tree = kd_tree

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


class GlobalTargetRobot(Robot, VelocityControlledAgent):
    def __init__(self, modulated=False, **kwargs):
        super(GlobalTargetRobot, self).__init__(**kwargs)
        # set position epsilon to 1cm
        self._pos_eps = .02
        self._modulated = modulated

        if self._modulated:
            self._action_space = gym.spaces.Box(low=np.r_[self._world_bounds[0], .0],
                                                high=np.r_[self._world_bounds[1], .2],
                                                dtype=np.float64)
        else:
            self._action_space = gym.spaces.Box(low=self._world_bounds[0], high=self._world_bounds[1], dtype=np.float64)

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def set_action(self, action):
        if self._modulated:
            self.two_wheels_controller.fwd_ctrl.speed = action[-1]
            self._target = action[:-1]
        else:
            self._target = action

    def action_completed(self) -> bool:
        pos_error = self._target - self.get_position()
        if np.linalg.norm(pos_error) < self._pos_eps:
            return True


class PolarCoordinateTargetRobot(GlobalTargetRobot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._action_space = gym.spaces.Box(low=np.array((-np.pi, .0)),
                                            high=np.array((np.pi, .3)),
                                            dtype=np.float64)

    def set_action(self, action):
        local_target = np.array((np.cos(action[0]), np.sin(action[0]))) * action[1]
        super(PolarCoordinateTargetRobot, self).set_action(self.get_global_point(local_target))
