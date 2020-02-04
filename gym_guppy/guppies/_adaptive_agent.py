import abc
import warnings

import gym
import numpy as np
from scipy.spatial.ckdtree import cKDTree

from gym_guppy.guppies import Agent, Guppy, ToTargetRobot
from gym_guppy.tools import Feedback
from gym_guppy.tools.math import is_point_left, normalize, rotation


class AdaptiveAgent(ToTargetRobot):
    def action_completed(self) -> bool:
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._feedback = Feedback()
        self._state: AdaptiveState = ApproachState()

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(high=np.inf, low=-np.inf, shape=tuple())

    def set_action(self, action):
        robot_pose = self._env_state[self.id]
        _, [nn] = self._env_kd_tree.query(robot_pose[:2], k=[2])
        guppy_pose = self._env_state[nn]

        self._feedback.update(guppy_pose, robot_pose)

        self._state = self._state.switch_state()
        self._target = self._state.compute_next_action(self, self._feedback, self._env_state, self._env_kd_tree)

    def compute_next_action(self, state: np.ndarray, kd_tree: cKDTree = None):
        # robot_pose = state[self.id]
        # _, [nn] = kd_tree.query(robot_pose[:2], k=[2])
        # guppy_pose = state[nn]
        #
        # self._feedback.update(guppy_pose, robot_pose)
        #
        # self._state = self._state.switch_state()
        # self._target = self._state.compute_next_action(self, self._feedback, state, kd_tree)
        pass


class AdaptiveState(abc.ABC):
    _dt = 0.01

    @abc.abstractmethod
    def compute_next_action(self, agent: Agent, feedback: Feedback, state: np.ndarray, kd_tree: cKDTree = None) -> \
            np.ndarray:
        pass

    @abc.abstractmethod
    def switch_state(self) -> "AdaptiveState":
        pass


class MillingState(AdaptiveState):
    _mill_diameter = 0.2
    _mill_center = np.array([.0, .0])

    def __init__(self):
        # TODO compute mill points
        self._mill_points = []
        self._mill_index = 0

    def compute_next_action(self, agent: Agent, feedback: Feedback, state: np.ndarray, kd_tree: cKDTree = None):
        # TODO implement
        pass

    def switch_state(self) -> "AdaptiveState":
        # TODO implement
        pass


class ApproachState(AdaptiveState):
    _decision_bound = 0.5
    _learning_rate = 0.075
    _linear_speed_correction = 0.2
    _comfort_zone = 0.12

    def __init__(self):
        super().__init__()

        self._close_enough = False
        self._approach_parameter = 0.5

    def compute_next_action(self, agent: Agent, feedback: Feedback, state: np.ndarray, kd_tree: cKDTree = None):
        # get own position and nearest fish position
        robot_pos = state[agent.id][:2]
        _, [nn] = kd_tree.query(robot_pos, k=[2])
        guppy_pos = state[nn, :2]

        # get fear and compute integrate
        fear = feedback.fear
        integrate = (fear - self._decision_bound) * self._dt * self._learning_rate

        self._approach_parameter += integrate
        self._approach_parameter = min(max(self._approach_parameter, 0.0), 1.0)

        # infer fish motion
        guppy_dir = feedback.infer_guppy_motion_direction(1.0)

        # compute motion target
        guppy_robot_vec = robot_pos - guppy_pos

        if np.linalg.norm(guppy_robot_vec) < self._comfort_zone:
            self._close_enough = True
            return robot_pos

        target = guppy_pos + normalize(guppy_robot_vec) * self._comfort_zone / 2.
        rot_dir = 1 if is_point_left(guppy_pos, guppy_pos + guppy_dir, robot_pos) else -1
        r = rotation(self._approach_parameter * np.pi * .5 * rot_dir)
        return r.dot(target - robot_pos) + robot_pos

    def switch_state(self) -> "AdaptiveState":
        if self._close_enough:
            print('Switching to LeadState')
            return LeadState()
        return self


class LeadState(AdaptiveState):
    _target_radius = 0.05
    _max_lead_dist = 0.1

    _targets = np.array([[0.4, 0.4],
                         [0.4, -0.4],
                         [-0.4, -0.4],
                         [-0.4, 0.4]])

    def __init__(self):
        super().__init__()

        self._waiting_counter = 0
        self._target_idx = None

    def compute_next_action(self, agent: Agent, feedback: Feedback, state: np.ndarray, kd_tree: cKDTree = None):
        robot_pos = state[agent.id, :2]
        _, [nn] = kd_tree.query(robot_pos, k=[2])
        guppy_pos = state[nn, :2]

        # compute next target
        if self._target_idx is None:
            target_dists = np.linalg.norm(self._targets - robot_pos, axis=1)
            self._target_idx = np.argmin(target_dists)

        # check if robot is close enough to target
        if np.linalg.norm(self._targets[self._target_idx] - robot_pos) < self._target_radius:
            self._target_idx += 1
            self._target_idx %= len(self._targets)

        # check if guppy is close enough to robot
        if np.linalg.norm(guppy_pos - robot_pos) > self._max_lead_dist:
            print('Waiting for Guppy')
            self._waiting_counter += 1
            return robot_pos

        self._waiting_counter = 0
        return self._targets[self._target_idx]

    def switch_state(self):
        if self._waiting_counter > 10:
            print('Switching to ApproachState')
            return ApproachState()
        return self
