import abc
from collections import namedtuple
from copy import copy

import numpy as np

from gym_guppy.tools.math import sigmoid


def _compute_errors(pose, target):
    local_target_vec = target[:2] - pose[:2]
    target_phi = np.arctan2(local_target_vec[1], local_target_vec[0])
    ori_error = target_phi - pose[2]
    ori_error = (ori_error + np.pi) % (2 * np.pi) - np.pi
    pos_error = np.linalg.norm(local_target_vec)

    return ori_error, pos_error


class MotorSpeeds:
    _robot_radius = (.09133 + .084) / 2. / 2.
    _wheel_radius = .032
    _wheel_circumference = _wheel_radius * 2. * np.pi
    _max_rps = 164.179 / 60.
    _max_pwm = 1023.
    _max_vel = _max_rps * 2. * _wheel_radius * np.pi
    _speed_clip = .35
    _pwm_clip = _speed_clip / _wheel_circumference / _max_rps * _max_pwm

    def __init__(self, left=.0, right=.0, is_vel=True):
        self.is_vel = is_vel
        self.left = left
        self.right = right

    @classmethod
    def _vel_to_pwm(cls, vel):
        rps = vel / cls._wheel_circumference
        pwm = np.clip(rps, -cls._max_rps, cls._max_rps) / cls._max_rps * cls._max_pwm

        return np.clip(pwm, -cls._pwm_clip, cls._pwm_clip)
        # return np.clip(pwm, -255, 255)

    @classmethod
    def _pwm_to_vel(cls, pwm):
        rps = pwm / cls._max_pwm * cls._max_rps
        vel = rps * cls._wheel_circumference

        return np.clip(vel, -cls._max_vel, cls._max_vel)

    def vel_to_pwm(self):
        assert self.is_vel
        pwm_left = self._vel_to_pwm(self.left)
        pwm_right = self._vel_to_pwm(self.right)
        return MotorSpeeds(pwm_left, pwm_right, False)

    def pwm_to_vel(self):
        assert not self.is_vel
        vel_left = self._pwm_to_vel(self.left)
        vel_right = self._pwm_to_vel(self.right)
        return MotorSpeeds(vel_left, vel_right)

    def get_local_velocities(self):
        # l_phi_d = self._get_rot_speed_from_pwm(self.left)
        # r_phi_d = self._get_rot_speed_from_pwm(self.right)
        #
        # x_vel = self._wheel_radius * (l_phi_d + r_phi_d) / 2
        # r_vel = self._wheel_radius * (r_phi_d - l_phi_d) / (2 * self._robot_radius)
        x_vel = (self.left + self.right) / 2.
        r_vel = (self.right - self.left) / (2. * self._robot_radius)

        return x_vel, r_vel

    def __add__(self, other):
        assert self.is_vel == other.is_vel
        return MotorSpeeds(self.left + other.left, self.right + other.right)

    def __mul__(self, other):
        if isinstance(other, MotorSpeeds):
            assert self.is_vel == other.is_vel
            return MotorSpeeds(self.left * other.left, self.right * other.right)
        else:
            return MotorSpeeds(self.left * other, self.right * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def invert_gears(self):
        return MotorSpeeds(self.right, self.left)


class BaseController(abc.ABC):
    def __init__(self, p=1., i=1., d=.0, slope=1., speed=1., p_error_factor=1., d_error_factor=1.):
        super().__init__()

        # TODO: set correct parameters
        self._p = p
        self._i = i  # 1.
        self._d = d  # 1.

        self._inv_sigmoid = True
        self._slope = slope
        self._speed = speed

        self._p_error_factor = p_error_factor

        self._d_error_factor = d_error_factor
        self._prev_error = .0

        self._i_error = .0
        self._i_error_max = 1.

    @abc.abstractmethod
    def speeds(self, ori_error, pos_error) -> MotorSpeeds:
        pass

    def ctrl_p(self, error):
        s = sigmoid(error * self._p_error_factor, self._slope)
        if self._inv_sigmoid:
            s = 1. - s

        return self._p * s

    def ctrl_i(self, error):
        # self._i_error = np.minimum(self._i_error + error, self._i_error_max)
        return self._i * self._i_error

    def ctrl_d(self, error):
        # TODO: why is prev_error not scaled by d_error_factor?
        s = sigmoid(error * self._d_error_factor, self._slope)
        err = sigmoid(self._prev_error, self._slope)
        if self._inv_sigmoid:
            s = 1. - s
            err = 1. - err

        self._prev_error = error

        return self._d * (s - err)


class OrientationController(BaseController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # c.f. robocontrol::ActionProcessing::TwoWheelsControl.cpp:13
        self._inv_sigmoid = False

    def speeds(self, ori_error, pos_error):
        ctrl_term = 0
        if self._p:
            ctrl_term += self.ctrl_p(ori_error)
        if self._i:
            ctrl_term += self.ctrl_i(ori_error)
        if self._d:
            ctrl_term += self.ctrl_d(ori_error)

        return MotorSpeeds(-ctrl_term * self._speed, ctrl_term * self._speed)


class ForwardController(BaseController):
    def __init__(self, p_dist_error_factor=1., **kwargs):
        super().__init__(**kwargs)

        self._p_dist_error_factor = p_dist_error_factor

        # see robocontrol::ActionProcessing::TwoWheelsControl.cpp:14
        self._inv_sigmoid = True

        self._last_pos_error = None
        self._i_error_dist = .0

    def speeds(self, ori_error, pos_error) -> MotorSpeeds:
        ori_error = np.abs(ori_error)
        ori_ctrl_term = 0
        if self._p:
            ori_ctrl_term += self.ctrl_p(ori_error)
        if self._i:
            ori_ctrl_term += self.ctrl_i(ori_error)
        if self._d:
            ori_ctrl_term += self.ctrl_d(ori_error)

        dist_ctrl_term = 0
        if self._p:
            dist_ctrl_term += self.ctrl_p_dist(pos_error)
        if self._i:
            dist_ctrl_term += self.ctrl_i_dist(pos_error)
        if self._d:
            dist_ctrl_term += self.ctrl_d_dist(pos_error)
        ctrl_term = ori_ctrl_term * dist_ctrl_term

        return MotorSpeeds(ctrl_term * self._speed, ctrl_term * self._speed)

    def ctrl_p_dist(self, error):
        return self._p * sigmoid(error * self._p_dist_error_factor, self._slope)

    def ctrl_i_dist(self, error):
        # self._i_error_dist = np.minimum(self._i_error_dist + error, self._i_error_dist_max)
        return self._i * self._i_error

    def ctrl_d_dist(self, error):
        if self._last_pos_error is None:
            return .0
        d = self._d * (sigmoid(error * self._p_dist_error_factor, self._slope) - sigmoid(self._last_pos_error * self._p_dist_error_factor, self._slope))
        self._last_pos_error = error

        return d


class TwoWheelsController:
    def __init__(self, ori_ctrl_params=None, fwd_ctrl_params=None):
        if ori_ctrl_params is None:
            ori_ctrl_params = {}
        if fwd_ctrl_params is None:
            fwd_ctrl_params = {}
        self._ori_ctrl = OrientationController(**ori_ctrl_params)
        self._fwd_ctrl = ForwardController(**fwd_ctrl_params)

    def speeds(self, pose, target):
        ori_error, pos_error = _compute_errors(pose, target)
        # ori_error = np.rad2deg(ori_error)

        turn_commands = self._ori_ctrl.speeds(ori_error, pos_error)
        fwd_commands = self._fwd_ctrl.speeds(ori_error, pos_error)

        return turn_commands + fwd_commands

    def speed_parts(self, pose, target):
        ori_error, pos_error = _compute_errors(pose, target)
        # ori_error = np.rad2deg(ori_error)

        turn_commands = self._ori_ctrl.speeds(ori_error, pos_error)
        fwd_commands = self._fwd_ctrl.speeds(ori_error, pos_error)

        return turn_commands, fwd_commands

