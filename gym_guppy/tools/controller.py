from typing import Union

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

    def __init__(self, left: Union[float, np.ndarray] = .0, right: Union[float, np.ndarray] = .0, is_vel=True):
        self.is_vel = is_vel
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            assert isinstance(right, np.ndarray) and isinstance(left, np.ndarray), "either both, `left` and `right`, " \
                                                                                   "have to be of type float or both " \
                                                                                   "of type numpy.ndarray!"
            assert left.shape == right.shape, 'both, `left` and `right` need to have the same shape!'
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


class SaturatedPIDController:
    def __init__(self, p=1, i=1, d=1, slope=1.):
        self._p = p
        self._i = i
        self._d = d
        self._slope = slope

        self._i_error = .0
        self._i_error_max = 100.
        self._d_prev_error = .0

    def __call__(self, error):
        ctrl_term = 0
        if self._p:
            ctrl_term += self.p(error)
        if self._i:
            ctrl_term += self.i(error)
        if self._d:
            ctrl_term += self.d(error)

        return ctrl_term

    def p(self, error):
        s = sigmoid(error, self._slope)
        return self._p * s

    def i(self, error):
        self._i_error = np.minimum(self._i_error + error, self._i_error_max)
        return self._i * self._i_error

    def d(self, error):
        s = sigmoid(error, self._slope)
        err = s - self._d_prev_error
        self._d_prev_error = s
        return self._d * err


class ControllerBase:
    def __init__(self, speed=1., **pid_args):
        self._pid = SaturatedPIDController(**pid_args)
        self._speed = speed

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        assert .0 <= speed <= 1.
        self._speed = speed

    def __call__(self, ori_error, pos_error):
        raise NotImplementedError


class TurnController(ControllerBase):
    def __call__(self, ori_error, pos_error) -> MotorSpeeds:
        ctrl_term = self._pid(ori_error)
        s = ctrl_term * self._speed

        return MotorSpeeds(-s, s)


class ForwardController(ControllerBase):
    def __init__(self, ori_gate_slope=1., **kwargs):
        super(ForwardController, self).__init__(**kwargs)

        self._ori_gate_slope = ori_gate_slope
        self.prev_ori_error = .0

    def _ori_gate(self, ori_error):
        ori_error = np.abs(ori_error)
        p_part = 1 - sigmoid(ori_error, self._ori_gate_slope)

        curr_err = 1. - sigmoid(ori_error, self._ori_gate_slope)
        prev_err = 1. - sigmoid(self.prev_ori_error, self._ori_gate_slope)
        self.prev_ori_error = ori_error
        d_part = curr_err - prev_err
        # d_part = .0

        return p_part + d_part

    def __call__(self, ori_error, pos_error) -> MotorSpeeds:
        gate_value = self._ori_gate(ori_error)
        ctrl_term = self._pid(pos_error)
        s = gate_value * ctrl_term * self._speed

        return MotorSpeeds(s, s)


class TwoWheelsController:
    def __init__(self, ori_ctrl_params=None, fwd_ctrl_params=None):
        if ori_ctrl_params is None:
            ori_ctrl_params = {}
        if fwd_ctrl_params is None:
            fwd_ctrl_params = {}
        self._ori_ctrl = TurnController(**ori_ctrl_params)
        self._fwd_ctrl = ForwardController(**fwd_ctrl_params)

    @property
    def ori_ctrl(self):
        return self._ori_ctrl

    @property
    def fwd_ctrl(self):
        return self._fwd_ctrl

    def speeds(self, pose, target):
        ori_error, pos_error = _compute_errors(pose, target)

        turn_commands = self._ori_ctrl(ori_error, 100 * pos_error)
        # turn_commands.right = int(100*turn_commands.right)/100.
        fwd_commands = self._fwd_ctrl(ori_error, 100 * pos_error)

        return turn_commands + fwd_commands

    def speed_parts(self, pose, target):
        ori_error, pos_error = _compute_errors(pose, target)

        turn_commands = self._ori_ctrl(ori_error, 100 * pos_error)
        # turn_commands.right = int(100*turn_commands.right)/100.
        fwd_commands = self._fwd_ctrl(ori_error, 100 * pos_error)

        return turn_commands, fwd_commands
