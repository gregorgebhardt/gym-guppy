import collections
from collections import deque

import numpy as np
import typing

from gym_guppy.guppies import Agent


class Feedback:
    def __init__(self, length=100, update_rate=10):
        self._update_rate = update_rate

        self._alpha_fear = .5
        self._alpha_follow = .5

        self._fear_value_recorder = deque(maxlen=length)
        self._fear_average_recorder = deque(maxlen=length)
        self._follow_value_recorder = deque(maxlen=length)
        self._follow_average_recorder = deque(maxlen=length)

        self._alphaE = 0.0025
        self._alphaO = 0.005

        self._fear_scaling = 8.0
        self._follow_scaling = 2.0

        self._guppy_history = PoseHistory(maxlen=length)
        self._robot_history = PoseHistory(maxlen=length)

        self._leading_zone = 0.28

        self._safe_speed = 0.03
        self._panic_speed = 0.2

    @property
    def fear(self):
        return self._alpha_fear

    @property
    def follow(self):
        return self._alpha_follow

    def update(self, guppy_pose: np.ndarray, robot_pose: np.ndarray):
        self._robot_history.store(robot_pose)
        self._guppy_history.store(guppy_pose)

        # compute distance
        dist = np.linalg.norm(robot_pose - guppy_pose)
        is_social = dist < self._leading_zone * 2

        if len(self._fear_value_recorder) == 0:
            self._fear_value_recorder.appendleft(0)
        if len(self._follow_value_recorder) == 0:
            self._follow_value_recorder.appendleft(0)

        e = is_social * self._get_fear() * self._fear_scaling
        o = is_social * self._get_follow() * self._follow_scaling
        e = self._alphaE * e + (1 - self._alphaE) * self._fear_value_recorder[0]
        o = self._alphaO * o + (1 - self._alphaO) * self._follow_value_recorder[0]

        self._fear_value_recorder.appendleft(e)
        self._follow_value_recorder.appendleft(o)

        self._alpha_fear = min(max(e, 0.0), 1.0)
        self._alpha_follow = min(max(o, 0.0), 1.0)

    def _get_fear(self):
        approach_dist = self._get_approach_dist()
        # crop approach dist to negative values and take the double absolute value
        approach_dist = -2. * min(approach_dist, 0.0)

        # normalize approach dist to range [safe_speed, panic_speed]
        return min(max(approach_dist - self._safe_speed, 0.0), self._panic_speed) / self._panic_speed

    def _get_follow(self):
        approach_dist = self._get_approach_dist()
        # crop approach dist to negative values and take the double absolute value
        approach_dist = 2. * max(approach_dist, 0.0)

        # multiply approach dist with exponential correction term
        approach_dist *= 1. + np.exp(-approach_dist / 3.)

        # normalize approach dist to range [safe_speed, panic_speed]
        return min(max(approach_dist - self._safe_speed, 0.0), self._panic_speed) / self._panic_speed

    def _get_approach_dist(self):
            guppy_now = self._guppy_history.positions[10]
            guppy_prv = self._guppy_history.positions[0]

            robot_prv = self._robot_history.positions[10]

            # movement vector of the fish
            guppy_vec = guppy_now - guppy_prv
            # fish-robot vector at previous time step
            guppy_robot_vec = robot_prv - guppy_prv

            # approach distance is the projection of the fish movement vector onto the line robot-fish
            return guppy_vec.dot(guppy_robot_vec) / np.linalg.norm(guppy_robot_vec)

    def infer_guppy_motion_direction(self, time_window=1):
        time_steps = int(time_window * self._update_rate)

        guppy_prv = self._guppy_history.positions[time_steps]
        guppy_now = self._guppy_history.positions[time_steps]

        guppy_vec = guppy_now - guppy_prv
        return guppy_vec / np.linalg.norm(guppy_vec)


class PoseHistory(collections.Iterable, collections.Sized):
    def __init__(self, maxlen):
        self._pose_history = deque(maxlen=maxlen)

        self._pose_iterable = PartIterable(self._pose_history, slice(None))
        self._position_iterable = PartIterable(self._pose_history, slice(None, 2))
        self._orientation_iterable = PartIterable(self._pose_history, 2)

    def __len__(self):
        return self._pose_history.__len__()

    def __iter__(self):
        return self._pose_history.__iter__()

    def store(self, pose):
        assert len(pose) == 3
        self._pose_history.appendleft(np.asarray(pose))

    @property
    def poses(self):
        return self._pose_iterable

    @property
    def positions(self):
        return self._position_iterable

    @property
    def orientations(self):
        return self._orientation_iterable


_I = typing.TypeVar('_I', typing.Iterable, typing.Sized)


class PartIterable:
    def __init__(self, iterable: _I, sl):
        self._iterable = iterable
        self._slice = sl
        self._idx = -1

    def __iter__(self):
        self._idx = -1
        return self

    def __next__(self):
        self._idx += 1
        if len(self._iterable) == self._idx:
            raise StopIteration
        return self._iterable[self._idx][self._slice]

    def __getitem__(self, item):
        return self._iterable[item][self._slice]