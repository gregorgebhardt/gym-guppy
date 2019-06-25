import collections
from collections import deque

import numpy as np
import typing

from gym_guppy.guppies import Agent


class Feedback:
    def __init__(self):
        self._alpha_fear = .5
        self._alpha_follow = .5

        self._fear_value_recorder = BoundedDeque(maxlen=500)
        self._fear_average_recorder = BoundedDeque(maxlen=500)
        self._follow_value_recorder = BoundedDeque(maxlen=500)
        self._follow_average_recorder = BoundedDeque(maxlen=500)

        self._alphaE = 0.0025
        self._alphaO = 0.005

        self._fear_scaling = 8.0
        self._follow_scaling = 2.0

        self._guppy_history = PoseHistory(maxlen=500)
        self._robot_history = PoseHistory(maxlen=500)

        self._leading_zone = 0.28

        self._safe_speed = 0.03
        self._panic_speed = 0.2

    def update(self, guppy: Agent, robot: Agent):
        robot_pose = robot.get_pose()
        guppy_pose = guppy.get_pose()

        self._robot_history.append(robot_pose)
        self._guppy_history.append(guppy_pose)

        # compute distance
        dist = np.linalg.norm(robot_pose - guppy_pose)
        is_social = dist < self._leading_zone * 2

        if len(self._fear_value_recorder) == 0:
            self._fear_value_recorder.append(0)
        if len(self._follow_value_recorder) == 0:
            self._follow_value_recorder.append(0)

        e = is_social * self.get_fear() * self._fear_scaling
        o = is_social * self.get_follow() * self._follow_scaling
        e = self._alphaE * e + (1 - self._alphaE) * self._fear_value_recorder[-1]
        o = self._alphaO * o + (1 - self._alphaO) * self._follow_value_recorder[-1]

        self._fear_value_recorder.append(e)
        self._follow_value_recorder.append(o)

        self._alpha_fear = min(max(e, 0.0), 1.0)
        self._alpha_follow = min(max(o, 0.0), 1.0)

    def get_fear(self):
        approach_dist = self.get_approach_dist()
        # crop approach dist to negative values and take the double absolute value
        approach_dist = -2. * min(approach_dist, 0.0)

        # normalize approach dist to range [safe_speed, panic_speed]
        return min(max(approach_dist - self._safe_speed, 0.0), self._panic_speed) / self._panic_speed

    def get_follow(self):
        approach_dist = self.get_approach_dist()
        # crop approach dist to negative values and take the double absolute value
        approach_dist = 2. * max(approach_dist, 0.0)

        # multiply approach dist with exponential correction term
        approach_dist *= 1. + np.exp(-approach_dist / 3.)

        # normalize approach dist to range [safe_speed, panic_speed]
        return min(max(approach_dist - self._safe_speed, 0.0), self._panic_speed) / self._panic_speed

    def get_approach_dist(self):
            guppy_now = self._guppy_history.positions[0]
            guppy_prv = self._guppy_history.positions[-1]

            robot_prv = self._robot_history.positions[-1]

            # movement vector of the fish
            fish_vector = guppy_now - guppy_prv
            # fish-robot vector at previous time step
            fish_robot_vector = robot_prv - guppy_prv

            # approach distance is the projection of the fish movement vector onto the line robot-fish
            return fish_vector.dot(fish_robot_vector) / np.linalg.norm(fish_robot_vector)


class BoundedDeque(deque):
    def append(self, x) -> None:
        if len(self) >= self.maxlen:
            self.popleft()
        super(BoundedDeque, self).append(x)

    def appendleft(self, x) -> None:
        if len(self) >= self.maxlen:
            self.pop()
        super(BoundedDeque, self).appendleft(x)


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

    def append(self, pose):
        if len(self._pose_history) == self._pose_history.maxlen:
            self._pose_history.popleft()

        self._pose_history.extend(np.asarray(pose))

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