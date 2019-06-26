import numpy as np
from typing import List

from numba import jit, njit
from scipy.spatial import cKDTree

from gym_guppy.guppies import TurnBoostAgent, ConstantVelocityAgent, Agent, Guppy

_ZOR = 0.005  # zone of repulsion
_ZOO = 0.09  # zone of orientation
_ZOA = 2.0  # zone of attraction
_FOP = np.deg2rad(270)  # field of perception

_WALL_NORMALS = np.array([[1, 0],
                          [0, 1],
                          [-1, 0],
                          [0, -1]])
_WALL_NORMALS_DIRECTION = np.deg2rad([0, 90, 180, -90])

@jit
def _compute_local_state(state):
    c, s = np.cos(state[0, 2]), np.sin(state[0, 2])
    R = np.array(((c, -s), (s, c)))
    local_positions = (state[:, :2] - state[0, :2]).dot(R)
    local_orientations = state[:, 2] - state[0, 2]
    # filter local neighbor orientations to [-π, π]
    local_orientations = (local_orientations + np.pi) % (2 * np.pi) - np.pi

    # compute polar coordinates
    dist = np.linalg.norm(local_positions, axis=1)
    phi = np.arctan2(local_positions[:, 1], local_positions[:, 0])

    return local_positions, local_orientations, dist, phi


@njit
def _compute_zone_indices(dist, phi, zor=_ZOR, zoo=_ZOO, zoa=_ZOA, field_of_perception=_FOP):
    i_fop = np.abs(phi) <= field_of_perception / 2.

    i_r = np.logical_and(0.0 < dist, dist <= zor)
    i_o = np.logical_and(np.logical_and(zor < dist, dist <= zor + zoo), i_fop)
    i_a = np.logical_and(np.logical_and(zor + zoo < dist, dist <= zoa), i_fop)

    return i_r, i_o, i_a, i_fop


@jit
def _compute_couzin_direction(local_positions, local_orientations, i_r, i_o, i_a):
    if np.any(i_r):
        # compute desired direction to evade fish in zor
        d_i = -1 * np.mean(local_positions[i_r] / np.linalg.norm(local_positions[i_r], axis=1, keepdims=True), axis=0)

        # compute angle between desired direction and own direction
        d_theta = np.arctan2(d_i[1], d_i[0])

    else:
        d_theta = .0
        denominator = .0

        # compute desired direction from fish in zoo
        if np.any(i_o):
            d_theta += local_orientations[i_o].mean(axis=0)
            denominator += 1.

        # compute desired direction from fish in zoa
        if np.any(i_a):
            d_ia = local_positions[i_a].mean(axis=0)
            d_theta += np.arctan2(d_ia[1], d_ia[0])

            d_theta /= denominator + 1.

    return d_theta


@njit
def _compute_couzin_boost(local_positions, max_boost, i_r, i_o, i_a):
    if np.any(i_r):
        d_boost = max_boost * i_r.sum() / len(local_positions)
    else:
        n_o = i_o.sum()
        n_a = i_a.sum()

        d_boost = (n_o + 1) * np.random.wald(0.05 * max_boost, 0.2 * max_boost)

        if n_a:
            d_boost += np.linalg.norm(np.sum(local_positions[i_a], axis=0)) * .2 * max_boost

        d_boost /= n_a + n_o + 1

    return d_boost


@njit
def _wall_repulsion(self_pos, self_theta, world_bounds, zor=_ZOR):
    theta_w = .0
    dist_to_walls = np.abs(world_bounds - self_pos)

    close_walls = dist_to_walls.flatten() < zor
    if np.any(close_walls):
        if np.argmin(dist_to_walls[:, 1]):
            sign = -1
        else:
            sign = 1
        theta_w += np.mean(sign * np.abs(_WALL_NORMALS_DIRECTION[close_walls])) - self_theta

    return theta_w


@jit
def _compute_couzin_action(state, world_bounds, zor=_ZOR, zoo=_ZOO, zoa=_ZOA, field_of_perception=_FOP):
    if len(state) == 1:
        return 0.0

    local_positions, local_orientations, dist, phi = _compute_local_state(state)

    # check for walls
    theta_w = _wall_repulsion(state[0, :2], state[0, 2], world_bounds, 5*zor)

    i_r, i_o, i_a, i_fop = _compute_zone_indices(dist, phi, zor, zoo, zoa, field_of_perception)

    theta_i = _compute_couzin_direction(local_positions, local_orientations, i_r, i_o, i_a)

    return theta_i + theta_w


@jit
def _compute_couzin_boost_action(state, world_bounds, max_boost, zor=_ZOR, zoo=_ZOO, zoa=_ZOA,
                                 field_of_perception=_FOP):
    local_positions, local_orientations, dist, phi = _compute_local_state(state)

    i_r, i_o, i_a, i_fop = _compute_zone_indices(dist, phi, zor, zoo, zoa, field_of_perception)

    theta_i = _compute_couzin_direction(local_positions, local_orientations, i_r, i_o, i_a)
    boost_i = _compute_couzin_boost(local_positions, max_boost, i_r, i_o, i_a)

    # check for walls
    theta_w = _wall_repulsion(state[0, :2], state[0, 2], world_bounds, 5*zor)
    if theta_w:
        theta_i += theta_w
        theta_i /= 2.

    return theta_i, boost_i


class ClassicCouzinGuppy(Guppy, ConstantVelocityAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._turn_noise = .1
        self._k_neighbors = 20

    def compute_next_action(self, state: List[Agent], kd_tree: cKDTree = None):
        d, i = kd_tree.query(state[self.id, :2], k=self._k_neighbors)
        self._turn = _compute_couzin_action(state[i, :], self._world_bounds) + np.random.randn() * self._turn_noise


class BoostCouzinGuppy(Guppy, TurnBoostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._turn_noise = .1
        self._boost_noise = .005
        self._k_neighbors = 20

    def compute_next_action(self, state: List[Agent], kd_tree: cKDTree = None):
        k = min(self._k_neighbors, len(state))
        d, i = kd_tree.query(state[self.id, :2], k=k)
        if k == 1:
            i = [i]

        d_theta, d_boost = _compute_couzin_boost_action(state[i, :], self._world_bounds, self._max_boost)
        # d_theta, d_boost = _compute_couzin_boost_action(state, self.id, self._max_boost)
        self._turn = d_theta + np.random.randn() * self._turn_noise
        # todo change noise
        self._boost = d_boost + np.abs(np.random.randn()) * self._boost_noise
