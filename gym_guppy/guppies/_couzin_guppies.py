import abc

import numpy as np
from typing import List

from numba import jit, njit
from scipy.spatial import cKDTree

from gym_guppy.guppies import TurnBoostAgent, ConstantVelocityAgent, Agent, Guppy
from gym_guppy.tools import Feedback
from gym_guppy.tools.math import rotation, row_norm

_ZOR = 0.005  # zone of repulsion
_ZOO = 0.09  # zone of orientation
_ZOA = 0.4  # zone of attraction
_FOP = np.deg2rad(320)  # field of perception
_ZOI = _ZOR + _ZOO + _ZOA

_WALL_NORMALS = np.array([[1, 0],
                          [0, 1],
                          [-1, 0],
                          [0, -1]])
_WALL_NORMALS_DIRECTION = np.deg2rad([0, 90, 180, -90])


@njit(fastmath=True)
def _compute_local_state(state):
    # c, s = np.cos(state[0, 2]), np.sin(state[0, 2])
    # R = np.array(((c, -s), (s, c)))
    R = rotation(state[0, 2])
    local_positions = (state[:, :2] - state[0, :2]).dot(R)
    local_orientations = state[:, 2] - state[0, 2]
    # filter local neighbor orientations to [-π, π]
    local_orientations = (local_orientations + np.pi) % (2 * np.pi) - np.pi

    # compute polar coordinates
    # dist = np.linalg.norm(local_positions, axis=1)
    dist = row_norm(local_positions)
    phi = np.arctan2(local_positions[:, 1], local_positions[:, 0])

    return local_positions, local_orientations, dist, phi


@njit(fastmath=True)
def _compute_zone_indices(dist, phi, zor=_ZOR, zoo=_ZOO, zoa=_ZOA, field_of_perception=_FOP):
    i_fop = np.abs(phi) <= field_of_perception / 2.

    i_r = np.logical_and(0.0 < dist, dist <= zor)
    i_o = np.logical_and(np.logical_and(zor < dist, dist <= zor + zoo), i_fop)
    i_a = np.logical_and(np.logical_and(zor + zoo < dist, dist <= zor + zoo + zoa), i_fop)
    # print(f"i_fop: {i_fop} n_r: {sum(i_r)}; n_o: {sum(i_o)}; n_a: {sum(i_a)} ", end='')

    return i_r, i_o, i_a, i_fop


@njit(fastmath=True)
def _compute_couzin_direction(local_positions, local_orientations, i_r, i_o, i_a):
    if np.any(i_r):
        # compute desired direction to evade fish in zor
        d_i = -1 * np.sum(local_positions[i_r] / (np.reshape(row_norm(local_positions[i_r]), (-1, 1)) + 1e-8), axis=0)
        # length not important for angle
        # d_i /= len(i_r)

        # compute angle between desired direction and own direction
        d_theta = np.arctan2(d_i[1], d_i[0])

    else:
        d_theta = .0
        denominator = .0

        # compute desired direction from fish in zoo
        if np.any(i_o):
            # length not important for angle
            # d_theta += np.sum(local_orientations[i_o], axis=0) / len(i_o)
            d_theta += np.sum(local_orientations[i_o], axis=0) / len(i_o)
            denominator += 1.

        # compute desired direction from fish in zoa
        if np.any(i_a):
            # length not important for angle
            # d_ia = np.sum(local_positions[i_a], axis=0) / len(i_a)
            d_ia = np.sum(local_positions[i_a], axis=0)
            d_theta += np.arctan2(d_ia[1], d_ia[0])

            d_theta /= denominator + 1.

        if len(i_a) + len(i_o) == 0:
            noise = np.random.normal() * .5
            # print(f"adding noise: {noise} ", end='')
            d_theta += noise

    # print(f"d_theta: {d_theta} ", end='')

    return d_theta


@njit(fastmath=True)
def _compute_couzin_boost(local_positions, max_boost, i_r, i_o, i_a, approach_norm=1.):
    if np.any(i_r):
        # d_boost = 0.5 * max_boost * i_r.sum() / len(local_positions)
        d_boost = max_boost
    elif np.any(i_o) or np.any(i_a):
        n_o = i_o.sum()
        n_a = i_a.sum()

        d_boost = n_o * np.random.wald(0.001, 0.05)

        if n_a:
            d_boost += np.linalg.norm(np.sum(local_positions[i_a], axis=0)) / approach_norm * max_boost * 0.5

        d_boost /= n_a + n_o
    else:
        d_boost = .0

    d_boost += np.random.wald(0.0005, 0.05)

    return d_boost


@njit(fastmath=True)
def _wall_repulsion(self_pos, self_theta, world_bounds, zor=_ZOR):
    theta_w = .0
    dist_to_walls = np.abs(world_bounds - self_pos)

    close_walls = dist_to_walls.flatten() < zor
    if np.any(close_walls):
        # TODO check this for upper left and lower right corner, could be buggy!!
        if np.argmin(dist_to_walls[:, 1]):
            sign = -1
        else:
            sign = 1
        theta_w += np.mean(sign * np.abs(_WALL_NORMALS_DIRECTION[close_walls])) - self_theta

    # print(f"theta_w: {theta_w} ", end='')
    return theta_w


@njit(fastmath=True)
def _compute_couzin_action(state, world_bounds, zor=_ZOR, zoo=_ZOO, zoa=_ZOA, field_of_perception=_FOP):
    if len(state) == 1:
        return 0.0

    local_positions, local_orientations, dist, phi = _compute_local_state(state)

    # check for walls
    theta_w = _wall_repulsion(state[0, :2], state[0, 2], world_bounds, 5 * zor)

    i_r, i_o, i_a, i_fop = _compute_zone_indices(dist, phi, zor, zoo, zoa, field_of_perception)

    theta_i = _compute_couzin_direction(local_positions, local_orientations, i_r, i_o, i_a)

    return theta_i + theta_w


@njit(fastmath=True)
def _compute_couzin_boost_action(state, world_bounds, max_boost, zor=_ZOR, zoo=_ZOO, zoa=_ZOA,
                                 field_of_perception=_FOP):
    local_positions, local_orientations, dist, phi = _compute_local_state(state)

    i_r, i_o, i_a, i_fop = _compute_zone_indices(dist, phi, zor, zoo, zoa, field_of_perception)

    theta_i = _compute_couzin_direction(local_positions, local_orientations, i_r, i_o, i_a)
    boost_i = _compute_couzin_boost(local_positions, max_boost, i_r, i_o, i_a, zor + zoo + zoa)

    # check for walls
    theta_w = _wall_repulsion(state[0, :2], state[0, 2], world_bounds, .03)
    if theta_w:
        theta_i += theta_w
        theta_i /= 2.

    # print(f"theta_i: {theta_i} ", end='')

    return theta_i, boost_i


class BaseCouzinGuppy(Guppy, abc.ABC):
    @property
    def zor(self):
        return _ZOR

    @property
    def zoo(self):
        return _ZOO

    @property
    def zoa(self):
        return _ZOA

    @property
    def zoi(self):
        return _ZOI

    @property
    def couzin_zones(self):
        return _ZOR, _ZOO, _ZOA


class ClassicCouzinGuppy(BaseCouzinGuppy, ConstantVelocityAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._turn_noise = .1
        self._k_neighbors = 20

    def compute_next_action(self, state: np.ndarray, kd_tree: cKDTree = None):
        d, i = kd_tree.query(state[self.id, :2], k=self._k_neighbors)
        self._turn = _compute_couzin_action(state[i, :], self._world_bounds) + np.random.randn() * self._turn_noise


class BoostCouzinGuppy(BaseCouzinGuppy, TurnBoostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._turn_noise = .1
        self._k_neighbors = 20
        self._boost_noise = .005

    def compute_next_action(self, state: np.ndarray, kd_tree: cKDTree = None):
        k = min(self._k_neighbors, len(state))
        d, i = kd_tree.query(state[self.id, :2], k=k)
        if k == 1:
            i = [i]

        out = _compute_couzin_boost_action(state[i, :], self._world_bounds, self._max_boost_per_step)
        d_theta, d_boost = out
        # d_theta, d_boost = _compute_couzin_boost_action(state, self.id, self._max_boost)
        theta_noise = np.minimum(np.maximum(np.random.randn() * self._turn_noise, -.3), .3)
        # print(f'theta_noise: {theta_noise}')
        self.turn = d_theta + theta_noise
        # there is already noise on boost when no guppy in z_r
        self.boost = d_boost


class AdaptiveCouzinGuppy(BoostCouzinGuppy):
    def __init__(self, *,
                 initial_zone_factor=.95,
                 max_zone_factor=1.,
                 min_zone_factor=.15,
                 zoo_factor=2.,
                 zone_radius_mean=.3,
                 zone_radius_noise=.01,
                 zone_grow_factor=1.05,
                 zone_shrink_factor=.99,
                 unknown_agents: List[Agent] = None, **kwargs):
        super().__init__(**kwargs)

        if unknown_agents is None:
            self._unknown_agents = []
        else:
            self._unknown_agents = unknown_agents
        self._unknown_agents_ids = [a.id for a in self._unknown_agents]

        # currently not used
        self._feedback = [Feedback() for _ in self._unknown_agents]

        # zone factors for computing the couzin zones towards the unknown agents
        self._initial_zone_factor = initial_zone_factor
        self._max_zone_factor = max_zone_factor
        self._min_zone_factor = min_zone_factor
        self._zoo_factor = zoo_factor

        # add some noise to the zone radius
        self._zone_radius_mean = zone_radius_mean
        self._zone_radius_noise = zone_radius_noise

        # zone dynamics
        self._adaptive_zone_grow_factor = zone_grow_factor
        self._adaptive_zone_shrink_factor = zone_shrink_factor

        # initialize zones
        self._adaptive_zone_factors = np.array([self._initial_zone_factor] * len(self._unknown_agents))
        self._zone_radius = self._zone_radius_mean + np.random.rand() * self._zone_radius_noise

        self._update_zones()

    def _update_zones(self):
        zor = self._zone_radius * self._adaptive_zone_factors
        zoo = (self._zone_radius - zor) * min(self._adaptive_zone_factors * self._zoo_factor, self._max_zone_factor)
        zoa = self._zone_radius - zoo - zor

        self._zone_cache = zor, zoo, zoa

    @property
    def couzin_zones(self):
        return self.zor, self.zoo, self.zoa

    def compute_next_action(self, state: np.ndarray, kd_tree: cKDTree = None):
        k = min(self._k_neighbors, len(state))
        _, sorted_state_i = kd_tree.query(state[self.id, :2], k=k)
        if k == 1:
            sorted_state_i = [sorted_state_i]

        # compute couzin actions for known agents
        known_agents_ids = [i for i in sorted_state_i if i not in self._unknown_agents_ids]
        known_agents_state = state[known_agents_ids]
        d_theta_known, d_boost_known = _compute_couzin_boost_action(known_agents_state, self._world_bounds,
                                                                    self._max_boost)
        d_theta_known += np.random.randn() * self._turn_noise

        unknown_agents_ids = [i for i in sorted_state_i if i in self._unknown_agents_ids]
        unknown_agents_state = state[unknown_agents_ids]

        # compute zones with current factors
        self._update_zones()

        # adapt zone factors
        for ua_id, ua_state in zip(unknown_agents_ids, unknown_agents_state):
            zor = self._zone_cache[0][ua_id]
            # if robot is in fish's zor increase zor
            if np.linalg.norm(state[ua_id, :2] - state[self.id, :2]) < zor:
                self._adaptive_zone_factors[ua_id] = min(
                    self._max_zone_factor, self._adaptive_zone_factors[ua_id] * self._adaptive_zone_grow_factor)
            # else decrease zor
            else:
                self._adaptive_zone_factors[ua_id] = max(
                    self._min_zone_factor, self._adaptive_zone_shrink_factor * self._adaptive_zone_factors[ua_id])

        self._update_zones()
        d_theta_unknown, d_boost_unknown = .0, .0
        for i, (ua_id, zor, zoo, zoa) in enumerate(zip(self._unknown_agents_ids, *self._zone_cache)):
            d_theta_i, d_boost_i = _compute_couzin_boost_action(state[[self.id, ua_id]], self._world_bounds,
                                                                self._max_boost, zor, zoo, zoa)
            d_theta_unknown += d_theta_i
            d_boost_unknown += d_boost_i

        # combine desired movements
        # self.turn = d_theta_known * len(known_agents_ids) + d_theta_unknown
        # self.turn /= len(self._unknown_agents) + len(known_agents_ids)
        # self.boost = d_boost_known * len(known_agents_ids) + d_boost_unknown
        # self.boost /= len(self._unknown_agents) + len(known_agents_ids)

        if abs(d_theta_known) > abs(d_theta_unknown):
            self.turn = d_theta_known
        else:
            self.turn = d_theta_unknown
        self.boost = max(d_boost_known, d_boost_unknown)

    @property
    def zor(self):
        return self._zone_cache[0].squeeze()

    @property
    def zoo(self):
        return self._zone_cache[1].squeeze()

    @property
    def zoa(self):
        return self._zone_cache[2].squeeze()

    @property
    def zoi(self):
        return np.sum(self._zone_cache, axis=-1).squeeze()


class BiasedAdaptiveCouzinGuppy(AdaptiveCouzinGuppy):
    def __init__(self, *, attraction_points=None, repulsion_points=None, bias_gain=.1, **kwargs):
        super(BiasedAdaptiveCouzinGuppy, self).__init__(**kwargs)

        self.attraction_points = attraction_points
        self.repulsion_points = repulsion_points
        self.bias_gain = bias_gain

    def compute_next_action(self, state: np.ndarray, kd_tree: cKDTree = None):
        super(BiasedAdaptiveCouzinGuppy, self).compute_next_action(state, kd_tree)

        turn_bias = .0

        if self.attraction_points is not None:
            for ap in self.attraction_points:
                local_ap = self.get_local_point(ap)
                ap_r = np.linalg.norm(local_ap)
                ap_th = np.arctan2(local_ap[1], local_ap[0])

                if _ZOA * .5 >= ap_r >= _ZOR:
                    turn_bias += np.sign(ap_th) * self.bias_gain * self._max_turn_per_step * (1 - ap_r)

        if self.repulsion_points is not None:
            for rp in self.repulsion_points:
                local_rp = self.get_local_point(rp)
                rp_r = np.linalg.norm(local_rp)
                rp_th = np.arctan2(local_rp[1], local_rp[0])

                if rp_r <= _ZOA:
                    turn_bias += -1 * np.sign(rp_th) * self.bias_gain * self._max_turn_per_step * (1 - rp_r)

        self.turn += turn_bias
