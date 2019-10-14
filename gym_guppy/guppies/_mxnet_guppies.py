import re
from abc import ABC
from typing import List

import h5py
import mxnet as mx
import numpy as np
from scipy.spatial.ckdtree import cKDTree

from gym_guppy.guppies import Agent, Guppy, TurnBoostAgent
from gym_guppy.tools.math import ray_casting_agents, ray_casting_walls


class MXNetGuppy(Guppy, TurnBoostAgent, ABC):
    def __init__(self, *, hdf_file, **kwargs):
        super().__init__(**kwargs)

        self._locomotion = [.0, .0]
        self._max_turn = 3.14
        self._max_boost = .2

        with h5py.File(hdf_file) as f:
            symbol_json = f.attrs.get('symbols')

            locomotion_size = f.attrs.get('locomotion_size')
            rc_agents_size = f.attrs.get('view_of_agents_size')
            rc_walls_size = f.attrs.get('view_of_walls_size')

            self._rays = f.attrs.get('view_of_walls_rays')
            self._far_plane = f.attrs.get('far_plane') / 100.

            turn_start, turn_stop, turn_size, speed_start, speed_stop, speed_size = f.attrs.get('locomotion')
            self._turn_bins = np.linspace(turn_start, turn_stop, turn_size)
            self._speed_bins = np.linspace(speed_start, speed_stop, speed_size)

            params = f['params']['0013']

        with mx.cpu() as ctx:
            self._symbols = mx.symbol.load_json(symbol_json)

            obs_shape = (1, locomotion_size + rc_agents_size + rc_walls_size)

            # infer shapes of recurrent inputs
            regex = re.compile(r'^(.*_)?state[0-9]+$')
            r_names = list(filter(regex.search, self._symbols.list_arguments()))
            _, out_shapes, _ = self._symbols.infer_shape_partial(feature=obs_shape)
            r_binds = dict(zip(r_names, out_shapes[-len(r_names):]))

            self._executor = self._symbols.simple_bind(ctx=ctx, grad_req='null', feature=obs_shape, **r_binds)
            self._executor.copy_params_from({k: mx.ndarray.array(v) for k, v in params.items()})

    def compute_next_action(self, state: List[Agent], kd_tree: cKDTree = None):
        pose = state[self.id, :]

        i = kd_tree.query_ball_point(pose[:2], r=self._far_plane)
        i = np.atleast_2d(i)

        assert self.id == i[0]

        # TODO check if ray casting delivers the same output as in Moritz code
        rc_agents = ray_casting_agents(pose, state[i[1:], :], self._rays, self._far_plane)
        rc_walls = ray_casting_walls(pose, self._world_bounds, self._rays, self._far_plane)
        rc_walls = np.maximum(rc_walls, .0)

        feature = np.concatenate(self._locomotion, rc_agents, rc_walls)

        loc_turn, loc_speed, *states = self._executor.forward(feature=feature)

        # sample from categorical distributions
        turn_idx = mx.random.multinomial(loc_turn)
        self.turn = np.random.uniform(self._turn_bins[turn_idx], self._turn_bins[turn_idx+1])

        speed_idx = mx.random.multinomial(loc_speed)
        self.boost = np.random.uniform(self._speed_bins[speed_idx], self._speed_bins[speed_idx + 1])

        self._locomotion = [self.turn, self.boost]
