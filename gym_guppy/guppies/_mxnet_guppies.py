import re
from abc import ABC
from typing import List

import h5py
import mxnet as mx
import numpy as np
from scipy.spatial.ckdtree import cKDTree

from gym_guppy.guppies import Agent, Guppy, TurnSpeedAgent
from gym_guppy.tools.math import compute_dist_bins, ray_casting_walls


class MXNetGuppy(Guppy, TurnSpeedAgent, ABC):
    _linear_damping = .0
    _angular_damping = .0

    def __init__(self, *, training_file, epoch, world_size_cm=100, model_fps=25, **kwargs):
        super().__init__(**kwargs)

        self._world_size_cm = float(world_size_cm)
        self._model_fps = float(model_fps)

        self._locomotion = np.array([[.0, .0]])

        with h5py.File(training_file, "r") as f:
            symbol_json = f.attrs["symbols"]

            locomotion_size = f.attrs["locomotion_size"]
            rc_agents_size = f.attrs["view_of_agents_size"]
            rc_walls_size = f.attrs["view_of_walls_size"]

            self._agents_sectors = f.attrs["view_of_agents_sectors"]
            self._wall_rays = f.attrs["view_of_walls_rays"]
            self._far_plane = f.attrs["far_plane"] / self._world_size_cm

            (turn_start, turn_stop, turn_size), (speed_start, speed_stop, speed_size) = f.attrs["locomotion"]
            self._turn_bins = np.linspace(turn_start, turn_stop, int(turn_size) + 1)
            self._speed_bins = np.linspace(speed_start, speed_stop, int(speed_size) + 1) / self._world_size_cm * self._model_fps

            params = {k: mx.ndarray.array(v) for k, v in f['params'][f"{epoch:04}"].items()}

        with mx.cpu() as ctx:
            self._symbols = mx.symbol.load_json(symbol_json)

            obs_shape = (1, locomotion_size + rc_agents_size + rc_walls_size)

            # infer shapes of recurrent inputs
            regex = re.compile(r'^(.*_)?state[0-9]+$')
            self._r_names = list(filter(regex.search, self._symbols.list_arguments()))
            _, out_shapes, _ = self._symbols.infer_shape_partial(feature=obs_shape)
            r_binds = dict(zip(self._r_names, out_shapes[-len(self._r_names):]))
            self._r_states = [v for k, v in params.items() if k in self._r_names]

            self._executor = self._symbols.simple_bind(ctx=ctx, grad_req='null', feature=obs_shape, **r_binds)
            self._executor.copy_params_from(params)

        self._last_pose = np.asarray(self.get_pose())

    def compute_next_action(self, state: List[Agent], kd_tree: cKDTree = None):
        pose = state[self.id, :]

        i = kd_tree.query_ball_point(pose[:2], r=self._far_plane)
        i = np.atleast_1d(i)

        rc_agents = compute_dist_bins(pose, state[i[1:], :], self._agents_sectors, self._far_plane).reshape(1, -1)
        rc_walls = ray_casting_walls(pose, self._world_bounds, self._wall_rays, self._far_plane).reshape(1, -1)
        rc_walls = np.maximum(rc_walls, .0)

        feature = np.concatenate((self._locomotion, rc_agents, rc_walls), axis=1)

        r_params = dict(zip(self._r_names, self._r_states))
        loc_turn, loc_speed, *self._r_states = self._executor.forward(feature=feature, **r_params)

        # sample from categorical distributions
        turn_idx = mx.random.multinomial(loc_turn).asscalar()
        self.turn = np.random.uniform(self._turn_bins[turn_idx], self._turn_bins[turn_idx+1])

        speed_idx = mx.random.multinomial(loc_speed).asscalar()
        self.speed = np.random.uniform(self._speed_bins[speed_idx], self._speed_bins[speed_idx + 1])

        self._locomotion[:] = self.turn, self.speed / self._model_fps * self._world_size_cm
