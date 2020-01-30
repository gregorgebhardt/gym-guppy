import abc
import types
import warnings
from typing import Union

import gym
import numpy as np
from Box2D import b2PolygonShape, b2Vec2, b2World
from scipy.spatial import cKDTree

from gym_guppy.bodies import Body, _world_scale
from gym_guppy.guppies import Agent, Guppy
from gym_guppy.tools.reward_function import RewardConst, RewardFunctionBase, reward_registry


class GuppyEnv(gym.Env, metaclass=abc.ABCMeta):
    metadata = {'render.modes': ['human', 'video', 'rgb_array']}

    world_size = world_width, world_height = 1., 1.
    # world_size = world_width, world_height = .5, .5
    screen_size = screen_width, screen_height = 800, 800

    _observe_objects = False
    _observe_light = True

    __sim_steps_per_second = 100
    __sim_velocity_iterations = 8
    __sim_position_iterations = 3
    __steps_per_action = 50
    _guppy_steps_per_action = 10

    def __init_subclass__(cls, **kwargs):
        pass

    def __new__(cls, *args, **kwargs):
        cls.sim_steps_per_second = cls.__sim_steps_per_second
        cls.sim_step = 1. / cls.__sim_steps_per_second
        cls.world_x_range = -cls.world_width / 2, cls.world_width / 2
        cls.world_y_range = -cls.world_height / 2, cls.world_height / 2
        cls.world_bounds = (np.array([-cls.world_width / 2, -cls.world_height / 2]),
                            np.array([cls.world_width / 2, cls.world_height / 2]))

        cls.__fps = cls.__sim_steps_per_second / cls.__steps_per_action
        cls.metadata['video.frames_per_second'] =  cls.__fps

        class WrappedCls(cls):
            def __init__(self, *_args, **_kwargs):
                super(WrappedCls, self).__init__(*_args, **_kwargs)

                # TODO: do we need to call this here?
                # TODO: BUG!!!
                self._reset()

                self.action_space = self._get_action_space()
                self.observation_space = self._get_observation_space()
                self.state_space = self._get_state_space()

                self._step_world()

        return super(GuppyEnv, cls).__new__(WrappedCls)

    def __init__(self, *args, **kwargs):
        super(GuppyEnv, self).__init__(*args, **kwargs)
        self.__sim_steps = 0
        self.__reset_counter = 0

        # create the world in Box2D
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.tank = self.world.CreateStaticBody(position=(.0, .0))

        wall_shape = b2PolygonShape()
        wall_width = .05
        wall_eps = .005

        for s in -1, 1:
            wall_shape.SetAsBox(wall_width / 2 * _world_scale,
                                (self.world_height / 2 + wall_width) * _world_scale,
                                b2Vec2((self.world_width / 2 + wall_width / 2 - wall_eps) * _world_scale * s, .0), .0)
            self.tank.CreateFixture(shape=wall_shape)
            wall_shape.SetAsBox((self.world_width / 2 + wall_width) * _world_scale,
                                wall_width / 2 * _world_scale,
                                b2Vec2(.0, (self.world_height / 2 + wall_width / 2 - wall_eps) * _world_scale * s), .0)
            self.tank.CreateFixture(shape=wall_shape)

        self.kd_tree = None

        self.__agents: [Agent] = []
        self.__robots_idx: [int] = []
        self.__guppies_idx: [int] = []
        # self.__robot: Union[Robot, None] = None
        # self.__guppies: [Guppy] = []

        self.__objects: [Body] = []

        self.__seed = 0

        self._screen = None
        self.render_mode = 'human'
        self.video_path = None

        self.set_reward(RewardConst(0.0))

        self.action_space = None
        self.observation_space = None
        self.state_space = None

        # remove, since we wrap now the __init__ function
        # self._configure_environment()
        #
        # self.action_space = self._get_action_space()
        # self.observation_space = self._get_observation_space()
        # self.state_space = self._get_state_space()
        #
        # self._step_world()

    @property
    def _sim_steps(self):
        return self.__sim_steps

    @property
    def robots(self):
        return (self.__agents[r_idx] for r_idx in self.__robots_idx)

    @property
    def robot(self):
        if len(self.__robots_idx):
            return self.__agents[self.__robots_idx[0]]
        else:
            return None

    @property
    def robots_idx(self):
        return tuple(self.__robots_idx)

    @property
    def num_robots(self):
        return len(self.__robots_idx)

    @property
    def guppies(self):
        return (self.__agents[g_idx] for g_idx in self.__guppies_idx)
        # return tuple(self.__guppies)

    @property
    def guppy_idx(self):
        return tuple(self.__guppies_idx)

    @property
    def num_guppies(self):
        return len(self.__guppies_idx)

    @property
    def objects(self):
        return tuple(self.__objects)

    def _get_action_space(self):
        actions_low = np.asarray([r.action_space.low for r in self.robots])
        actions_high = np.asarray([r.action_space.high for r in self.robots])

        return gym.spaces.Box(low=actions_low, high=actions_high)

    def _get_observation_space(self):
        return self._get_state_space()

    def _get_state_space(self):
        state_low = np.concatenate([self.world_bounds[0], [-np.inf]])
        state_low = np.tile(state_low, (self.num_robots + self.num_guppies, 1))
        state_high = np.concatenate([self.world_bounds[1], [np.inf]])
        state_high = np.tile(state_high, (self.num_robots + self.num_guppies, 1))

        return gym.spaces.Box(low=state_low, high=state_high)

    @property
    def _steps_per_action(self):
        return self.__steps_per_action

    @_steps_per_action.setter
    def _steps_per_action(self, spa: int):
        warnings.warn(f'Setting steps_per_action to {spa}')
        self.__steps_per_action = spa

    def __add_agent(self, agent: Agent):
        if agent in self.__agents:
            warnings.warn("Agent " + agent.id + " has already been registered before.")
            return False

        next_id = len(self.__agents)
        agent.set_id(next_id)
        self.__agents.append(agent)

        return True

    def _add_guppy(self, guppy: Guppy):
        if self.__add_agent(guppy):
            self.__guppies_idx.append(guppy.id)
            return True
        return False

    def _add_robot(self, robot: Agent):
        if self.__add_agent(robot):
            self.__robots_idx.append(robot.id)
            return True
        return False

    def _add_object(self, body: Body):
        self.__objects.append(body)

    def get_state(self):
        return np.array([a.get_state() for a in self.__agents])

    def get_robots_state(self):
        return self.get_indexed_state(self.__robots_idx)

    def get_guppies_state(self):
        return self.get_indexed_state(self.__guppies_idx)

    def get_indexed_state(self, index):
        return np.array([self.__agents[i].get_state() for i in index])

    def get_observation(self, state):
        return state

    def _reward_function(self, state, action, next_state):
        return .0

    def set_reward(self, reward_function: Union[RewardFunctionBase, str]):
        if isinstance(reward_function, RewardFunctionBase):
            # self._reward_function = reward_function
            self._reward_function = types.MethodType(reward_function, self)
        else:
            # self._reward_function = eval(reward_function, reward_registry)
            self._reward_function = types.MethodType(eval(reward_function, reward_registry), self)

    def get_reward(self, state, action, new_state):
        return self._reward_function(state, action, new_state)

    def has_finished(self, state, action):
        return False

    def get_info(self, state, action):
        return {}

    def destroy(self):
        del self.__objects[:]
        del self.__agents[:]
        self.__guppies_idx.clear()
        self.__robots_idx.clear()
        if self._screen is not None:
            del self._screen
            self._screen = None

    def close(self):
        self.destroy()

    def seed(self, seed=None):
        if seed is not None:
            self.__seed = seed
        return [self.__seed]

    @abc.abstractmethod
    def _reset(self):
        pass

    def reset(self):
        self.__reset_counter += 1
        self.destroy()
        self.__sim_steps = 0
        self._reset()

        # step to resolve
        self._step_world()

        return self.get_observation(self.get_state())

    def step(self, action: np.ndarray):
        # state before action is applied
        state = self.get_state()

        # action[:] = np.NaN
        action = np.atleast_2d(action)
        # apply action to robots
        for r, a in zip(self.robots, action):
            if r.action_space.contains(a):
                r.set_action(a)
            else:
                r.set_action(np.zeros(r.action_space.shape))

        while self._internal_sim_loop_condition():
            self._compute_guppy_actions(state)

            for a in self.__agents:
                a.step(self.sim_step)

            # step world
            self.world.Step(self.sim_step, self.__sim_velocity_iterations, self.__sim_position_iterations)
            # self.world.ClearForces()

            self.__sim_steps += 1

            if self._screen is not None and self.__sim_steps % 4 == 0:
                # self.render(self.render_mode, framerate=self.__sim_steps_per_second)
                self.render()

        # state
        next_state = self.get_state()
        # assert self.state_space.contains(next_state)

        # update KD-tree
        self._update_kdtree(next_state)

        # observation
        observation = self.get_observation(next_state)

        # reward
        reward = self.get_reward(state, action, next_state)

        # done
        done = self.has_finished(next_state, action)

        # info
        info = self.get_info(next_state, action)

        return observation, reward, done, info

    def _internal_sim_loop_condition(self):
        return self.__sim_steps > 0 and self.__sim_steps % self.__steps_per_action == 0

    def _update_kdtree(self, state):
        self.kd_tree = cKDTree(state[:, :2])

    def _compute_guppy_actions(self, state):
        if self.sim_step % self._guppy_steps_per_action == 0:
            for i, g in enumerate(self.guppies):
                g.compute_next_action(state=state, kd_tree=self.kd_tree)

    def _step_world(self):
        self.world.Step(self.sim_step, self.__sim_velocity_iterations, self.__sim_position_iterations)
        self.world.ClearForces()

        state = self.get_state()
        self._update_kdtree(state)

    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode
        else:
            self.render_mode = mode
        if mode is 'human':
            return self._render_human()
        elif mode is 'video':
            return self._render_human(display=False)
        elif mode is 'rgb_array':
            return self._render_rgb_array()

    def _render_rgb_array(self):
        rgb_array = np.ones(self.screen_size + (3,), dtype=np.uint8) * 255

        scale = np.divide(self.screen_size, self.world_size) * (.0, -1.)
        offset = self.world_bounds[0]

        for r in self.robots:
            r_state = r.get_state()
            col, row = np.int32(np.round((r_state[:2] - offset) * scale))
            for r in range(max(row-3, 0), min(row+3, self.screen_height)):
                for c in range(max(col-3, 0), min(col+3, self.screen_width)):
                    rgb_array[r, c, 1:3] = 0

        for g in self.guppies:
            g_state = g.get_state()
            col, row = np.int32(np.round((g_state[:2] - offset) * scale))
            for r in range(max(row - 3, 0), min(row + 3, self.screen_height)):
                for c in range(max(col - 3, 0), min(col + 3, self.screen_width)):
                    rgb_array[r, c, 0:2] = 0

        return rgb_array

    def _render_human(self, display=True):
        fps = self.__fps
        from gym_guppy.tools import rendering
        if self._screen is None:
            caption = self.spec.id if self.spec else ""
            if self.video_path:
                import os
                os.makedirs(self.video_path, exist_ok=True)
                _video_path = os.path.join(self.video_path, str(self.__reset_counter) + '.mp4')
            else:
                _video_path = None

            self._screen = rendering.GuppiesViewer(self.screen_width, self.screen_height, caption=caption,
                                                   fps=fps, display=display, record_to=_video_path)
            world_min, world_max = self.world_bounds
            self._screen.set_bounds(world_min[0], world_max[0], world_min[1], world_max[1])
        elif self._screen.close_requested():
            self._screen.close()
            self._screen = None
            # TODO how to handle this event?

        # render table
        x_min, x_max = self.world_x_range
        y_min, y_max = self.world_y_range
        self._screen.draw_polygon([(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max)],
                                  color=(255, 255, 255))
        # self._screen.draw_polyline([(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)],
        #                            width=.001)

        # allow to draw on table
        self._draw_on_table(self._screen)

        # render objects
        for o in self.__objects:
            o.draw(self._screen)

        # render guppies
        for a in self.__agents:
            a.draw(self._screen)

        # allow to draw on top
        self._draw_on_top(self._screen)

        self._screen.render()

    def _draw_on_table(self, screen):
        pass

    def _draw_on_top(self, screen):
        pass


class UnknownObjectException(Exception):
    pass


class InvalidArgumentException(Exception):
    pass
