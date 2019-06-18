import time
import warnings

import gym

import numpy as np
from scipy.spatial import cKDTree

from Box2D import b2World, b2ChainShape

from gym_guppy.guppies._couzin_guppies import ClassicCouzinGuppy, BoostCouzinGuppy
from ..bodies import Body, _world_scale
from ..guppies import Guppy

class GuppyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    world_size = world_width, world_height = 1., 1.
    # world_size = world_width, world_height = .5, .5
    screen_size = screen_width, screen_height = 800, 800

    _observe_objects = False
    _observe_light = True

    __sim_steps_per_second = 100
    __sim_velocity_iterations = 6
    __sim_position_iterations = 2
    __steps_per_action = 10

    def __new__(cls, **kwargs):
        cls.sim_steps_per_second = cls.__sim_steps_per_second
        cls.sim_step = 1. / cls.__sim_steps_per_second
        cls.world_x_range = -cls.world_width / 2, cls.world_width / 2
        cls.world_y_range = -cls.world_height / 2, cls.world_height / 2
        cls.world_bounds = (np.array([-cls.world_width / 2, -cls.world_height / 2]),
                            np.array([cls.world_width / 2, cls.world_height / 2]))

        return super(GuppyEnv, cls).__new__(cls)

    def __init__(self, **kwargs):
        self.__sim_steps = 0
        self.__reset_counter = 0

        # create the world in Box2D
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.tank = self.world.CreateStaticBody(position=(.0, .0))
        self.tank.CreateFixture(
            shape=b2ChainShape(vertices=[(_world_scale * self.world_x_range[0], _world_scale * self.world_y_range[1]),
                                         (_world_scale * self.world_x_range[0], _world_scale * self.world_y_range[0]),
                                         (_world_scale * self.world_x_range[1], _world_scale * self.world_y_range[0]),
                                         (_world_scale * self.world_x_range[1], _world_scale * self.world_y_range[1])]))

        self.kd_tree = None

        self.__robots: [Guppy] = []
        self.__guppies: [Guppy] = []
        self.__objects: [Body] = []

        self.__seed = 0

        self._screen = None
        self.render_mode = 'human'
        self.video_path = None

        self._configure_environment()

        self._step_world()

    @property
    def _sim_steps(self):
        return self.__sim_steps

    @property
    def robots(self):
        return tuple(self.__robots)

    @property
    def num_robots(self):
        return len(self.__robots)

    @property
    def guppies(self):
        return tuple(self.__guppies)

    @property
    def num_guppies(self):
        return len(self.__guppies)

    @property
    def objects(self):
        return tuple(self.__objects)

    @property
    def action_space(self):
        # TODO implement
        return None

    @property
    def observation_space(self):
        return NotImplemented

    @property
    def state_space(self):
        return NotImplemented

    @property
    def _steps_per_action(self):
        return self.__steps_per_action

    def _add_guppy(self, guppy: Guppy):
        if guppy in self.__guppies:
            warnings.warn("Guppy " + guppy.id + " has already been registered before. Can only register guppy once.")
            return False

        next_id = self.num_guppies
        guppy.set_id(next_id)
        self.__guppies.append(guppy)
        return True

    def _add_robot(self, robot: Guppy):
        if self._add_guppy(robot):
            self.__robots.append(robot)
            return True
        return False

    def _add_object(self, body: Body):
        self.__objects.append(body)

    # @abc.abstractmethod
    def _configure_environment(self):
        num_guppies = 50
        # random initialization
        positions = np.random.normal(loc=.0, scale=.05, size=(num_guppies, 2))
        orientations = np.random.rand(num_guppies) * 2 * np.pi - np.pi

        for p, o in zip(positions, orientations):
            # self._add_guppy(ClassicCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
            #                                    position=p, orientation=o))
            self._add_guppy(BoostCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
                                             position=p, orientation=o))

    def get_state(self):
        return np.array([g.get_state() for g in self.__guppies])

    def get_observation(self):
        return self.get_state()

    # @abc.abstractmethod
    def get_reward(self, state, action, new_state):
        return None
        # raise NotImplementedError

    def has_finished(self, state, action):
        return False

    def get_info(self, state, action):
        return ""

    def destroy(self):
        del self.__objects[:]
        del self.__guppies[:]
        del self.__robots[:]
        if self._screen is not None:
            del self._screen
            self._screen = None

    def close(self):
        self.destroy()

    def seed(self, seed=None):
        if seed is not None:
            self.__seed = seed
        return [self.__seed]

    def reset(self):
        self.__reset_counter += 1
        self.destroy()
        self._configure_environment()
        self.__sim_steps = 0

        # step to resolve
        self._step_world()

        return self.get_observation()

    def step(self, action: np.ndarray):
        # if self.action_space and action is not None:
        #     assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # state before action is applied
        state = self.get_state()

        # apply action to robots
        for r, a in zip(self.__robots, action):
            r.set_action(a)

        # step guppies
        for i, g in enumerate(self.__guppies):
            g.compute_next_action(state=state, kd_tree=self.kd_tree)

        for i in range(self.__steps_per_action):
            for g in self.__guppies:
                g.step(self.sim_step)

            # step world
            self.world.Step(self.sim_step, self.__sim_velocity_iterations, self.__sim_position_iterations)

            self.__sim_steps += 1

            if self._screen is not None and self.__sim_steps % 4 == 0:
                # self.render(self.render_mode, framerate=self.__sim_steps_per_second)
                self.render(self.render_mode)

        # state
        next_state = self.get_state()

        # observation
        observation = self.get_observation()

        # reward
        reward = self.get_reward(state, action, next_state)

        # done
        done = self.has_finished(next_state, action)

        # info
        info = self.get_info(next_state, action)

        # update KD-tree
        self.kd_tree = cKDTree(next_state[:, :2])

        return observation, reward, done, info

    def _step_world(self):
        self.world.Step(self.sim_step, self.__sim_velocity_iterations, self.__sim_position_iterations)
        self.world.ClearForces()

        state = self.get_state()
        self.kd_tree = cKDTree(state[:, :2])

    def render(self, mode=None, fps=25):
        # if close:
        #     if self._screen is not None:
        #         self._screen.close()
        #         self._screen = None
        #     return
        if mode is None:
            mode = self.render_mode

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
                                                   fps=fps, display=mode == 'human', record_to=_video_path)
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
        for g in self.__guppies:
            g.draw(self._screen)

        # render robots
        for r in self.__robots:
            r.draw(self._screen)

        # allow to draw on top
        self._draw_on_top(self._screen)

        self._screen.render()

    def get_objects(self) -> [Body]:
        return self.__objects

    def get_guppies(self) -> [Guppy]:
        return self.__guppies

    def _draw_on_table(self, screen):
        pass

    def _draw_on_top(self, screen):
        pass


class UnknownObjectException(Exception):
    pass


class InvalidArgumentException(Exception):
    pass
