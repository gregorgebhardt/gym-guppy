import abc
from typing import List

import numpy as np
from Box2D import b2Vec2
from scipy.spatial import cKDTree

from gym_guppy.bodies import FishBody


class Agent(FishBody, abc.ABC):
    _max_linear_velocity = 0.2  # meters / s
    _max_angular_velocity = 0.5 * np.pi  # radians / s

    # adjust these parameters
    _density = 0.00025
    _friction = 0.0
    _restitution = 0.0

    _linear_damping = 15.  # * _world_scale
    _angular_damping = 60.  # * _world_scale

    def __init__(self, *, world_bounds, **kwargs):
        # all parameters in real world units
        super().__init__(length=0.02, width=0.004, **kwargs)
        self._body.bullet = True

        self._world_bounds = np.asarray(world_bounds)

        # will be set by the environment
        self._id = None

        # put all guppies into same group (negative so that no collisions are detected)
        self._fixtures[0].filterData.groupIndex = -1

        self._color = np.array((133, 133, 133))
        self._highlight_color = (255, 255, 255)

    def set_id(self, id):
        self._id = id

    @property
    def id(self):
        return self._id

    def set_color(self, color):
        self._highlight_color = color

    @abc.abstractmethod
    def step(self, time_step):
        pass


class TorqueThrustAgent(Agent, abc.ABC):
    # TODO add action space
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_torque = .3
        self._max_thrust = 10.

        self._angular_vel_threshold = .2
        self._linear_vel_threshold = .03

        self._torque = None
        self._thrust = None

    def step(self, time_step):
        print(str(self._body.linearVelocity) + " - " + str(self._body.angularVelocity))

        if self._torque and np.linalg.norm(self._body.angularVelocity) < self._angular_vel_threshold:
            # self._body.ApplyTorque(self._torque, wake=True)
            self._body.ApplyAngularImpulse(self._torque, wake=True)
            self._torque = None
        elif self._thrust and np.linalg.norm(self._body.angularVelocity) < self._angular_vel_threshold:
            # self._body.ApplyForceToCenter(self._body.GetWorldVector(b2Vec2(self._thrust, .0)), wake=True)
            self._body.ApplyLinearImpulse(self._body.GetWorldVector(b2Vec2(self._thrust, .0)),
                                          point=self._body.worldCenter, wake=True)
            self._thrust = None
        elif np.linalg.norm(self._body.linearVelocity) < self._linear_vel_threshold \
                and np.linalg.norm(self._body.angularVelocity) < self._angular_vel_threshold:
            # random movement
            if np.random.rand() >= .5:
                thrust = np.random.rand() * self._max_thrust
                # self._body.ApplyForceToCenter(self._body.GetWorldVector(b2Vec2(thrust, .0)), wake=True)
                self._body.ApplyLinearImpulse(self._body.GetWorldVector(b2Vec2(self._thrust, .0)),
                                              point=self._body.worldCenter, wake=True)
            else:
                torque = (np.random.rand() - .5) * self._max_torque
                # self._body.ApplyTorque(torque, wake=True)
                self._body.ApplyAngularImpulse(torque, wake=True)


class TurnBoostAgent(Agent, abc.ABC):
    """
    The TurnBoosGuppy approximates the behaviour observed with real fish, by first turning into a desired direction
    and then performing a forward boost.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._max_turn = np.pi / 10.
        self._max_boost = .05

        self.__turn = None
        self.__boost = None

    @property
    def turn(self):
        return self.__turn

    @turn.setter
    def turn(self, turn):
        self.__turn = turn

    @property
    def boost(self):
        return self.__boost

    @boost.setter
    def boost(self, boost):
        self.__boost = np.maximum(boost, .0)

    def step(self, time_step):
        if self.turn:
            t = np.minimum(np.maximum(self.turn, -self._max_turn), self._max_turn)
            self._body.angle += t
            self._body.linearVelocity = b2Vec2(.0, .0)
            self.turn -= t
        elif self.boost:
            b = np.minimum(self.boost, self._max_boost)
            self._body.ApplyLinearImpulse(self._body.GetWorldVector(b2Vec2(b, .0)),
                                          point=self._body.worldCenter, wake=True)
            self.boost -= b


class ConstantVelocityAgent(Agent, abc.ABC):
    _linear_damping = .0
    _angular_damping = .0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._velocity = .2

        self._turn = None
        self._max_turn = np.pi / 10.

    def step(self, time_step):
        if self._turn:
            t = np.minimum(np.maximum(self._turn, -self._max_turn), self._max_turn)
            self._body.angle += t
            self._turn -= t

        self.set_angular_velocity(.0)
        self.set_linear_velocity((self._velocity, .0), local=True)
