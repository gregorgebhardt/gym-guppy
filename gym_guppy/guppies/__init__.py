from ._base_agents import Agent, TurnBoostAgent, TorqueThrustAgent, ConstantVelocityAgent, TurnSpeedAgent, \
    VelocityControlledAgent
from ._guppy import Guppy
from ._robot import Robot, TurnBoostRobot, GoToRobot, GlobalTargetRobot, PolarCoordinateTargetRobot
from ._couzin_guppies import BaseCouzinGuppy, ClassicCouzinGuppy, BoostCouzinGuppy, AdaptiveCouzinGuppy, \
    BiasedAdaptiveCouzinGuppy
from ._adaptive_agent import AdaptiveAgent
from ._randomized_guppies import RandomizedCouzinGuppy

__all__ = [
    'Agent',
    'TurnBoostAgent',
    'ConstantVelocityAgent',
    'TurnSpeedAgent',
    'VelocityControlledAgent',
    'Guppy',
    'BaseCouzinGuppy',
    'ClassicCouzinGuppy',
    'BoostCouzinGuppy',
    'AdaptiveCouzinGuppy',
    'BiasedAdaptiveCouzinGuppy',
    'RandomizedCouzinGuppy',
    'Robot',
    'TurnBoostRobot',
    'GlobalTargetRobot',
    'PolarCoordinateTargetRobot'
]

import importlib.util
spec = importlib.util.find_spec('mxnet')
if spec is None:
    class MXNetGuppy(Guppy):
        def __init__(self, *, hdf_file, **kwargs):
            raise ModuleNotFoundError("You need to install `mxnet` to use MXNetGuppy, you can use `pip install mxnet`")

else:
    from ._mxnet_guppies import MXNetGuppy
    __all__.append('MXNetGuppy')
