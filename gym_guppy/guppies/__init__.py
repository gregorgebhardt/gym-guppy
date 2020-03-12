from ._base_agents import Agent, TurnBoostAgent, TorqueThrustAgent, ConstantVelocityAgent, TurnSpeedAgent, VelocityControlledAgent
from ._guppy import Guppy
from ._robot import Robot, TurnBoostRobot, GoToRobot, GlobalTargetRobot, PolarCoordinateTargetRobot
from ._couzin_guppies import ClassicCouzinGuppy, BoostCouzinGuppy, AdaptiveCouzinGuppy, BiasedAdaptiveCouzinGuppy
from ._adaptive_agent import AdaptiveAgent
from ._randomized_guppies import RandomizedCouzinGuppy

__all__ = [
    'ClassicCouzinGuppy',
    'BoostCouzinGuppy',
    'AdaptiveCouzinGuppy',
    'BiasedAdaptiveCouzinGuppy',
    'RandomizedCouzinGuppy'
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
