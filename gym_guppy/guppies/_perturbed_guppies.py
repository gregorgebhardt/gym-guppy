import numpy as np
from ._couzin_guppies import AdaptiveCouzinGuppy


class PerturbedAdaptiveCouzinGuppy(AdaptiveCouzinGuppy):
    """
    randomized guppy simulation
    """
    def __init__(self, *,
                 initial_zone_factor_std=0.1,
                 zone_radius_std=0.1,
                 adaptive_zone_grow_factor_noise=0.1,
                 adaptive_zone_shrink_factor_noise=0.01,
                 zoo_factor_noise=0.2,
                 **kwargs):
        super(PerturbedAdaptiveCouzinGuppy, self).__init__(**kwargs)

        # The following overrides what was defined by calling super().__init__(...)
        initial_zone_factor = min(1., self._initial_zone_factor + np.random.randn() * initial_zone_factor_std)
        self._adaptive_zone_factors = np.array([initial_zone_factor] * len(self._unknown_agents))
        self._zone_radius = max(0.1, self._zone_radius_mean + np.random.randn() * zone_radius_std)
        self._adaptive_zone_grow_factor = self._adaptive_zone_grow_factor + np.abs(
            np.random.randn()) * adaptive_zone_grow_factor_noise
        self._adaptive_zone_shrink_factor = self._adaptive_zone_shrink_factor - np.abs(
            np.random.randn()) * adaptive_zone_shrink_factor_noise

        self._zoo_factor = self._zoo_factor + np.abs(np.random.randn()) * zoo_factor_noise

        self._update_zones()
