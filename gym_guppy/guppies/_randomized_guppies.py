import numpy as np

from gym_guppy.guppies import BiasedAdaptiveCouzinGuppy


class RandomizedCouzinGuppy(BiasedAdaptiveCouzinGuppy):
    def __init__(self,
                 *,
                 initial_zone_factor_std=0,
                 zone_radius_std=0,
                 adaptive_zone_grow_factor_noise=0,
                 adaptive_zone_shrink_factor_noise=0,
                 zoo_factor_noise=0,
                 bias_gain_std=0,
                 attraction_points=None,
                 repulsion_points=None,
                 rng=None, # random number generator that can be used to fix the randomization
                 **kwargs):
        super().__init__(attraction_points=attraction_points,
                         repulsion_points=repulsion_points,
                         **kwargs)

        rng = np.random.RandomState() if rng is None else rng
        # The following overrides what was defined by calling super().__init__(...)
        initial_zone_factor = min(0.99, self._initial_zone_factor + rng.randn() * initial_zone_factor_std)
        self._adaptive_zone_factors = np.array([initial_zone_factor] * len(self._unknown_agents))
        self._zone_radius = max(0.1, self._zone_radius_mean + rng.randn() * zone_radius_std)
        self._adaptive_zone_grow_factor = self._adaptive_zone_grow_factor + np.abs(
            rng.randn()) * adaptive_zone_grow_factor_noise
        self._adaptive_zone_shrink_factor = self._adaptive_zone_shrink_factor - np.abs(
            rng.randn()) * adaptive_zone_shrink_factor_noise

        self.bias_gain = self.bias_gain + np.abs(rng.randn()) * bias_gain_std

        self._zoo_factor = 1.0 + np.abs(rng.randn()) * zoo_factor_noise

    # overrides parent method
    def _update_zones(self):
        zor = self._zone_radius * self._adaptive_zone_factors
        zoo = (self._zone_radius - zor) * self._adaptive_zone_factors * self._zoo_factor
        zoa = self._zone_radius - zoo - zor

        self._zone_cache = zor, zoo, zoa

    @property
    def dr_parameter_list(self):
        return (self._adaptive_zone_factors,
                self._zone_radius,
                self._adaptive_zone_grow_factor,
                self._adaptive_zone_shrink_factor,
                self.bias_gain,
                self._zoo_factor
                )
