import os

from gym_guppy.guppies import BoostCouzinGuppy, AdaptiveCouzinGuppy, BiasedAdaptiveCouzinGuppy
from gym_guppy.guppies import AdaptiveAgent

# os.environ['NUMBA_DISABLE_JIT'] = "1"

import numpy as np
from gym_guppy.envs import GuppyEnv


class TestEnv(GuppyEnv):
    def _configure_environment(self):
        # adaptive_agent = AdaptiveAgent(world=self.world, world_bounds=self.world_bounds)
        # self._add_guppy(adaptive_agent)

        num_guppies = 5
        # random initialization
        positions = np.random.normal(loc=.0, scale=.05, size=(num_guppies, 2))
        orientations = np.random.rand(num_guppies) * 2 * np.pi - np.pi

        for p, o in zip(positions, orientations):
            # self._add_guppy(AdaptiveCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
            #                                     position=p, orientation=o, unknown_agents=[adaptive_agent]))
            self._add_guppy(BiasedAdaptiveCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
                                                      position=p, orientation=o,
                                                      # attraction_points=[[.0, .0]],
                                                      repulsion_points=[[.0, .0]]
                                                      ))
            # self._add_guppy(BoostCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
            #                                  position=p, orientation=o))

    def _draw_on_table(self, screen):
        for g in self.guppies:
            if isinstance(g, AdaptiveCouzinGuppy):
                zors, zoos, zoas = g.adaptive_couzin_zones()

                width = .002
                for zor, zoo, zoa in zip(zors, zoos, zoas):
                    screen.draw_circle(g.get_position(), zor + zoo + zoa, color=(0, 100, 0), filled=False, width=width)
                    if zoo + zor > width:
                        screen.draw_circle(g.get_position(), zor + zoo, color=(50, 100, 100), filled=False, width=width)
                    if zor > width:
                        screen.draw_circle(g.get_position(), zor, color=(100, 0, 0), filled=False, width=width)


if __name__ == '__main__':
    env = TestEnv()
    env.reset()
    # env.video_path = 'video_out'

    for t in range(1000):
        env.render()

        state_t, reward_t, done, info = env.step(np.array([]))
