import os

from gym_guppy import TurnBoostRobot
from gym_guppy.guppies import BoostCouzinGuppy, AdaptiveCouzinGuppy, BiasedAdaptiveCouzinGuppy
from gym_guppy.guppies import AdaptiveAgent
from gym_guppy.wrappers import LocalObservationsWrapper

# os.environ['NUMBA_DISABLE_JIT'] = "1"

import numpy as np
from gym_guppy.envs import GuppyEnv

h5file = '/Users/gregor/Downloads/data/training/eyj_2_lstm_100_zero_30_live_female_female_20190913T141403U264471.hdf5'


class TestEnv(GuppyEnv):
    def _reset(self):
        self._steps_per_action = 4
        adaptive_agent = AdaptiveAgent(world=self.world, world_bounds=self.world_bounds)
        self._add_robot(adaptive_agent)
        # self._add_robot(TurnBoostRobot(world=self.world, world_bounds=self.world_bounds,
        #                                position=np.array([-0.4980772, -0.30953531]), orientation=1.5089165))

        num_guppies = 5
        # random initialization
        positions = np.random.normal(loc=.0, scale=.1, size=(num_guppies, 2))
        orientations = np.random.rand(num_guppies) * 2 * np.pi - np.pi

        attraction_points = np.asarray(self.world_bounds * 2)
        attraction_points[2:4, 0] *= -1
        repulsion_points = np.array([[.0, .0]])

        for p, o in zip(positions, orientations):
            # self._add_guppy(AdaptiveCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
            #                                     position=p, orientation=o, unknown_agents=[adaptive_agent]))
            # self._add_guppy(BiasedAdaptiveCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
            #                                           position=p, orientation=o,
            #                                           # attraction_points=[[.0, .0]],
            #                                           repulsion_points=[[.0, .0]]
            #                                           ))
            # self._add_guppy(BoostCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
            #                                  position=p, orientation=o))
            self._add_guppy(BiasedAdaptiveCouzinGuppy(world=self.world, world_bounds=self.world_bounds,
                                                      position=p, orientation=o,
                                                      attraction_points=attraction_points,
                                                      repulsion_points=repulsion_points,
                                                      bias_gain=.7))
            # self._add_guppy(MXNetGuppy(world=self.world, world_bounds=self.world_bounds,
            #                            position=p, orientation=o,
            #                            hdf_file=h5file))

    # def _draw_on_table(self, screen):
    #     for g in self.guppies:
    #         if isinstance(g, AdaptiveCouzinGuppy):
    #             zors, zoos, zoas = g.adaptive_couzin_zones()
    #
    #             width = .002
    #             for zor, zoo, zoa in zip(zors, zoos, zoas):
    #                 screen.draw_circle(g.get_position(), zor + zoo + zoa, color=(0, 100, 0), filled=False, width=width)
    #                 if zoo + zor > width:
    #                     screen.draw_circle(g.get_position(), zor + zoo, color=(50, 100, 100), filled=False, width=width)
    #                 if zor > width:
    #                     screen.draw_circle(g.get_position(), zor, color=(100, 0, 0), filled=False, width=width)


if __name__ == '__main__':
    # env = LocalObservationsWrapper(TestEnv())
    env = TestEnv()
    env.reset()
    # env.video_path = 'video_out'

    for t in range(2000):
        env.render(mode='human')

        # state_t, reward_t, done, info = env.step(np.array([1.366212, 0.859359]))
        state_t, reward_t, done, info = env.step(.0)
