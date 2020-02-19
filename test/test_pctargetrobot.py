from gym_guppy import VariableStepGuppyEnv, PolarCoordinateTargetRobot


class TestEnv(VariableStepGuppyEnv):
    def _reset(self):
        controller_params = {
            'ori_ctrl_params': {
                'p':     1.,
                'i':     0.,
                'd':     0.,
                'speed': .2,
                'slope': 1.
            },
            'fwd_ctrl_params': {
                'p':              1.,
                'i':              0.,
                'd':              0.,
                'speed':          .2,
                'slope':          100.,
                'ori_gate_slope': 1.
            }
        }

        self._add_robot(PolarCoordinateTargetRobot(world=self.world,
                                                   world_bounds=self.world_bounds,
                                                   position=(0, 0),
                                                   orientation=0,
                                                   ctrl_params=controller_params))


if __name__ == '__main__':
    # env = LocalObservationsWrapper(TestEnv())
    env = TestEnv()
    env.reset()
    # env.video_path = 'video_out'

    for t in range(2000):
        env.render(mode='human')

        # state_t, reward_t, done, info = env.step(np.array([1.366212, 0.859359]))
        state_t, reward_t, done, info = env.step(env.action_space.sample())
