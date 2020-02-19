import numpy as np
import matplotlib.pyplot as plt

from gym_guppy.envs import VariableStepGuppyEnv
from gym_guppy.guppies import GlobalTargetRobot


controller_params = {
    'ori_ctrl_params': {
        'p':     1.,
        'speed': .2,
        # 'slope': .65
    },
    'fwd_ctrl_params': {
        'p':                   1.,
        'speed':               .2,
        'p_dist_error_factor': 100.,
        # 'slope': .3
    }
}


class TestEnv(VariableStepGuppyEnv):
    world_size = world_width, world_height = 1., 1.

    def _reset(self):
        self._add_robot(GlobalTargetRobot(world=self.world,
                                          world_bounds=self.world_bounds,
                                          position=np.array([-0.01584549, 0.02128319]),
                                          orientation=2.4684281,
                                          ctrl_params=controller_params))


if __name__ == '__main__':
    env = TestEnv()
    env.reset()
    env.render('human')

    action_list = np.array([[.0, .0],
                            [-.3, .3],
                            [.3, .3],
                            [.3, -.3],
                            [-.3, -.3],
                            [-.3, .3],
                            [.3, -.3],
                            [.0, .0]])

    # action_list = np.array([[.0, .0],
    #                         [-.1, .0]])

    steps = []

    for a in action_list:
        print(f'next target: {a}')
        # local_action = env.robot.get_local_point(a)
        observation, reward, done, info = env.step(a)
        steps.append(info['steps'])

    steps_array = np.concatenate(steps)
    plt.plot(steps_array[:, 0], steps_array[:, 1])