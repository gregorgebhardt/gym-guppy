import gym
import numpy as np


class NormalizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(NormalizeActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,))
        self.factors = self.env.action_space.low

    def action(self, action):
        action[0] = action[0] * 2 - 1

    def reverse_action(self, action):
        raise NotImplementedError


class FlatActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(FlatActionWrapper, self).__init__(env)
        self._original_action_space = self.action_space

        self.action_space = gym.spaces.Box(low=self.action_space.low.flatten(),
                                           high=self.action_space.high.flatten())

    def action(self, action):
        return np.reshape(action, self._original_action_space.shape)

    def reverse_action(self, action):
        raise NotImplementedError


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, num_bins_turn_rate=20, num_bins_speed=20):
        super(DiscreteActionWrapper, self).__init__(env)
        assert isinstance(self.action_space, gym.spaces.Box)
        max_turn_rate = self.action_space.high[0]
        self.turn_rate_bins = np.linspace(-max_turn_rate, max_turn_rate, num_bins_turn_rate + 1)
        max_speed = self.action_space.high[1]
        self.speed_bins = np.linspace(0, max_speed, num_bins_speed + 1)
        self.action_space = gym.spaces.MultiDiscrete([num_bins_turn_rate, num_bins_speed])

    def action(self, action):
        turn_rate = action[0]
        speed = action[1]
        sample_from_bins = np.random.uniform([self.turn_rate_bins[turn_rate], self.speed_bins[speed]],
                                             [self.turn_rate_bins[turn_rate + 1], self.speed_bins[speed + 1]]).astype(
            'float32')
        return sample_from_bins

    def reverse_action(self, action):
        raise NotImplementedError


class MovementLimitWrapper(gym.ActionWrapper):
    def __init__(self, env, turn_limit=None, speed_limit=None):
        super(MovementLimitWrapper, self).__init__(env)
        if turn_limit is not None:
            self.action_space.low[0] = - turn_limit
            self.action_space.high[0] = turn_limit
        if speed_limit is not None:
            self.action_space.high[1] = speed_limit

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)

    def reverse_action(self, action):
        raise NotImplementedError


class RandomizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        print('Randomizing Actions')
        super(RandomizeActionWrapper, self).__init__(env)
        # TODO: check env.action_space and allow for multi-agent environments
        # TODO: specify which dimensions should be randomized, not fixed names
        self.turn_mul: float = 1.
        self.turn_bias: float = .0
        self.speed_mul: float = 1.
        self.randomize_turn = False
        self.randomize_boost = True

    def reset(self):
        self.turn_mul = np.random.normal(loc=1.0, scale=0.1)
        self.turn_bias = np.random.normal(loc=0.0, scale=self.env.action_space.high[0] / 20.)
        self.speed_mul = np.random.normal(loc=1.0, scale=0.1)
        return super(RandomizeActionWrapper, self).reset()

    def action(self, action):
        # TODO: implement this in environment
        # if self.env.robot_blocked:
        #     action = np.array([0.0, 0.0], dtype=np.float32)
        #     return self.env.step(action)
        if self.randomize_turn:
            action[0] = action[0] * self.turn_mul + self.turn_bias
        if self.randomize_boost:
            action[1] = action[1] * self.speed_mul
        return action

    def reverse_action(self, action):
        raise NotImplementedError


class Local2GlobalWrapper(gym.ActionWrapper):
    def action(self, action):
        global_actions = []
        for r, a in zip(self.robots, action):
            global_actions.append(r.get_global_point(action))

        return np.asarray(global_actions)

    def reverse_action(self, action):
        local_actions = []
        for r, a in zip(self.robots, action):
            local_actions.append(r.get_local_point(action))

        return np.asarray(local_actions)
