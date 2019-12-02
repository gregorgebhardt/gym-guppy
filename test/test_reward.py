import unittest

from gym_guppy.tools.reward_function import reward_function, reward_function_with_args, reward_wrapper, reward_registry


@reward_function
def test_reward(state, action, next_state):
    return state + action + next_state


@reward_function
def test_reward2(state, action, next_state):
    return state * action * next_state


@reward_function_with_args
def test_reward_args(state, _action, _next_state, c):
    return state + c


@reward_function_with_args
def test_reward_2args(state, _action, _next_state, c, m):
    return (state + c) * m


@reward_wrapper
def test_wrapper(r):
    return 2 * r


@reward_wrapper
def clipping_wrapper(r):
    return min(max(r, 0.), 5.)


@reward_wrapper
def test_wrapper_with_arg(r, c):
    return c * r


class RewardTestCase(unittest.TestCase):
    def test_reward(self):
        self.assertEqual(test_reward(1, 2, 3), test_reward._reward_function(1, 2, 3))

    def test_addition(self):
        reward = test_reward + 2
        self.assertEqual(reward(1, 2, 3), test_reward._reward_function(1, 2, 3) + 2)
        reward = 2 + test_reward
        self.assertEqual(reward(1, 2, 3), test_reward._reward_function(1, 2, 3) + 2)
        reward = test_reward + test_reward2
        self.assertEqual(reward(1, 2, 3), test_reward._reward_function(1, 2, 3)
                         + test_reward2._reward_function(1, 2, 3))

    def test_multiplication(self):
        reward = test_reward * 2
        self.assertEqual(reward(1, 2, 3), test_reward._reward_function(1, 2, 3) * 2)
        reward = 2 * test_reward
        self.assertEqual(reward(1, 2, 3), test_reward._reward_function(1, 2, 3) * 2)
        reward = test_reward * test_reward2
        self.assertEqual(reward(1, 2, 3), test_reward._reward_function(1, 2, 3) *
                         test_reward2._reward_function(1, 2, 3))

    def test_subtraction(self):
        reward = test_reward - 2
        self.assertEqual(reward(1, 2, 3), test_reward._reward_function(1, 2, 3) - 2)
        reward = 2 - test_reward
        self.assertEqual(reward(1, 2, 3), 2 - test_reward._reward_function(1, 2, 3))
        reward = test_reward - test_reward2
        self.assertEqual(reward(1, 2, 3), test_reward._reward_function(1, 2, 3) -
                         test_reward2._reward_function(1, 2, 3))

    def test_division(self):
        reward = test_reward / 2
        self.assertEqual(reward(1, 2, 3), test_reward._reward_function(1, 2, 3) / 2)
        reward = 2 / test_reward
        self.assertEqual(reward(1, 2, 3), 2 / test_reward._reward_function(1, 2, 3))
        reward = test_reward / test_reward2
        self.assertEqual(reward(1, 2, 3), test_reward._reward_function(1, 2, 3) /
                         test_reward2._reward_function(1, 2, 3))

    def test_parentheses(self):
        reward = (test_reward + 2) * 3
        self.assertEqual(reward(1, 2, 3), (test_reward._reward_function(1, 2, 3) + 2) * 3)
        reward2 = 3 * (test_reward + 2)
        self.assertEqual(reward2(1, 2, 3), 3 * (test_reward._reward_function(1, 2, 3) + 2))
        self.assertEqual(reward2(1, 2, 3), reward(1, 2, 3))

    def test_power_with_const(self):
        reward = test_reward ** 2
        self.assertEqual(reward(1, 2, 3), test_reward._reward_function(1, 2, 3) ** 2)
        reward = 2 ** test_reward
        self.assertEqual(reward(1, 2, 3), 2 ** test_reward._reward_function(1, 2, 3))

    def test_power_with_reward(self):
        reward = test_reward ** test_reward
        self.assertEqual(reward(1, 2, 3),
                         test_reward._reward_function(1, 2, 3) ** test_reward._reward_function(1, 2, 3))

    def test_abs(self):
        reward = abs(test_reward)
        self.assertEqual(reward(-1, -2, -3), abs(test_reward._reward_function(-1, -2, -3)))


class RewardWithArgsTestCase(unittest.TestCase):
    def test_keyword_argument(self):
        reward = test_reward_args(c=2)
        self.assertEqual(reward(1, 2, 3), reward._reward_function(1, 2, 3, c=2))

    def test_positional_argument(self):
        reward = test_reward_args(2)
        self.assertEqual(reward(1, 2, 3), reward._reward_function(1, 2, 3, 2))

    def test_mixed_arguments(self):
        reward = test_reward_2args(2, m=3)
        self.assertEqual(reward(1, 2, 3), reward._reward_function(1, 2, 3, 2, m=3))
        reward = test_reward_2args(m=3, c=2)
        self.assertEqual(reward(1, 2, 3), reward._reward_function(1, 2, 3, m=3, c=2))


class RewardWrapperTestCase(unittest.TestCase):
    def test_wrapper(self):
        reward = test_wrapper(test_reward)
        self.assertEqual(reward(1, 2, 3), 2 * test_reward(1, 2, 3))

    def test_clipping_wrapper(self):
        reward = clipping_wrapper(test_reward)
        self.assertEqual(reward(1, 2, 3), 5)

    def test_wrapper_with_arg(self):
        reward = test_wrapper_with_arg(test_reward, c=2)
        self.assertEqual(reward(1, 2, 3), test_reward(1, 2, 3) * 2)
        reward2 = test_wrapper_with_arg(test_reward, 2)
        self.assertEqual(reward2(1, 2, 3), test_reward(1, 2, 3) * 2)


class EvalRewardStringTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_rewards = [
            ('test_reward', test_reward),
            ('test_reward + test_reward2', test_reward + test_reward2),
            ('test_wrapper(test_reward)', test_wrapper(test_reward)),
            ('test_reward_2args(2, m=3)', test_reward_2args(2, m=3)),
            ('(test_reward + 2) * 3', (test_reward + 2) * 3),
            ('3 * (test_reward + 2)', 3 * (test_reward + 2)),
            ('test_reward ** test_reward', test_reward ** test_reward),
            ('test_reward ** 2', test_reward ** 2),
            ('(test_reward + 2) ** 3', (test_reward + 2) ** 3),
            ('3 ** (test_reward + 2)', 3 ** (test_reward + 2)),
        ]

    def test_eval_string(self):
        for reward_str, reward in self.test_rewards:
            reward_from_str = eval(reward_str, reward_registry)
            self.assertEqual(reward(1, 2, 3), reward_from_str(1, 2, 3))

    # def test_eval_string_repr(self):
    #     """
    #     This test should check if the repr-strings equal the original strings.
    #     """
    #     for reward_str, reward in self.test_rewards:
    #         reward_from_str = eval(reward_str, reward_registry)
    #         self.assertEqual(repr(reward), repr(reward_from_str))
    #         self.assertEqual(repr(reward_from_str), reward_str)

    def test_repr_as_eval(self):
        """
        This test should check evaluation of the repr-string will give us the same results
        """
        for reward_str, reward in self.test_rewards:
            reward_from_str = eval(repr(reward), reward_registry)
            self.assertEqual(reward(1, 2, 3), reward_from_str(1, 2, 3))


if __name__ == '__main__':
    unittest.main()
