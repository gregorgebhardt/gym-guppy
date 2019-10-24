import abc
from functools import update_wrapper

import numpy as np
from typing import Callable, Any, Tuple, Union

_RewardFunctionT = Callable[[np.ndarray, np.ndarray, np.ndarray], float]
_RewardFunctionWithArgsT = Callable[[np.ndarray, np.ndarray, np.ndarray, Any], float]
_RewardWrapperT = Callable[[float, Any], float]
_InputWrapperT = Callable[[np.ndarray, np.ndarray, np.ndarray, Any], Tuple[np.ndarray, np.ndarray, np.ndarray]]

reward_registry = {}

# TODO: move registration of Reward-Function to init of RewardFunction?


def reward_function(f: _RewardFunctionT):
    _reward_function = RewardFunction(f)
    reward_registry[f.__name__] = _reward_function
    return _reward_function


def reward_function_with_args(f: _RewardFunctionWithArgsT):
    def _reward_function_factory(*f_args, **f_kwargs):
        return RewardFunction(f, function_args=f_args, function_kwargs=f_kwargs)

    reward_registry[f.__name__] = _reward_function_factory
    return _reward_function_factory


def reward_wrapper(f: _RewardWrapperT):
    def _reward_wrapper_factory(a: RewardFunctionBase, *w_args, **w_kwargs):
        return RewardWrapper(f, a, wrapper_args=w_args, wrapper_kwargs=w_kwargs)

    reward_registry[f.__name__] = _reward_wrapper_factory
    return _reward_wrapper_factory


def input_wrapper(f: _InputWrapperT):
    def _input_wrapper_factory(a: RewardFunctionBase, *w_args, **w_kwargs):
        return InputWrapper(f, a, wrapper_args=w_args, wrapper_kwargs=w_kwargs)

    reward_registry[f.__name__] = _input_wrapper_factory
    return _input_wrapper_factory


class RewardFunctionBase(abc.ABC):
    precedence = 0

    def __add__(self, other):
        if not isinstance(other, RewardFunctionBase):
            other = RewardConst(other)
        return RewardSum(self, other)

    def __radd__(self, other):
        if not isinstance(other, RewardFunctionBase):
            other = RewardConst(other)
        return RewardSum(other, self)

    def __sub__(self, other):
        if not isinstance(other, RewardFunctionBase):
            other = RewardConst(other)
        return RewardSub(self, other)

    def __rsub__(self, other):
        if not isinstance(other, RewardFunctionBase):
            other = RewardConst(other)
        return RewardSub(other, self)

    def __mul__(self, other):
        if not isinstance(other, RewardFunctionBase):
            other = RewardConst(other)
        return RewardMul(self, other)

    def __rmul__(self, other):
        if not isinstance(other, RewardFunctionBase):
            other = RewardConst(other)
        return RewardMul(other, self)

    def __truediv__(self, other):
        if not isinstance(other, RewardFunctionBase):
            other = RewardConst(other)
        return RewardDiv(self, other)

    def __rtruediv__(self, other):
        if not isinstance(other, RewardFunctionBase):
            other = RewardConst(other)
        return RewardDiv(other, self)

    def __pow__(self, power):
        if not isinstance(power, RewardFunctionBase):
            power = RewardConst(power)
        return RewardPow(self, power)

    def __rpow__(self, other):
        if not isinstance(other, RewardFunctionBase):
            other = RewardConst(other)
        return RewardPow(other, self)

    def __abs__(self):
        return RewardWrapper(abs, self)

    @abc.abstractmethod
    def __call__(self, state, action, next_state):
        raise NotImplementedError

    def __str__(self):
        return "RewardFunction: " + self.__repr__()

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError


class RewardFunction(RewardFunctionBase):
    def __init__(self, f: _RewardFunctionWithArgsT, function_args=(), function_kwargs=None):
        self._reward_function = f
        self._function_args = function_args
        self._function_kwargs = function_kwargs if function_kwargs else {}
        update_wrapper(self, f)

    def __call__(self, state, action, next_state):
        return self._reward_function(state, action, next_state, *self._function_args, **self._function_kwargs)

    def __repr__(self):
        if self._function_args or self._function_kwargs:
            args_list = tuple(map(str, self._function_args)) + tuple(map(lambda t: '{}={}'.format(*t),
                                                                         self._function_kwargs.items()))
            args_str = ', '.join(args_list)

            return self._reward_function.__name__ + f"({args_str})"
        return self._reward_function.__name__


class WrapperBase(RewardFunctionBase, abc.ABC):
    def __init__(self, f: Union[_RewardWrapperT, _InputWrapperT], a: RewardFunctionBase, *wrapper_args,
                 **wrapper_kwargs):
        self._a = a
        self._f = f
        self._wrapper_args = wrapper_args
        self._wrapper_kwargs = wrapper_kwargs if wrapper_kwargs else {}
        update_wrapper(self, f)

    def __repr__(self):
        args_str = ''
        if self._wrapper_args or self._wrapper_kwargs:
            args_list = tuple(map(str, self._wrapper_args)) + tuple(map(lambda t: '{}={}'.format(*t),
                                                                        self._wrapper_kwargs.items()))
            args_str = ', ' + ', '.join(args_list)

        return f"{self._f.__name__}({self._a.__repr__()}{args_str})"


class RewardWrapper(WrapperBase):
    def __init__(self, f: _RewardWrapperT, a: RewardFunctionBase, *wrapper_args, **wrapper_kwargs):
        super().__init__(f, a, wrapper_args, wrapper_kwargs)

    def __call__(self, state, action, next_state):
        return self._f(self._a(state, action, next_state), *self._wrapper_args, **self._wrapper_kwargs)


class InputWrapper(WrapperBase):
    def __init__(self, f: _InputWrapperT, a: RewardFunctionBase, *wrapper_args, **wrapper_kwargs):
        super().__init__(f, a, wrapper_args, wrapper_kwargs)

    def __call__(self, state, action, next_state):
        return self._a(*self._f(state, action, next_state, *self._wrapper_args, **self._wrapper_kwargs))


class RewardConst(RewardFunctionBase):
    def __init__(self, a):
        self._a = a

    def __call__(self, state, action, next_state):
        return self._a

    def __repr__(self):
        return self._a.__str__()


class RewardBinaryOp(RewardFunctionBase, abc.ABC):
    op = None

    def __init__(self, a: RewardFunctionBase, b: RewardFunctionBase):
        self._a = a
        self._b = b

    def _parentheses(self, other: RewardFunctionBase):
        if other.precedence > self.precedence:
            return f'({other.__repr__()})'
        return other.__repr__()

    def __repr__(self):
        return self._parentheses(self._a) + " " + self.op + " " + self._parentheses(self._b)


class RewardSum(RewardBinaryOp):
    precedence = 7
    op = "+"

    def __call__(self, state, action, next_state):
        return self._a(state, action, next_state) + self._b(state, action, next_state)


class RewardSub(RewardBinaryOp):
    precedence = 7
    op = "-"

    def __call__(self, state, action, next_state):
        return self._a(state, action, next_state) - self._b(state, action, next_state)


class RewardMul(RewardBinaryOp):
    precedence = 6
    op = "*"

    def __call__(self, state, action, next_state):
        return self._a(state, action, next_state) * self._b(state, action, next_state)


class RewardDiv(RewardBinaryOp):
    precedence = 6
    op = "/"

    def __call__(self, state, action, next_state):
        return self._a(state, action, next_state) / self._b(state, action, next_state)


class RewardPow(RewardBinaryOp):
    precedence = 4
    op = "**"

    def __call__(self, state, action, next_state):
        return self._a(state, action, next_state) ** self._b(state, action, next_state)
