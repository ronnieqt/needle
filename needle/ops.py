"""Operator implementations."""

# INFO: about dimensionality
# The size of out_grad will always be the size of the output of the operation.
# The sizes of the Tensor objects returned by gradient() have to always be
#     the same as the original inputs to the operator.

# %% import libs

from numbers import Number
from itertools import zip_longest
from typing import Optional, List, Union
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
from .backend_selection import BACKEND
import numpy

from .backend_selection import array_api, NDArray

# %% MakeTensorTuple

class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)

# %% TupleGetItem

class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)

# %% FusedAddScalars

class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)

# %% EWiseAdd

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        # input type Tensor (rather than NDArray) allows us to connect our
        #     gradient computations to the original foward-pass computational graph
        # out_grad: the output adjoint
        #           (adjoint of the particular Value node this op is attached to)
        # node    : the node corresponds to the current output value
        #           (the Value node this op is atteched to)
        # this function returns the partial adjoints of each input
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)  # create a tensor node and add it to the computational graph

# %% AddScalar

class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)

# %% EWiseMul

class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)

# %% MulScalar

class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)

# %% PowerScalar

class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** numpy.float32(self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        x, c = node.inputs[0], self.scalar
        return (out_grad * c * power_scalar(x, c-1),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)

# %% EWiseDiv

class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return (out_grad / rhs, out_grad * (-lhs)/power_scalar(rhs, 2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)

# %% DivScalar

class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a / numpy.float32(self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar, )
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)

# %% Transpose: reverses the order of two axes (axis1, axis2), defaults to the last two axes

class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # TODO: consider the case when a.ndim < 2
        axes = (-2, -1) if self.axes is None else self.axes
        return array_api.swapaxes(a, *axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)

# %% Reshape

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)

# %% Broadcast

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        res = array_api.broadcast_to(a, self.shape)
        return res.compact() if BACKEND == "nd" else res

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        out_shape = self.shape
        axes_to_sum_over = tuple(
            -idx for idx, (i, o) in enumerate(
                zip_longest(in_shape[::-1], out_shape[::-1], fillvalue=-1)
                , start=1
            ) if i != o
        )
        return reshape(summation(out_grad, axes=axes_to_sum_over), in_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

# %% Summation: sum of array elements over given axes

class Summation(TensorOp):
    def __init__(self, axes: Union[None, int, tuple] = None):
        self.axes = (axes,) if isinstance(axes, int) else axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        out_shape = tuple(
            1 if (self.axes is None) or (i in self.axes) else n
            for i, n in enumerate(in_shape)
        )
        return broadcast_to(reshape(out_grad, out_shape), in_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)

# %% MatMul

class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_ndim = len(lhs.shape)
        rhs_ndim = len(rhs.shape)
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)
        if lhs_ndim > rhs_ndim:
            rhs_grad = summation(rhs_grad, axes=tuple(range(lhs_ndim-rhs_ndim)))
        elif lhs_ndim < rhs_ndim:
            lhs_grad = summation(lhs_grad, axes=tuple(range(rhs_ndim-lhs_ndim)))
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)

# %% Negate

class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad, )
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)

# %% Log

class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)

# %% Exp

class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)

# %% ReLU

class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return (a > 0.0) * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        return out_grad * Tensor(a > 0.0, dtype=a.dtype)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

# %% LogSumExp

class LogSumExp(TensorOp):
    def __init__(self, axes: Union[None, int, tuple] = None):
        self.axes = (axes,) if isinstance(axes, int) else axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, self.axes, keepdims=True)
        if BACKEND == "nd":
            return array_api.log(
                array_api.sum(array_api.exp(Z - max_Z.broadcast_to(Z.shape)), self.axes)
            ) + array_api.squeeze(max_Z)
        else:
            return array_api.log(
                array_api.sum(array_api.exp(Z - max_Z), self.axes)
            ) + array_api.squeeze(max_Z)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X = node.inputs[0]
        # TODO: whether there are better solutions?
        c = Tensor.make_const(array_api.max(X.realize_cached_data(), self.axes, keepdims=True))
        if BACKEND == "nd":
            exp_X_stable = exp(X - c.broadcast_to(X.shape))
        else:
            exp_X_stable = exp(X - c)
        in_shape = X.shape
        out_shape = tuple(
            1 if (self.axes is None) or (i in self.axes) else n
            for i, n in enumerate(in_shape)
        )
        return out_grad.reshape(out_shape).broadcast_to(in_shape) \
               * exp_X_stable \
               / exp_X_stable.sum(self.axes).reshape(out_shape).broadcast_to(in_shape)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

# %% Tanh

class Tanh(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - tanh(node.inputs[0])**2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)

# %% Stack

class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        assert len(args) > 0, "Stack at least 1 array"
        shape, device = args[0].shape, args[0].device
        # construct an array buffer to store the stacked array
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))
        res = array_api.empty(new_shape, device=device)
        # fill in values
        for n, array in enumerate(args):
            assert array.shape == shape, "All stacked arrays must have the same shape"
            idxs = tuple(
                slice(0, s) if (i != self.axis) else slice(n, n+1)
                for i, s in enumerate(new_shape)
            )
            res[idxs] = array
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))

# %% Split

class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A: NDArray):
        ### BEGIN YOUR SOLUTION
        shape = A.shape
        new_shape = shape[:self.axis] + shape[self.axis+1:]
        res = tuple(
            A[tuple(
                slice(0,s) if i != self.axis else slice(n,n+1)
                for i, s in enumerate(shape)
            )].compact().reshape(new_shape)
            for n in range(shape[self.axis])
        )
        return tuple(res)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # out_grad    : same size as the output of this op
        # return value: same size as the input of this op
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)

# %% Flip

class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)

# %% Dilate and UnDilate

class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        new_shape = tuple(
            s + (s*self.dilation if (i in self.axes) else 0)
            for i, s in enumerate(a.shape)
        )
        res = array_api.full(new_shape, fill_value=0, dtype=a.dtype, device=a.device)
        idxs = tuple(
            slice(0, s, self.dilation+1 if (i in self.axes) else 1)
            for i, s in enumerate(new_shape)
        )
        res[idxs] = a
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        idxs = tuple(
            slice(0, s, self.dilation+1 if (i in self.axes) else 1)
            for i, s in enumerate(a.shape)
        )
        return a[idxs].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # out_grad    : same size as the output of this op
        # return value: same size as the input of this op
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)

# %% Conv

class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
