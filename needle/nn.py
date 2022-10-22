"""The module"""

# %% Import Libs

from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

# %% Parameter and Module Class

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# %% Identity Module

class Identity(Module):
    def forward(self, X: Tensor):
        return X

# %% Linear Module

class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        ### BEGIN YOUR SOLUTION
        # initialize module parameters
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features), **factory_kwargs
        )
        self.bias = Parameter(
            init.kaiming_uniform(out_features, 1).reshape((1, out_features)), **factory_kwargs
        ) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N, out_features = X.shape[0], self.weight.shape[1]
        out = X @ self.weight
        if self.bias is not None:
            out += self.bias.broadcast_to((N, out_features))
        return out
        ### END YOUR SOLUTION

# %% ReLU Module

class ReLU(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(X)
        ### END YOUR SOLUTION

# %% Sequence Module

class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        self.modules = modules

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        H = X
        for m in self.modules:
            H = m(H)
        return H
        ### END YOUR SOLUTION

# %% Softmax Loss

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # y is a list of true labels (numbers), not one-hot encoded
        y_one_hot = init.one_hot(logits.shape[1], y)
        losses = ops.logsumexp(logits, axes=(1,)) - ops.summation(logits * y_one_hot, axes=(1,))
        return ops.summation(losses) / losses.shape[0]
        ### END YOUR SOLUTION

# %% Layer Normalization

class LayerNorm1d(Module):
    def __init__(self, dim: int, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim  # used to init parameters
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = Parameter(init.ones(dim, **factory_kwargs))
        self.bias = Parameter(init.zeros(dim, **factory_kwargs))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        '''X: a 2D tensor with batches in the 1st dim and features on the 2nd'''
        ### BEGIN YOUR SOLUTION
        n, p = X.shape
        X_mean = (X.sum(axes=(1,)) / p).reshape((n,1)).broadcast_to((n,p))
        X_var = (((X - X_mean)**2).sum((1,)) / p).reshape((n,1)).broadcast_to((n,p))
        # normalizing each row of X
        X_normalized = (X - X_mean) / (X_var + self.eps)**0.5
        return X_normalized * self.weight.broadcast_to((n,p)) \
               + self.bias.broadcast_to((n,p))
        ### END YOUR SOLUTION

# %% Flatten

class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION

# %% Batch Normalization

class BatchNorm1d(Module):
    def __init__(self, dim: int, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim  # number of features
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = Parameter(init.ones(dim, **factory_kwargs))
        self.bias = Parameter(init.zeros(dim, **factory_kwargs))
        self.running_mean = Parameter(init.zeros(dim, **factory_kwargs))
        self.running_var = Parameter(init.ones(dim, **factory_kwargs))
        ### END YOUR SOLUTION


    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n, p = X.shape
        if self.training:
            X_mean = X.sum(axes=(0,)) / n
            X_var = ((X - X_mean.broadcast_to((n,p)))**2).sum(axes=(0,)) / n
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * X_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * X_var
            X_normalized = (X - X_mean.broadcast_to((n,p))) \
                           / (X_var.broadcast_to((n,p)) + self.eps)**0.5
            return X_normalized * self.weight.broadcast_to((n,p)) \
                   + self.bias.broadcast_to((n,p))
        else:  # model.eval()
            return (X - self.running_mean.broadcast_to((n,p))) \
                   / (self.running_var.broadcast_to((n,p)) + self.eps)**0.5
        ### END YOUR SOLUTION

# %% Dropout

class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p  # zeros some of the elements of the input tensor with probability p

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mask = init.randb(*X.shape, p=1-self.p)/(1-self.p) if self.training else init.ones(*X.shape)
        return X * mask
        ### END YOUR SOLUTION

# %% Residual

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(X) + X
        ### END YOUR SOLUTION
