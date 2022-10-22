"""Optimization module"""

from typing import Dict, Iterable

import needle as ndl

# %% Optimizer Base Class

class Optimizer:
    def __init__(self, params: Iterable[ndl.nn.Parameter]):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None

# %% Stochastic Gradient Descent (SGD) Optimizer

class SGD(Optimizer):

    u: Dict[ndl.nn.Parameter, ndl.Tensor]

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            # calculate gradients
            param_grad = param.grad.data
            if self.weight_decay != 0:
                param_grad += self.weight_decay * param.data
            # update momentum terms
            if param not in self.u:
                self.u[param] = (1-self.momentum) * param_grad
            else:
                self.u[param] = self.momentum * self.u[param].data + (1-self.momentum) * param_grad
            param.data -= self.lr * self.u[param]
        ### END YOUR SOLUTION

# Reference: https://discuss.pytorch.org/t/how-does-sgd-weight-decay-work/33105

# %%  Adaptive Moment Estimation (Adam) Optimizer

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
