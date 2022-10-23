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
            if param.grad is None: continue
            # calculate gradient
            param_grad = param.grad.data
            if self.weight_decay != 0:
                param_grad += self.weight_decay * param.data
            # update momentum term
            if param not in self.u:
                self.u[param] = (1-self.momentum) * param_grad
            else:
                self.u[param] = self.momentum * self.u[param].data + (1-self.momentum) * param_grad
            # update weights
            param.data -= self.lr * self.u[param].data
        ### END YOUR SOLUTION

# Reference: https://discuss.pytorch.org/t/how-does-sgd-weight-decay-work/33105

# %%  Adaptive Moment Estimation (Adam) Optimizer

class Adam(Optimizer):

    m: Dict[ndl.nn.Parameter, ndl.Tensor]
    v: Dict[ndl.nn.Parameter, ndl.Tensor]

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
        self.t += 1
        for param in self.params:
            if param.grad is None: continue
            # calculate gradient
            param_grad = param.grad.data
            if self.weight_decay != 0:
                param_grad += self.weight_decay * param.data
            # update running average of gradient
            if param not in self.m:
                self.m[param] = (1 - self.beta1) * param_grad
            else:
                self.m[param] = self.beta1 * self.m[param].data + (1 - self.beta1) * param_grad
            # update running average of square of gradient
            if param not in self.v:
                self.v[param] = (1 - self.beta2) * param_grad**2
            else:
                self.v[param] = self.beta2 * self.v[param].data + (1 - self.beta2) * param_grad**2
            # bias correction
            m_data = self.m[param].data / (1 - self.beta1**self.t)
            v_data = self.v[param].data / (1 - self.beta2**self.t)
            # update weights
            param.data -= self.lr * m_data / (v_data**0.5 + self.eps)
        ### END YOUR SOLUTION
