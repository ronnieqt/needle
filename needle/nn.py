"""The module"""

# %% Import Libs

from typing import List, Callable, Any, Optional
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

# %% Softmax Loss

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # y is a list of true labels (numbers), not one-hot encoded
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
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
        self.running_mean = init.zeros(dim, **factory_kwargs)
        self.running_var = init.ones(dim, **factory_kwargs)
        ### END YOUR SOLUTION


    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n, p = X.shape
        if self.training:
            X_mean = X.sum(axes=(0,)) / n
            X_var = ((X - X_mean.reshape((1,p)).broadcast_to((n,p)))**2).sum(axes=(0,)) / n
            # shape of X_mean and X_var: (p,)
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * X_mean.data
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * X_var.data
            X_normalized = (X - X_mean.reshape((1,p)).broadcast_to((n,p))) \
                           / (X_var.reshape((1,p)).broadcast_to((n,p)) + self.eps)**0.5
        else:  # model.eval()
            X_normalized = (X - self.running_mean.reshape((1,p)).broadcast_to((n,p))) \
                           / (self.running_var.reshape((1,p)).broadcast_to((n,p)) + self.eps)**0.5
        # shape of X_normalized: (n, p)
        return X_normalized * self.weight.reshape((1,p)).broadcast_to((n,p)) \
               + self.bias.reshape((1,p)).broadcast_to((n,p))
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, X: Tensor):
        # nchw -> nhcw -> nhwc
        s = X.shape
        _x = X.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))

# %% Dropout

class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p  # zeros some of the elements of the input tensor with probability p

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mask = init.randb(*X.shape, p=1-self.p, dtype=X.dtype) / (1-self.p) \
               if self.training else init.ones(*X.shape)
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

# %% ReLU Module

class ReLU(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(X)
        ### END YOUR SOLUTION

# %% Tanh

class Tanh(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(X)
        ### END YOUR SOLUTION

# %% Sigmoid

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.sigmoid(X)
        ### END YOUR SOLUTION

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

# %% Convolution(2D)

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        factory_kwargs = {"device": device, "dtype": dtype}
        receptive_field_size = self.kernel_size * self.kernel_size
        self.weight = Parameter(init.kaiming_uniform(
            fan_in=self.in_channels*receptive_field_size,
            fan_out=self.out_channels*receptive_field_size,
            shape=(self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)
        ), **factory_kwargs)
        bound = 1.0 / (self.in_channels * receptive_field_size)**0.5
        self.bias = Parameter(init.rand(
            self.out_channels,
            low=-bound, high=bound
        ), **factory_kwargs) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X_NHWC = X.transpose((1,3)).transpose((1,2))  # X: NCHW
        # calculate the appropriate padding to ensure input and output dimensions are the same
        # (in the stride=1 case, anyways)
        padding = self.kernel_size // 2
        out = ops.conv(X_NHWC, self.weight, self.stride, padding)
        if self.bias is not None:
            out += self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(out.shape)
        # NHWC => NCHW
        return out.transpose((1,2)).transpose((1,3))
        ### END YOUR SOLUTION

# %% RNN

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        assert nonlinearity in ("tanh", "relu")
        self.f_nonlinear = ops.tanh if nonlinearity == "tanh" else ops.relu

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.factory_kwargs = {"device": device, "dtype": dtype}
        bound = (1 / hidden_size)**0.5
        self.W_ih = Parameter(init.rand(input_size,  hidden_size, low=-bound, high=bound), **self.factory_kwargs)
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound), **self.factory_kwargs)

        self.bias = bias
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound), **self.factory_kwargs)
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound), **self.factory_kwargs)
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor, h: Optional[Tensor] = None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        A = X @ self.W_ih + (h @ self.W_hh if (h is not None) else 0)
        if self.bias:
            A += self.bias_ih.reshape((1,self.hidden_size)).broadcast_to(A.shape) \
               + self.bias_hh.reshape((1,self.hidden_size)).broadcast_to(A.shape)
        return self.f_nonlinear(A)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.rnn_cells = [
            RNNCell(
                input_size if (i == 0) else hidden_size,
                hidden_size,
                bias, nonlinearity,
                device, dtype
            )
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X: Tensor, h0: Optional[Tensor] = None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        Xs = ops.split(X, axis=0)                                                      # len = seq_len
        hs = [*ops.split(h0, axis=0)] if (h0 is not None) else [None]*self.num_layers  # len = num_layers
        outs = []
        for X in Xs:
            h = X
            for k, rnn_cell in enumerate(self.rnn_cells):
                h = rnn_cell(h, hs[k])
                hs[k] = h
            outs.append(hs[-1])  # h from the last layer of the RNN
        return ops.stack(outs, axis=0), ops.stack(hs, axis=0)
        ### END YOUR SOLUTION

# %% LSTM

class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.factory_kwargs = {"device": device, "dtype": dtype}
        bound = (1 / hidden_size)**0.5
        self.W_ih = Parameter(init.rand(input_size,  4*hidden_size, low=-bound, high=bound), **self.factory_kwargs)
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-bound, high=bound), **self.factory_kwargs)

        self.bias = bias
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, low=-bound, high=bound), **self.factory_kwargs)
            self.bias_hh = Parameter(init.rand(4*hidden_size, low=-bound, high=bound), **self.factory_kwargs)
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if isinstance(h, tuple) and (h[0] is None or h[1] is None):
            h = None
        A = X @ self.W_ih + (h[0] @ self.W_hh if (h is not None) else 0)
        if self.bias:
            A += self.bias_ih.reshape((1,-1)).broadcast_to(A.shape) \
               + self.bias_hh.reshape((1,-1)).broadcast_to(A.shape)
        batch_size = X.shape[0]
        As = ops.split(A.reshape((batch_size, 4, self.hidden_size)), axis=1)
        i = ops.sigmoid(As[0])
        f = ops.sigmoid(As[1])
        g = ops.tanh(As[2])
        o = ops.sigmoid(As[3])
        c_ = (f * h[1] if h is not None else 0) + i * g
        h_ = o * ops.tanh(c_)
        return h_, c_
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.lstm_cells = [
            LSTMCell(
                input_size if (i == 0) else hidden_size,
                hidden_size,
                bias, device, dtype
            )
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0  of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        Xs = ops.split(X, axis=0)  # len = seq_len
        hcs = [[*ops.split(h[0], axis=0)], [*ops.split(h[1], axis=0)]] \
              if (h is not None) else \
              [[None]*self.num_layers, [None]*self.num_layers]
        outs = []
        for X in Xs:
            h = X
            for k, lstm_cell in enumerate(self.lstm_cells):
                h, c = lstm_cell(h, (hcs[0][k], hcs[1][k]))
                hcs[0][k], hcs[1][k] = h, c
            outs.append(hcs[0][-1])
        return (
            ops.stack(outs, axis=0),                                # (seq_len, batch_size, hidden_size)
            (ops.stack(hcs[0], axis=0), ops.stack(hcs[1], axis=0))  # (num_layers, batch_size, hidden_size)
        )
        ### END YOUR SOLUTION

# %% Embedding

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

# %% main

if __name__ == "__main__":
    # === RNN Cell
    batch_size = 128
    input_size = 10
    hidden_size = 20
    X = Tensor(np.random.randn(batch_size, input_size))
    h = Tensor(np.random.randn(batch_size, hidden_size))
    model = RNNCell(input_size, hidden_size)
    print(model(X, h).shape)
    # === LSTM Cell
    h = (
        Tensor(np.random.randn(batch_size, hidden_size)),
        Tensor(np.random.randn(batch_size, hidden_size))
    )
    model = LSTMCell(input_size, hidden_size)
    res = model(X, h)
    print(res[0].shape, res[1].shape)
    # === RNN
    seq_len = 50
    num_layers = 2
    X = Tensor(np.random.randn(seq_len, batch_size, input_size))
    h0 = Tensor(np.random.randn(num_layers, batch_size, hidden_size))
    model = RNN(input_size, hidden_size, num_layers)
    res = model(X, h0)
    print(res[0].shape, res[1].shape)
    # === LSTM
    X = Tensor(np.random.randn(seq_len, batch_size, input_size))
    h = (
        Tensor(np.random.randn(num_layers, batch_size, hidden_size)),
        Tensor(np.random.randn(num_layers, batch_size, hidden_size))
    )
    model = LSTM(input_size, hidden_size, num_layers)
    res = model(X)

