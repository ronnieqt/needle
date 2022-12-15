#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import libs

# import sys
# sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

# %% ResNet

class ConvBN(ndl.nn.Module):

    def __init__(self, a: int, b: int, k: int, s: int, device=None, dtype="float32"):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.conv = nn.Conv(a, b, kernel_size=k, stride=s, **self.factory_kwargs)
        self.batch_norm = nn.BatchNorm2d(b, **self.factory_kwargs)
        self.relu = nn.ReLU()

    def forward(self, X: ndl.Tensor) -> ndl.Tensor:
        res = self.conv(X)
        res = self.batch_norm(res)
        res = self.relu(res)
        return res


class ResNet9(ndl.nn.Module):

    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.net = nn.Sequential(
            ConvBN(3,  16, 7, 4, **self.factory_kwargs),
            ConvBN(16, 32, 3, 2, **self.factory_kwargs),
            nn.Residual(nn.Sequential(
                ConvBN(32, 32, 3, 1, **self.factory_kwargs),
                ConvBN(32, 32, 3, 1, **self.factory_kwargs)
            )),
            ConvBN(32, 64,  3, 2, **self.factory_kwargs),
            ConvBN(64, 128, 3, 2, **self.factory_kwargs),
            nn.Residual(nn.Sequential(
                ConvBN(128, 128, 3, 1, **self.factory_kwargs),
                ConvBN(128, 128, 3, 1, **self.factory_kwargs)
            )),
            nn.Flatten(),
            nn.Linear(128, 128, **self.factory_kwargs),
            nn.ReLU(),
            nn.Linear(128, 10, **self.factory_kwargs)
        )
        ### END YOUR SOLUTION

    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return self.net(X)
        ### END YOUR SOLUTION

# %% Language Model

class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        assert seq_model in ("rnn", "lstm")
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(output_size, embedding_size, **self.factory_kwargs)
        seq = nn.RNN if (seq_model == "rnn") else nn.LSTM
        self.seq_model = seq(embedding_size, hidden_size, num_layers, **self.factory_kwargs)
        self.linear = nn.Linear(hidden_size, output_size, **self.factory_kwargs)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        X of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        embd = self.embedding(X)
        states, h = self.seq_model(embd, h)
        hidden_size = states.shape[-1]
        out = self.linear(states.reshape((-1,hidden_size)))
        return out, h
        ### END YOUR SOLUTION

# %% main

if __name__ == "__main__":
    # === sim data
    # x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    x_cpu = ndl.Tensor(ndl.init.randn(20, 3, 32, 32))  # NCHW
    x_gpu = ndl.Tensor(ndl.init.randn(20, 3, 32, 32), device=ndl.cuda())
    # === real data
    # cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    # train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu())
    # print(cifar10_train_dataset[1][0].shape)
    # === ConvBN
    # model_cpu = ConvBN(3, 16, 7, 4)
    # print(model_cpu(x_cpu).shape)
    # model_gpu = ConvBN(3, 16, 7, 4, device=ndl.cuda())
    # print(model_gpu(x_gpu).shape)
    # === ResNet9
    model_cpu = ResNet9()
    print(model_cpu(x_cpu).shape)
    model_gpu = ResNet9(device=ndl.cuda())
    print(model_gpu(x_gpu).shape)
    # === Language Model
    data_cpu = ndl.Tensor(np.random.randint(0, 500, (20, 256)))
    model_cpu = LanguageModel(50, 500, 100, 2, seq_model="rnn")
    res = model_cpu(data_cpu)
    print(res[0].shape, res[1].shape)
    print("Parameters:\n", len(model_cpu.parameters()))
    data_gpu = ndl.Tensor(np.random.randint(0, 500, (20, 256)), device=ndl.cuda())
    model_gpu = LanguageModel(50, 500, 100, 2, seq_model="rnn", device=ndl.cuda())
    res = model_gpu(data_gpu)
    print(res[0].shape, res[1].shape)
    input("end")
