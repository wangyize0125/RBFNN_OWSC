# -*- coding: utf-8 -*-
# Author: Yize Wang
# Time: 2023/3/27 15:20
# File: RBFNN_OWSC.py
# Email: wangyize@hust.edu.cn

import torch
import torch.nn as nn
import numpy as np


class TrainRBFNN(nn.Module):
    """
    RBFNN with one-dimensional output
    """
    def __init__(self, centers, sigma, w):
        super(TrainRBFNN, self).__init__()

        self.centers = nn.Parameter(torch.from_numpy(centers), requires_grad=True)
        self.sigma = nn.Parameter(torch.from_numpy(np.ones(self.centers.size()) * sigma), requires_grad=True)
        self.w = nn.Parameter(torch.from_numpy(w), requires_grad=True)
        self.num_centers = self.centers.size(0)
        self.dim_input = self.centers.size(1)

    def gaussian_kernel(self, batches):
        batch_size = batches.size(0)
        C = self.centers.view(self.num_centers, -1).repeat(batch_size, 1, 1)
        S = self.sigma.view(self.num_centers, -1).repeat(batch_size, 1, 1)
        X = batches.view(batch_size, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        result = torch.exp(-((X - C) ** 2 / 2 / S ** 2).sum(2, keepdims=False))
        return result

    def forward(self, batches):
        rbf_out = self.gaussian_kernel(batches)
        result = rbf_out.mm(self.w)
        return result

    def train_rbfnn(self, x_, y_, lr=0.01, momentum=0.9, epoch=200):
        x = torch.from_numpy(x_)
        y = torch.from_numpy(y_.reshape((-1, 1)))
        params = self.parameters()
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)

        losses = []
        for i in range(epoch):
            optimizer.zero_grad()

            y_predict = self.forward(x)
            loss = loss_fn(y_predict, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.data ** 0.5)
            print("Epoch {}: loss = {}".format(i + 1, losses[-1]))

        return losses

    def save_network(self, filename):
        np.savez(
            filename,
            centers=self.centers.detach().numpy(),
            sigma=self.sigma.detach().numpy(),
            w=self.w.detach().numpy()
        )


def normalize(x):
    bounds = [
        [1.5, 2.3],
        [0.37, 0.43],
        [0.4, 0.5],
        [1, 30],
        [0.05, 0.3]
    ]
    for i in range(len(bounds)):
        x[:, i] = ((x[:, i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0]) - 0.5) * 2
    return x


npzdata = np.load("./network.npz")
rbfnn = TrainRBFNN(**npzdata)

cases = np.array([
    [1.9, 0.4, 0.45, 15.5, 0.175],
    [1.9, 0.4, 0.45, 15.5, 0.185],
])
cases = normalize(cases)

y_predict = rbfnn.forward(torch.from_numpy(cases)).detach().numpy()
print(y_predict)
