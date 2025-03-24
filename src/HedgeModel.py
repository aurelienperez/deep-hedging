#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:04:53 2025

@author: aurelienperez
"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(NeuralNetwork, self).__init__()
        layers = [nn.Linear(input_size, layer_sizes[0]), nn.ReLU()]
        for ls_in, ls_out in zip(layer_sizes, layer_sizes[1:]):
            layers += [nn.Linear(ls_in, ls_out), nn.ReLU()]
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class HedgeModel(nn.Module):
    def __init__(self, N, d, layer_sizes, T):
        super(HedgeModel, self).__init__()
        self.p0 = nn.Parameter(torch.ones(1))
        self.delta0 = nn.Parameter(torch.ones(1, d))
        self.hedge_net = NeuralNetwork(d + 1, layer_sizes, d)
        self.N = N
        self.T = T

    def forward(self, n, x):
        if n == 0:
            batch_size = x.shape[0]
            return self.p0, self.delta0.expand(batch_size, -1)
        time_feature = torch.full((x.shape[0], 1), n * (self.T / (self.N - 1)), device=x.device)
        x_input = torch.cat([x, time_feature], dim=1)
        return self.hedge_net(x_input)
    
    def predict(self,n,x):
        x = torch.tensor([x])[None,:]
        if n == 0:
            return self.p0.item(), self.delta0.item()
        return self(n, x).detach().cpu().item()