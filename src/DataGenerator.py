#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:07:27 2025

@author: aurelienperez
"""

import torch
import math

class DataGenerator:
    def __init__(self, model, env):
        self.model = model
        self.env = env


    
    def generate_batch(self, batch_size = None): 
        if batch_size is None:
            batch_size = self.env.training.batch_size
        traj = self.model.generate_trajectories(S0=self.env.market.S0, T=self.env.market.T, N=self.env.market.N,
                                                n_paths=batch_size, in_torch = True)
        traj_tensor = traj.permute(1, 0, 2)
        traj_tensor = torch.exp(-self.env.market.r * torch.arange(self.env.market.N) * 
                                self.env.market.T/self.env.market.N)[None,:,None] * traj_tensor / self.env.market.scaling_factor
        S_T = traj[-1, :, :]
        payoff = math.exp(-self.env.market.r * self.env.market.T) * self.env.market.payoff_func(S_T) / self.env.market.scaling_factor
        zj = payoff
        return traj_tensor, zj