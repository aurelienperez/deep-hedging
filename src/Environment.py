#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:49:03 2025

@author: aurelienperez
"""

import numpy as np
import torch.optim as optim

class Env:
    def __init__(self, market_env, training_env):
        self.market = market_env
        self.training = training_env

class MarketEnv:
    def __init__(self, S0, T, N, r, sigma, payoff_func, tc, scaling_factor):
        self.S0 = S0            
        if np.isscalar(S0):
            self.d = 1
        else:
            self.d = len(S0)
        
        self.T = T               
        self.N = N                
        self.r = r              
        self.sigma = sigma
        self.tc = tc #transaction costs
        self.payoff_func = payoff_func  
        
        self.scaling_factor = scaling_factor # not really a market parameter but nevermind
        
class TrainingEnv:
    def __init__(self, model,batch_size, lr_p0, lr_rest, device):
        self.model = model
        self.batch_size = batch_size
        self.lr_p0 = lr_p0
        self.lr_rest = lr_rest
        self.device = device
        self.optimizer_p0 = optim.Adam([self.model.p0], lr=lr_p0)
        params_rest = [param for name, param in self.model.named_parameters() if name != 'p0']
        self.optimizer_rest = optim.Adam(params_rest, lr=lr_rest)
          

