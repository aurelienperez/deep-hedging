#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 15:54:57 2025

@author: aurelienperez
"""

import numpy as np
from numpy.random import default_rng
import torch
import math

class Heston():
    
    def __init__(self, v0, r, kappa, theta, sigma, rho, g):
        
        self.params = dict(v0 = v0, r = r, kappa = kappa,
                           theta = theta, sigma = sigma, rho = rho)
        self.g = g
    
    def generate_trajectories(self, S0, T, N, n_paths, in_torch = False):
        if in_torch:
            if np.isscalar(S0):
                return heston_1d_torch(S0, T, N, n_paths, self.params,
                             self.g)
            elif len(S0) == 1:
                return heston_1d_torch(S0, T, N, n_paths, self.params,
                             self.g)
            else:
                d = len(S0)
                return heston_multid_torch(S0, T, N, n_paths, self.params, d,
                             self.g)
        else:
            if np.isscalar(S0):
                return heston_1d(S0, T, N, n_paths, self.params,
                             self.g)
            elif len(S0) == 1:
                return heston_1d(S0, T, N, n_paths, self.params,
                             self.g)
            else:
                d = len(S0)
                return heston_multid(S0, T, N, n_paths, self.params, d,
                             self.g)
                
        
        

def heston_1d(S0, T, N, n_paths, params,
             g = np.abs, rng = default_rng()):
    
    dt = T/(N-1)
    G = rng.standard_normal(size = (N-1 ,n_paths, 2))
    
    S = np.empty(shape = (N, n_paths)) 
    S[0] = np.array(S0).repeat(n_paths)
    
    v = np.empty(shape = (N, n_paths)) 
    v[0] = np.array(params["v0"]).repeat(n_paths)

    
    for i in range(N-1):
        v[i+1] = g(v[i] + params["kappa"] * (params["theta"] - v[i]) * dt +\
                   params["sigma"] * np.sqrt(v[i] * dt) * G[i,:,0])
                   
        S[i+1] = S[i] + params["r"] * S[i] * dt +\
            np.sqrt(v[i] * dt) * S[i] * (params["rho"] * G[i,:,0] + np.sqrt(1 - params["rho"]**2) * G[i,:,1])
            
    return S

def heston_multid(S0, T, N, n_paths, params, d,
             g = np.abs, rng = default_rng()):
    
    dt = T/(N-1)

    L = np.linalg.cholesky(params["rho"])
    
    Z = rng.standard_normal(size = (N-1 ,n_paths, 2 * d))
    G = np.einsum("xy, nmy -> nmx", L, Z) # first d -> prices, next d -> volatilities

    
    S = np.empty(shape = (N, n_paths, d)) 
    S[0] = S0[None,:].repeat(n_paths, axis = 0)
    
    v = np.empty(shape = (N, n_paths, d)) 
    v[0] = params["v0"][None,:].repeat(n_paths, axis = 0)
    
    for i in range(N-1):
        v[i+1] = g(v[i] + params["kappa"] * (params["theta"] - v[i]) * dt +\
                   params["sigma"] * np.sqrt(v[i] * dt) * G[i,:,d+1:])
                   
        S[i+1] = S[i] + params["r"] * S[i] * dt +\
            np.sqrt(v[i] * dt) * S[i] * G[i,:,:d]
            
    return S

def heston_1d_torch(S0, T, N, n_paths, params,
             g = torch.abs, rng = torch.Generator()):
    
    dt = T/(N-1)
    G = torch.randn(size = (N-1 ,n_paths, 2), generator=rng)
    
    S = torch.empty(size = (N, n_paths)) 
    S[0] = torch.tensor(S0, dtype = torch.float32).repeat(n_paths)
    
    v = torch.empty(size = (N, n_paths)) 
    v[0] = torch.tensor(params["v0"], dtype = torch.float32).repeat(n_paths)

    
    for i in range(N-1):
        v[i+1] = g(v[i] + params["kappa"] * (params["theta"] - v[i]) * dt +\
                   params["sigma"] * torch.sqrt(v[i] * dt) * G[i,:,0])
                   
        S[i+1] = S[i] + params["r"] * S[i] * dt +\
            torch.sqrt(v[i] * dt) * S[i] * (params["rho"] * G[i,:,0] + math.sqrt(1 - params["rho"]**2) * G[i,:,1])
            
    return S

def heston_multid_torch(S0, T, N, n_paths, params, d,
             g = torch.abs, rng = torch.Generator()):
    
    dt = T/(N-1)

    L = torch.linalg.cholesky(params["rho"])
    
    Z = torch.randn(size = (N-1 ,n_paths, 2 * d), generator = rng)
    G = torch.einsum("xy, nmy -> nmx", L, Z) # first d -> prices, next d -> volatilities

    
    S = torch.empty(size = (N, n_paths, d)) 
    S[0] = S0[None,:].repeat_interleave(n_paths, dim = 0)
    
    v = torch.empty(size = (N, n_paths, d)) 
    v[0] = params["v0"][None,:].repeat_interleave(n_paths, dim = 0)
    
    for i in range(N-1):
        v[i+1] = g(v[i] + params["kappa"] * (params["theta"] - v[i]) * dt +\
                   params["sigma"] * torch.sqrt(v[i] * dt) * G[i,:,d+1:])
                   
        S[i+1] = S[i] + params["r"] * S[i] * dt +\
            torch.sqrt(v[i] * dt) * S[i] * G[i,:,:d]
            
    return S

