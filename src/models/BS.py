#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 14:21:01 2025

@author: aurelienperez
"""


import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt


class BS():
    
    def __init__(self, r, sigma, rho = None):

        self.params = dict(r = r, sigma = sigma, rho = rho)
    
    def generate_trajectories(self, S0, T, N, n_paths):
        
        if np.isscalar(S0):
            return bs_1d(S0, T, N, n_paths, self.params)
        else:
            d = len(S0)
            return bs_multid(S0, T, N, n_paths, self.params, d)
            
        
        
def bs_1d(S0, T, N, n_paths, params, rng = default_rng()):
    
    dt = T/(N-1)
    G = rng.standard_normal(size = (N-1 ,n_paths))
    
    S = np.empty(shape = (N, n_paths)) 
    S[0] = np.array(S0).repeat(n_paths)
    
    S[1:] = S[0] * np.cumprod(np.exp((params["r"] - 0.5 * params["sigma"]**2) * dt + params["sigma"] *
                      np.sqrt(dt) * G), axis = 0)
    
        
    return S

def bs_multid(S0, T, N, n_paths, params, d, rng = default_rng()):
    
    dt = T/(N-1)

    L = np.linalg.cholesky(params["rho"])
    
    Z = rng.standard_normal(size = (N-1 ,n_paths, d))
    G = np.einsum("xy, nmy -> nmx", L, Z) 

    
    S = np.empty(shape = (N, n_paths, d)) 
    S[0] = S0[None,:].repeat(n_paths, axis = 0)
    
    # S[1:] = S[0] * np.cumprod(np.exp((params["r"] - 0.5 * params["sigma"][None,None,:]**2) * dt + 
    #                                  params["sigma"][None,None,:] * np.sqrt(dt) * G), axis = 0)
    
    for i in range(N-1):
        S[i+1] = S[i] * np.exp((params["r"] - 0.5 * params["sigma"][None,None,:]**2) * dt + 
                                         params["sigma"][None,None,:] * np.sqrt(dt) * G[i])
    
    return S

