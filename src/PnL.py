#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:26:11 2025

@author: aurelienperez
"""

import torch
import math

class PnL():
    def __init__(self, market_env):
        self.market_env = market_env
        

    def compute_pnl(self, discounted_path, discounted_payoff, hedge_strategy):
        batch_size, num_steps, d = discounted_path.shape
        # Time 0
        out0 = hedge_strategy(0, discounted_path[:, 0, :])
        if isinstance(out0, tuple):
            p0, delta0 = out0
        else:
            raise ValueError("Strategy at time t=0 must return (p0, delta0)")
        pnl_init = p0 - (delta0 * discounted_path[:, 0, :]).sum(dim=1, keepdim=True)
        delta_prev = delta0
        pnl_rebalance = 0
  
        for n in range(1, self.market_env.N):
            delta_n = hedge_strategy(n, discounted_path[:, n, :])
            trade = delta_n - delta_prev
            cost = self.market_env.tc * (torch.abs(trade) * discounted_path[:, n, :]).sum(dim=1, keepdim=True)
            pnl_trade = - (trade * discounted_path[:, n, :]).sum(dim=1, keepdim=True) - cost
            pnl_rebalance = pnl_rebalance + pnl_trade
            delta_prev = delta_n
        pnl_final = (delta_prev * discounted_path[:, -1, :]).sum(dim=1, keepdim=True)
        pnl_total = pnl_init + pnl_rebalance + pnl_final - discounted_payoff
        pnl_total = pnl_total * math.exp(self.market_env.r * self.market_env.T)
        pnl_real = pnl_total * self.market_env.scaling_factor
        return pnl_real