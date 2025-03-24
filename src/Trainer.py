#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 14:10:35 2025

@author: aurelienperez
"""

from tqdm import tqdm

class Trainer():
    def __init__(self, model, env, data_generator, pnl_calc, risk_measure):
        self.model = model
        self.market_env = env.market
        self.training_env = env.training
        self.data_generator = data_generator
        self.pnl_calc = pnl_calc
        self.risk_measure = risk_measure
        
    
    def train(self, n_grad_upd = 128):
        losses = []
        with tqdm(range(n_grad_upd), desc="Training") as pbar:
            for _ in pbar:
                discounted_path, discounted_payoff = self.data_generator.generate_batch()
                pnl = self.pnl_calc.compute_pnl(discounted_path, discounted_payoff, self.model)
                loss = self.risk_measure(pnl)
                self.training_env.optimizer_p0.zero_grad()
                self.training_env.optimizer_rest.zero_grad()
                loss.backward()
                self.training_env.optimizer_rest.step()
                self.training_env.optimizer_p0.step()
                losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())
        return losses