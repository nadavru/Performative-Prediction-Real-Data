import torch
from torch import nn
import numpy as np
from data_prep import load_data

class EnvironmentStateless:
    def __init__(self, tran="tran1", data_folder="GiveMeSomeCredit", **kwargs):
        assert tran in ["tran1", "tran2"]
        ########################################
        self.x_src, self.y_src, _ = load_data(f"data/{data_folder}/train.csv")
        self.x_src = torch.tensor(self.x_src).float()
        self.y_src = torch.tensor(self.y_src).float()

        self.n_examples, self.n_features = self.x_src.shape
        ########################################
        self.meta = kwargs
        
        self.theta = torch.zeros((self.n_features+1,))
        
        self.strat_features = None
        if tran=="tran1":
            self.strat_features = torch.Tensor([1, 6, 8]).long() - 1
        elif tran=="tran2":
            self.strat_features = torch.arange(self.n_features)
        
        self.x, self.y = self.x_src, self.y_src

    @staticmethod
    def _update_theta(model, theta):
        ind = 0
        for p in model.parameters():
            if p.requires_grad:
                p_size = p.numel()
                theta[ind:ind+p_size] = p.detach().view(-1)
                ind += p_size
    
    def _update_x_y(self, theta):
        temp = torch.zeros(self.n_features)
        temp[self.strat_features] = theta[self.strat_features]
        new_x = self.x_src+self.meta["eps"]*temp.view(1,-1)
        new_y = self.y_src
        
        return new_x, new_y
    
    def peek(self, model, k=1):
        theta = torch.zeros_like(self.theta)
        self._update_theta(model, theta)
        
        # It doesn't matter if k>1! Stateless world...
        x, y = self._update_x_y(theta)
        return x, y
    
    def step(self, model):
        last_theta = self.theta.clone()
        self._update_theta(model, self.theta)
        theta_diff = torch.dist(self.theta, last_theta)

        self.x, self.y = self._update_x_y(self.theta)
        return theta_diff
    
    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.theta = self.theta.to(device)
        self.device = device

        return self
