import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from normalization import EmpiricalNormalization

class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], use_normalization=True):
        super(Actor, self).__init__()
        
        self.use_normalization = use_normalization
        if use_normalization:
            self.obs_normalizer = EmpiricalNormalization(state_dim)

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.0)

        self._init_weights()

    def _init_weights(self):
        # orthogonal initialization for all layers. output layer has small gain 0.01
        for name, module in self.named_modules():
            if name != "output_layer" and isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
            elif name == "output_layer" and isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # normalize observations, then forward pass
        if self.use_normalization:
            x = self.obs_normalizer(x)
        
        for layer in self.hidden_layers:
            x = F.elu(layer(x))
        mu = self.output_layer(x)
        std = torch.exp(self.log_std)
        return mu, std
    
    def update_normalization(self, obs):
        # update observation normalization statistics
        if self.use_normalization:
            self.obs_normalizer.update(obs)

class Critic(nn.Module):

    def __init__(self, state_dim, hidden_dims=[64, 64], use_normalization=True):
        super(Critic, self).__init__()
        
        self.use_normalization = use_normalization
        if use_normalization:
            self.obs_normalizer = EmpiricalNormalization(state_dim)
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self):
        # orthogonal initialization for all layers. output layer has gain 1
        for name, module in self.named_modules():
            if name != "output_layer" and isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
            elif name == "output_layer" and isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        if self.use_normalization:
            x = self.obs_normalizer(x)
        
        for layer in self.hidden_layers:
            x = F.elu(layer(x))
        x = self.output_layer(x)
        return x
    
    def update_normalization(self, obs):
        # update observation normalization statistics
        if self.use_normalization:
            self.obs_normalizer.update(obs)