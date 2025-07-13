import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

class MLENetwork(nn.Module):
    """Neural network for MLE estimation with linear reward model"""
    
    def __init__(self, d, B=2.0, device='cpu'):
        super().__init__()
        self.d = d
        self.B = B
        self.device = device
        
        # Linear layer for reward prediction
        self.reward_head = nn.Linear(d, 1, bias=False)
        
        # Initialize with small random weights
        nn.init.normal_(self.reward_head.weight, 0, 0.1)
        
    def forward(self, features):
        """Forward pass through the network"""
        return self.reward_head(features).squeeze(-1)
    
    def project_parameters(self):
        """Project parameters to constraint set"""
        with torch.no_grad():
            # Get current weights
            weights = self.reward_head.weight.data.squeeze()
            
            # Project to constraint set
            weights = weights - torch.mean(weights)
            norm = torch.norm(weights)
            if norm > self.B:
                weights = weights * self.B / norm
                
            # Update weights
            self.reward_head.weight.data = weights.unsqueeze(0)

class MLEEstimator:
    """Maximum Likelihood Estimator for pairwise comparisons"""
    
    def __init__(self, env, lr=0.01, device='cpu'):
        self.env = env
        self.device = device
        self.network = MLENetwork(env.d, env.B, device).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.data = []
        
    def add_comparison(self, action1, action2, preference):
        """Add a pairwise comparison to the dataset"""
        feat1 = self.env.action_features[action1]
        feat2 = self.env.action_features[action2]
        self.data.append((feat1, feat2, preference))
    
    def compute_loss(self, batch_size=None):
        """Compute negative log-likelihood loss for BTL model"""
        if len(self.data) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Use all data if batch_size not specified
        if batch_size is None:
            batch_size = len(self.data)
        
        # Sample batch
        indices = torch.randperm(len(self.data))[:batch_size]
        
        total_loss = 0.0
        
        # Get features and preferences for the batch and sort them into their indices
        feat1, feat2, y = [], [], []
        for d in self.data:
            feat1.append(d[0])
            feat2.append(d[1])
            y.append(d[2])
        feat1, feat2, y = torch.stack(feat1), torch.stack(feat2), torch.tensor(y, device=self.device)
        feat1, feat2, y = feat1[indices], feat2[indices], y[indices]
        
        # Forward pass to get rewards
        r1 = self.network(feat1)
        r2 = self.network(feat2)
        
        # BTL loss
        loss1 = -F.logsigmoid(r1 - r2)
        loss2 = -F.logsigmoid(r2 - r1)
        total_loss = torch.where(y == 1, loss1, loss2).sum()
        total_loss /= batch_size
        return total_loss
    
    def fit(self, epochs=1000, batch_size=None, verbose=False):
        """Fit MLE using PyTorch optimization"""
        if len(self.data) == 0:
            return
        
        self.network.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            loss = self.compute_loss(batch_size)
            loss.backward()
            
            self.optimizer.step()
            
            # Project parameters to constraint set
            self.network.project_parameters()
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        return self.get_theta()
    
    def get_theta(self):
        """Get current parameter estimate"""
        return self.network.reward_head.weight.data.squeeze().clone()
    
    def get_greedy_policy(self):
        """Get greedy policy under estimated theta"""
        self.network.eval()
        with torch.no_grad():
            rewards = self.network(self.env.action_features)
            return torch.argmax(rewards).item()
