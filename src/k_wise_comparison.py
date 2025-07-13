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

from mle import MLENetwork, MLEEstimator

class KwiseComparison:
    """PyTorch implementation for K-wise comparisons using Plackett-Luce model"""
    
    def __init__(self, env, K=4, device='cpu'):
        self.env = env
        self.K = K
        self.device = device
        self.data = []
        self.network = MLENetwork(env.d, env.B, device).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)
        
    def sample_ranking(self, actions, theta=None):
        """Sample a ranking under Plackett-Luce model"""
        if theta is None:
            theta = self.env.theta_star
            
        # Compute rewards for all actions
        action_indices = torch.tensor(actions, device=self.device)
        rewards = torch.matmul(self.env.action_features[action_indices], theta)
        
        # Sample ranking using PL model
        ranking = []
        remaining_actions = list(range(len(actions)))
        remaining_rewards = rewards.clone()
        
        for _ in range(len(actions)):
            # Compute probabilities for remaining actions
            exp_rewards = torch.exp(remaining_rewards)
            probs = exp_rewards / torch.sum(exp_rewards)
            
            # Sample next action
            choice_idx = torch.multinomial(probs, 1).item()
            chosen_action = remaining_actions[choice_idx]
            
            ranking.append(chosen_action)
            remaining_actions.pop(choice_idx)
            remaining_rewards = torch.cat([remaining_rewards[:choice_idx], 
                                         remaining_rewards[choice_idx+1:]])
            
        return ranking
    
    def add_ranking(self, actions, ranking):
        """Add a K-wise ranking to the dataset"""
        action_features = self.env.action_features[actions]
        self.data.append((action_features, ranking))
    
    def compute_mlek_loss(self):
        """Compute loss for true K-wise MLE (MLEK)"""
        if len(self.data) == 0:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        
        features, ranking = [], []
        for feat, rank in self.data:
            features.append(feat)
            ranking.append(rank)
        features = torch.stack(features)
        features = features[0]
        ranking = torch.tensor(ranking, device=self.device)
        
        # breakpoint()
        for k in range(ranking.shape[1]):
            try:
                # Compute numerator (reward for the k-th item in the ranking)
                numerator = self.network(features[ranking[:, k]].unsqueeze(1))
                # Compute denominator (sum over remaining items)
                remaining_features = features[ranking[:, k:]]
                remaining_rewards = self.network(remaining_features)
                denominator = torch.logsumexp(remaining_rewards, dim=1)
                
                total_loss -= (numerator - denominator).sum()
            except Exception as e:
                breakpoint()
        return total_loss / len(self.data)
        
        # for features, ranking in self.data:
        #     for k in range(len(ranking)):
        #         numerator = self.network(features[ranking[k]].unsqueeze(0))
                
        #         # Compute denominator (sum over remaining items)
        #         remaining_features = features[ranking[k:]]
        #         remaining_rewards = self.network(remaining_features)
        #         denominator = torch.logsumexp(remaining_rewards, dim=0)
                
        #         total_loss -= (numerator - denominator)
                
        # return total_loss / len(self.data)
    
    def fit_mle_k(self, epochs=1000):
        """Fit MLE for K-wise comparisons (MLEK)"""
        if len(self.data) == 0:
            return torch.zeros(self.env.d, device=self.device)
        
        self.network.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            loss = self.compute_mlek_loss()
            loss.backward()
            
            self.optimizer.step()
            self.network.project_parameters()
            
        return self.network.reward_head.weight.data.squeeze().clone()
    
    def fit_mle_2(self, epochs=1000):
        """Fit MLE by splitting K-wise into pairwise (MLE2)"""
        pairwise_mle = MLEEstimator(self.env, device=self.device)
        
        # Convert K-wise rankings to pairwise comparisons
        for features, ranking in self.data:
            for i in range(len(ranking)):
                for j in range(i+1, len(ranking)):
                    # ranking[i] is preferred over ranking[j]
                    # Find original action indices
                    feat_i = features[ranking[i]]
                    feat_j = features[ranking[j]]
                    
                    # Find which actions these correspond to
                    action_i = ranking[i]
                    action_j = ranking[j]
                    
                    pairwise_mle.add_comparison(action_i, action_j, 1)
        
        return pairwise_mle.fit(epochs=epochs)
