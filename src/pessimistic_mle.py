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

class PessimisticMLE:
    """Pessimistic MLE implementation"""
    
    def __init__(self, env, mle_estimator, confidence_level=0.05, device='cpu'):
        self.env = env
        self.mle = mle_estimator
        self.delta = confidence_level
        self.device = device
        self.sigma_d = None
        self.confidence_radius = None
        
    def compute_data_covariance(self):
        """Compute data covariance matrix Sigma_D"""
        if len(self.mle.data) == 0:
            return torch.eye(self.env.d, device=self.device)
        
        n = len(self.mle.data)
        sigma_d = torch.zeros((self.env.d, self.env.d), device=self.device)
        
        for feat1, feat2, _ in self.mle.data:
            diff = feat1 - feat2
            sigma_d += torch.outer(diff, diff)
            
        self.sigma_d = sigma_d / n
        return self.sigma_d
    
    def compute_confidence_radius(self, lambda_reg=0.01):
        """Compute confidence radius for pessimistic MLE"""
        n = len(self.mle.data)
        if n == 0:
            return 1.0
            
        d = self.env.d
        
        # Compute gamma (from paper)
        L = torch.max(torch.norm(self.env.action_features, dim=1))
        gamma = 1 / (2 + torch.exp(-L * self.env.B) + torch.exp(L * self.env.B))
        
        # Confidence radius
        radius = torch.sqrt((d + torch.log(torch.tensor(1/self.delta))) / (gamma**2 * n) + lambda_reg * self.env.B**2)
        
        self.confidence_radius = radius
        return radius
    
    def pessimistic_value(self, action, theta_hat, lambda_reg=0.01):
        """Compute pessimistic value for an action"""
        if self.sigma_d is None:
            self.compute_data_covariance()
            
        if self.confidence_radius is None:
            self.compute_confidence_radius(lambda_reg)
            
        action_feat = self.env.action_features[action]
        
        # Pessimistic reward = point estimate - confidence penalty
        point_estimate = torch.dot(theta_hat, action_feat)
        
        # Confidence penalty
        regularized_cov = self.sigma_d + lambda_reg * torch.eye(self.env.d, device=self.device)
        try:
            L = torch.linalg.cholesky(regularized_cov + 1e-6 * torch.eye(self.env.d, device=self.device))
            inv_sqrt_cov = torch.linalg.inv(L)
            penalty = torch.norm(torch.matmul(inv_sqrt_cov, action_feat)) * self.confidence_radius
        except:
            penalty = torch.norm(action_feat) * self.confidence_radius
            
        return point_estimate - penalty
    
    def get_pessimistic_policy(self, lambda_reg=0.01):
        """Get pessimistic policy"""
        theta_hat = self.mle.get_theta()
        
        breakpoint()
        pessimistic_values = torch.zeros(4, device=self.device)
        for action in range(4):
            pessimistic_values[action] = self.pessimistic_value(action, theta_hat, lambda_reg)
            
        return torch.argmax(pessimistic_values).item()
