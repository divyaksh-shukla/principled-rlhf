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

class KRandomBanditRLHFEnv(gym.Env):
    """
    Gymnasium environment for random action feature bandit RLHF setting.
    Based on the example in Appendix B.3 (Theorem 3.9 proof).
    """
    
    def __init__(self, d=3, B=2.0, device='cpu', actions=4):
        super().__init__()
        
        self.d = d
        self.B = B
        self.actions = actions
        self.device = device
        # set the seed for reproducibility
        self.seed()
        
        # Define action space and observation space
        self.action_space = spaces.Discrete(actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(d,), dtype=np.float32)
        
        # Define the actions with their features (from Appendix B.3)
        self.action_features = torch.randn((actions, d), device=device, dtype=torch.float32)
        
        # True reward parameter (from the paper's example)
        self.theta_star = torch.randn((d), device=device, dtype=torch.float32)
        
        # Ensure theta_star satisfies constraints
        # assert torch.equal(self.theta_star, self._project_to_constraint_set(self.theta_star)), \
        #     "theta_star must be in the constraint set Theta_B = {theta: <1, theta> = 0, ||theta||_2 <= B}"
        self.theta_star = self._project_to_constraint_set(self.theta_star)
        
        # Current state (single state for contextual bandit)
        self.state = torch.zeros(d, device=device, dtype=torch.float32)
        
        # Optimal action under true reward
        self.optimal_action = self._get_optimal_action(self.theta_star)
        
        
    
    def seed(self, seed=42):
        """Set the random seed for reproducibility"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        return seed
        
    def _project_to_constraint_set(self, theta):
        """Project theta to the constraint set Theta_B = {theta: <1, theta> = 0, ||theta||_2 <= B}"""
        # First, make sum equal to 0, i.e. recenter the parameters
        theta_proj = theta - torch.mean(theta)
        
        # Then project to L2 ball
        norm = torch.norm(theta_proj)
        if norm > self.B:
            theta_proj = theta_proj * self.B / norm
            
        return theta_proj
    
    def _get_optimal_action(self, theta):
        """Get the optimal action under given theta"""
        rewards = torch.matmul(self.action_features, theta)
        return torch.argmax(rewards).item()
    
    def get_reward(self, action, theta=None):
        """Get reward for an action under given theta (or true theta)"""
        if theta is None:
            theta = self.theta_star
        return torch.dot(theta, self.action_features[action])
    
    def sample_comparison(self, action1, action2, theta=None):
        """
        Sample a pairwise comparison under Bradley-Terry-Luce model.
        Returns 1 if action1 is preferred, 0 if action2 is preferred.
        """
        if theta is None:
            theta = self.theta_star
            
        r1 = self.get_reward(action1, theta)
        r2 = self.get_reward(action2, theta)
        
        # BTL model probability
        prob_1_wins = torch.exp(r1) / (torch.exp(r1) + torch.exp(r2))
        
        return torch.bernoulli(prob_1_wins).item()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = torch.zeros(self.d, device=self.device, dtype=torch.float32)
        return self.state.cpu().numpy(), {}
    
    def step(self, action):
        reward = self.get_reward(action).item()
        terminated = True  # Single step episode for contextual bandit
        truncated = False
        info = {
            'optimal_action': self.optimal_action,
            'optimal_reward': self.get_reward(self.optimal_action).item(),
            'action_features': self.action_features[action].cpu().numpy(),
            'reward': reward,
            'optimal_action_comparison': self.sample_comparison(action, self.optimal_action)
        }
        
        return self.state.cpu().numpy(), reward, terminated, truncated, info
