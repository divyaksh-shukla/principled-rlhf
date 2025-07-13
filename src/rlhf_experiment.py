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
import os
from pathlib import Path

from mle import MLEEstimator
from pessimistic_mle import PessimisticMLE

from linear_bandit_env import LinearBanditRLHFEnv
from k_random_bandit_env import KRandomBanditRLHFEnv
from k_wise_comparison import KwiseComparison
from tqdm import tqdm

class PyTorchRLHFExperiment:
    """PyTorch-based experimental framework for RLHF comparison"""
    
    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = device
        self.results = {
            'mle_estimation_error': [],
            'mle_estimation_error_std': [],
            'mle_suboptimality': [],
            'mle_suboptimality_std': [],
            'pessimistic_suboptimality': [],
            'pessimistic_suboptimality_std': [],
            'sample_sizes': []
        }
    
    def run_experiment(self, sample_sizes, num_trials=100, epochs=500):
        """Run the experiment comparing MLE and Pessimistic MLE"""
        
        for n in sample_sizes:
            print(f"Running experiment with n={n} samples...")
            
            mle_errors = []
            mle_subopt = []
            pess_subopt = []
            
            
            for _ in range(num_trials):
                # Generate data following the paper's setup
                mle_est = MLEEstimator(self.env, device=self.device)
                
                # Generate n-1 comparisons between a1 and a2
                for _ in range(n-1):
                    preference = self.env.sample_comparison(0, 1)  # a1 vs a2
                    mle_est.add_comparison(0, 1, preference)
                
                # Generate 1 comparison between a2 and a3
                preference = self.env.sample_comparison(1, 2)  # a2 vs a3
                mle_est.add_comparison(1, 2, preference)
                
                feat1, feat2 = [], []
                for d in mle_est.data:
                    feat1.append(d[0])
                    feat2.append(d[1])
                feat1, feat2 = torch.stack(feat1), torch.stack(feat2)
                action_diffs = feat1 - feat2  # Differences in features
                # Compute the variance of the action differences
                action_diffs = action_diffs.T @ action_diffs  # Outer product to get variance matrix
                action_diffs = action_diffs / (n - 1)  # Normalize by number of samples
                # action_diffs = action_diffs.sqrt()
                
                # Fit MLE
                theta_hat = mle_est.fit(epochs=epochs)
                
                # Compute estimation error
                parameter_diff = (theta_hat - self.env.theta_star).view(-1, 1)  # Ensure it's a column vector
                error = parameter_diff.T @ action_diffs @ parameter_diff
                error = torch.sqrt(error).item()  # Convert to scalar
                mle_errors.append(error)
                
                # Compute suboptimality for MLE policy
                mle_policy = mle_est.get_greedy_policy()
                mle_reward = self.env.get_reward(mle_policy).item()
                optimal_reward = self.env.get_reward(self.env.optimal_action).item()
                mle_subopt.append(optimal_reward - mle_reward)
                
                # Compute suboptimality for Pessimistic MLE policy
                pess_mle = PessimisticMLE(self.env, mle_est, device=self.device)
                pess_policy = pess_mle.get_pessimistic_policy()
                pess_reward = self.env.get_reward(pess_policy).item()
                pess_subopt.append(optimal_reward - pess_reward)
            
            # Store results
            self.results['sample_sizes'].append(n)
            self.results['mle_estimation_error'].append(np.mean(mle_errors))
            self.results['mle_estimation_error_std'].append(np.std(mle_errors))
            self.results['mle_suboptimality'].append(np.mean(mle_subopt))
            self.results['mle_suboptimality_std'].append(np.std(mle_subopt))
            self.results['pessimistic_suboptimality'].append(np.mean(pess_subopt))
            self.results['pessimistic_suboptimality_std'].append(np.std(pess_subopt))
    
    def plot_results(self):
        """Plot experimental results"""
        self.save_results()  # Save results before plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: MLE estimation error
        # ax1.loglog(self.results['sample_sizes'], self.results['mle_estimation_error'], 
                #    'b-o', label='MLE Estimation Error')
        ax1.plot(self.results['sample_sizes'], self.results['mle_estimation_error'], 
                   'b-o', label='MLE Estimation Error')
        ax1.fill_between(self.results['sample_sizes'], 
                         np.array(self.results['mle_estimation_error']) - np.array(self.results['mle_estimation_error_std']),
                         np.array(self.results['mle_estimation_error']) + np.array(self.results['mle_estimation_error_std']),
                         color='blue', alpha=0.2)
        ax1.set_xlabel('Number of Samples (n)')
        ax1.set_ylabel('Estimation Error ||θ̂ - θ*||')
        ax1.set_title('MLE Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Suboptimality comparison
        # ax2.loglog(self.results['sample_sizes'], self.results['mle_suboptimality'], 
        #            'r-s', label='MLE Suboptimality')
        # ax2.loglog(self.results['sample_sizes'], self.results['pessimistic_suboptimality'], 
        #            'g-^', label='Pessimistic MLE Suboptimality')
        ax2.plot(self.results['sample_sizes'], self.results['mle_suboptimality'], 
                   'r-s', label='MLE Suboptimality')
        ax2.fill_between(self.results['sample_sizes'], 
                         np.array(self.results['mle_suboptimality']) - np.array(self.results['mle_suboptimality_std']),
                         np.array(self.results['mle_suboptimality']) + np.array(self.results['mle_suboptimality_std']),
                         color='red', alpha=0.2)
        ax2.plot(self.results['sample_sizes'], self.results['pessimistic_suboptimality'], 
                   'g-^', label='Pessimistic MLE Suboptimality')
        ax2.fill_between(self.results['sample_sizes'], 
                         np.array(self.results['pessimistic_suboptimality']) - np.array(self.results['pessimistic_suboptimality_std']),
                         np.array(self.results['pessimistic_suboptimality']) + np.array(self.results['pessimistic_suboptimality_std']),
                         color='green', alpha=0.2)
        ax2.set_xscale('linear')
        ax2.set_yscale('linear')
        ax2.set_xlabel('Number of Samples (n)')
        ax2.set_ylabel('Suboptimality')
        ax2.set_title('Policy Performance Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('figures/rlhf_experiment_results.pdf', bbox_inches='tight')
        
        return fig
    
    def save_results(self, filename=None):
        """Save results to a file"""
        if filename is None:
            os.makedirs('saves', exist_ok=True)
            filename = 'saves/rlhf_experiment_results.npz'
            
        np.savez(filename, **self.results)
    
    def load_results(self, filename=None):
        """Load results from a file"""
        if filename is None:
            filename = 'saves/rlhf_experiment_results.npz'
        
        data = np.load(filename)
        self.results = {key: data[key] for key in data.files}
        self.plot_results()  # Automatically plot loaded results
    
class KwiseComparisonExperiment:
    """K-wise comparison experiment for RLHF"""
    
    def __init__(self, env, sample_sizes, K=4, device='cpu'):
        self.env = env
        self.K = K
        self.device = device
        
        self.sample_sizes = sample_sizes
        self.results = {
            'theta_mlek': [],
            'theta_mle2': [],
            'theta_mlek_error': [],
            'theta_mle2_error': [],
            'k_action_feat': [],
            '2_action_feat': []
        }
    
    def run_experiment(self, num_trials=100, epochs=500):
        """Run K-wise comparison experiment"""
        
        pbar = tqdm(total=len(self.sample_sizes), ncols=120, desc=f"K={self.K}-wise Comparison")
        for n in self.sample_sizes:
            pbar.write(f"Running K-wise comparison with n={n} samples...")
            trial_results = {
                'theta_mlek': [],
                'theta_mle2': [],
                'theta_mlek_error': [],
                'theta_mle2_error': [],
                'k_action_feat': [],
                '2_action_feat': []
            }
            
            for _ in range(num_trials):
                kwise = KwiseComparison(self.env, K=self.K, device=self.device)
                # Generate some K-wise rankings
                actions = list(range(self.env.action_space.n))
                for _ in range(n):
                    ranking = kwise.sample_ranking(actions)
                    kwise.add_ranking(actions, ranking)
                
                # Fit both estimators
                theta_mlek = kwise.fit_mle_k(epochs=epochs)
                theta_mle2 = kwise.fit_mle_2(epochs=epochs)
                
                features, ranking = [], []
                for feat, rank in kwise.data:
                    features.append(feat)
                    ranking.append(rank)
                features = torch.stack(features)
                ranking = torch.tensor(ranking, device=self.device)
                
                # Compute error
                parameter_diff_mlek = (theta_mlek - self.env.theta_star).view(-1, 1)
                # make a sparse matrix of action features
                action_features = self.env.action_features[ranking]
                diffs = action_features[:, :-1, :] - action_features[:, 1:, :]  # Differences in features
                diffs = diffs.sum(dim=1)  # Sum over K action differences
                action_diffs = diffs.T @ diffs  # Outer product to get variance matrix
                action_diffs = action_diffs / (n - 1)  # Normalize by
                
                theta_mlek_error = torch.sqrt((parameter_diff_mlek.T @ action_diffs @ parameter_diff_mlek)).item()
                theta_mle2_error = torch.norm(theta_mle2 - self.env.theta_star).item()
                
                trial_results['theta_mlek'].append(theta_mlek.cpu().numpy())
                trial_results['theta_mle2'].append(theta_mle2.cpu().numpy())
                trial_results['theta_mlek_error'].append(theta_mlek_error)
                trial_results['theta_mle2_error'].append(theta_mle2_error)
                trial_results['k_action_feat'].append(features.cpu().numpy())
                trial_results['2_action_feat'].append(self.env.action_features[ranking].cpu().numpy())
            
            self.results['theta_mlek'].append(np.mean(trial_results['theta_mlek'], axis=0))
            self.results['theta_mle2'].append(np.mean(trial_results['theta_mle2'], axis=0))
            self.results['theta_mlek_error'].append(np.mean(trial_results['theta_mlek_error']))
            self.results['theta_mle2_error'].append(np.mean(trial_results['theta_mle2_error']))
            self.results['k_action_feat'].append(np.mean(trial_results['k_action_feat'], axis=0))
            self.results['2_action_feat'].append(np.mean(trial_results['2_action_feat'], axis=0))  
            pbar.update(1) 
            
        self.save_results()  # Save results after running the experiment
    
    def plot_results(self):
        """Plot K-wise comparison results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.sample_sizes, self.results['theta_mlek_error'],
                'b-o', label='MLEK Error')
        ax.plot(self.sample_sizes, self.results['theta_mle2_error'],
                'r-s', label='MLE2 Error')
        
        ax.set_xlabel('Sample Sizes')
        ax.set_ylabel('Estimation Error ||θ̂ - θ*||')
        ax.set_title('K-wise Comparison Results')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'figures/kwise_comparison_results_K{self.K}.pdf', bbox_inches='tight')
    
    def save_results(self, filename=None):
        """Save K-wise comparison results to a file"""
        import pickle
        if filename is None:
            os.makedirs('saves', exist_ok=True)
            filename = f'saves/kwise_comparison_results_K{self.K}.npz'
        
        pickle.dump(self.results, open(filename, 'wb'))
    
    def load_results(self, filename=None):
        """Load K-wise comparison results from a file"""
        import pickle
        if filename is None:
            filename = f'saves/kwise_comparison_results_K{self.K}.npz'
        
        self.results = pickle.load(open(filename, 'rb'))
        self.plot_results()

if __name__ == "__main__":
    """RLHF experiment main entry point"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize environment
    env = LinearBanditRLHFEnv(device=device)
    
    print(f"Environment Setup:")
    print(f"- True theta*: {env.theta_star}")
    print(f"- Optimal action: {env.optimal_action}")
    print(f"- Action features:")
    for i, feat in enumerate(env.action_features):
        reward = env.get_reward(i).item()
        print(f"  a{i+1}: {feat.cpu().numpy()} -> reward: {reward:.3f}")
    
    # Run experiments
    experiment = PyTorchRLHFExperiment(env, device=device)
    
    # load and plot the results if they exist
    results_file = 'saves/rlhf_experiment_results.npz'
    if Path(results_file).exists():
        print(f"Loading existing results from {results_file}")
        experiment.load_results(results_file)
    else:
        print("No existing results found, please run the experiment.")
