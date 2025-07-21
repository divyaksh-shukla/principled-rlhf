import torch
import numpy as np

from linear_bandit_env import LinearBanditRLHFEnv
from k_random_bandit_env import KRandomBanditRLHFEnv
from mle import MLEEstimator
from rlhf_experiment import PyTorchRLHFExperiment, KwiseComparisonExperiment
from k_wise_comparison import KwiseComparison

def main():
    """Main execution function with PyTorch and Gymnasium"""
    print("PyTorch + Gymnasium RLHF Implementation")
    print("=" * 50)
    
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
    
###############################################################################
# Plot 1
###############################################################################
    # Run experiments
    experiment = PyTorchRLHFExperiment(env, device=device)
    sample_sizes = [20] + np.arange(50, 501, 50).tolist()  # Sample sizes from 50 to 500 in steps of 50
    
    print(f"\nRunning experiments with sample sizes: {sample_sizes}")
    experiment.run_experiment(sample_sizes, num_trials=20, epochs=300)
    
    # Plot results
    print("\nGenerating plots...")
    fig = experiment.plot_results()
    
    # Print summary results
    print("\nExperimental Results Summary:")
    print("-" * 40)
    for i, n in enumerate(sample_sizes):
        print(f"n={n:3d}: MLE Error={experiment.results['mle_estimation_error'][i]:.4f}, "
              f"MLE SubOpt={experiment.results['mle_suboptimality'][i]:.4f}, "
              f"Pess SubOpt={experiment.results['pessimistic_suboptimality'][i]:.4f}")

###############################################################################
# Plot 2
###############################################################################
    # Demonstrate K-wise comparison
    
    sample_sizes = [10] + np.arange(100, 501, 100).tolist()  # Sample sizes from 50 to 500 in steps of 50
    env = KRandomBanditRLHFEnv(device=device, actions=4)  # Use 4 actions for K-wise comparison
    experiment = KwiseComparisonExperiment(env, sample_sizes=sample_sizes, K=4, device=device)
    
    print(f"\nRunning K=4-wise comparison experiments with sample sizes: {sample_sizes}")
    experiment.run_experiment(num_trials=50, epochs=300)
    
    # Plot K-wise comparison results
    experiment.plot_results()
    
    env = KRandomBanditRLHFEnv(device=device, actions=9)  # Use 9 actions for K-wise comparison
    experiment = KwiseComparisonExperiment(env, sample_sizes=sample_sizes, K=9, device=device)
    
    print(f"\nRunning K=9-wise comparison experiments with sample sizes: {sample_sizes}")
    experiment.run_experiment(num_trials=10, epochs=300)
    
    # Plot K-wise comparison results
    experiment.plot_results()
    
    print("DONE")
    print("=" * 50)

if __name__ == "__main__":
    main()
