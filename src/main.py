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
    # # Run experiments
    # experiment = PyTorchRLHFExperiment(env, device=device)
    # sample_sizes = [20] + np.arange(50, 501, 50).tolist()  # Sample sizes from 50 to 500 in steps of 50
    
    # print(f"\nRunning experiments with sample sizes: {sample_sizes}")
    # experiment.run_experiment(sample_sizes, num_trials=20, epochs=300)
    
    # # Plot results
    # print("\nGenerating plots...")
    # fig = experiment.plot_results()
    
    # # Print summary results
    # print("\nExperimental Results Summary:")
    # print("-" * 40)
    # for i, n in enumerate(sample_sizes):
    #     print(f"n={n:3d}: MLE Error={experiment.results['mle_estimation_error'][i]:.4f}, "
    #           f"MLE SubOpt={experiment.results['mle_suboptimality'][i]:.4f}, "
    #           f"Pess SubOpt={experiment.results['pessimistic_suboptimality'][i]:.4f}")

###############################################################################
# Plot 2
###############################################################################
    # Demonstrate K-wise comparison
    
    sample_sizes = [10] + np.arange(100, 501, 100).tolist()  # Sample sizes from 50 to 500 in steps of 50
    # env = KRandomBanditRLHFEnv(device=device, actions=4)  # Use 4 actions for K-wise comparison
    # experiment = KwiseComparisonExperiment(env, sample_sizes=sample_sizes, K=4, device=device)
    
    # print(f"\nRunning K=4-wise comparison experiments with sample sizes: {sample_sizes}")
    # experiment.run_experiment(num_trials=50, epochs=300)
    
    # # Plot K-wise comparison results
    # experiment.plot_results()
    
    env = KRandomBanditRLHFEnv(device=device, actions=9)  # Use 9 actions for K-wise comparison
    experiment = KwiseComparisonExperiment(env, sample_sizes=sample_sizes, K=9, device=device)
    
    print(f"\nRunning K=9-wise comparison experiments with sample sizes: {sample_sizes}")
    experiment.run_experiment(num_trials=50, epochs=300)
    
    # Plot K-wise comparison results
    experiment.plot_results()
    
    print("DONE")
    print("=" * 50)
    
    
    # print(f"\nK-wise Comparison Demo:")
    # print("-" * 30)
    # kwise = KwiseComparison(env, K=4, device=device)
    
    # # Generate some K-wise rankings
    # actions = [0, 1, 2, 3]
    # for _ in range(20):
    #     ranking = kwise.sample_ranking(actions)
    #     kwise.add_ranking(actions, ranking)
    
    # # Fit both estimators
    # theta_mlek = kwise.fit_mle_k(epochs=500)
    # theta_mle2 = kwise.fit_mle_2(epochs=500)
    
    # print(f"True theta*:      {env.theta_star.cpu().numpy()}")
    # print(f"MLEK estimate:    {theta_mlek.cpu().numpy()}")
    # print(f"MLE2 estimate:    {theta_mle2.cpu().numpy()}")
    # print(f"MLEK error:       {torch.norm(theta_mlek - env.theta_star).item():.4f}")
    # print(f"MLE2 error:       {torch.norm(theta_mle2 - env.theta_star).item():.4f}")

if __name__ == "__main__":
    main()
