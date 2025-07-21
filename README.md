# Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons

An implementation of the paper "Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons"

- [arxiv](http://arxiv.org/abs/2301.11270)
- [PMLR](https://proceedings.mlr.press/v202/zhu23f/zhu23f.pdf)
- [Presentation](https://drive.google.com/file/d/1f-vedJ-mnQGxdQ15O32rctQmuqq3LCyv/view?usp=sharing)
---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Experiments](#experiments)
- [Saving & Loading Results](#saving--loading-results)
- [Visualization](#visualization)

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/divyaksh-shukla/principled-rlhf.git
   cd principled-rlhf
   ```

2. **Setup environment** (Linux/macOS):
```bash
bash scripts/setup.sh
```
For Windows (PowerShell):
```powershell
.\scripts\setup.ps1
```

This will:
- Create a Conda environment (`comprehensive`) with Python 3.11
- Install PyTorch (with CUDA support if available)
- Install all required Python packages

---

## Usage

### Run main RLHF experiment:

```bash
bash scripts/run.sh
```
This executes `src/main.py`, running the core RLHF experiments.

### Manual execution:

```bash
conda activate comprehensive
python src/main.py
```

### Recreate the plots using the saved results:

```bash
python src/rlhf_experiment.py
```

#### Main script functionality:
- Initializes the environment, prints setup details (true parameters, optimal actions, feature vectors).
- Runs RLHF experiments with adjustable sample sizes and trials.
- Plots and saves results to `figures/rlhf_experiment_results.pdf` and `saves/rlhf_experiment_results.npz`.

---

## Repository Structure

```
principled-rlhf/
├── src/
│   ├── main.py                # Main entry point for RLHF experiments
│   ├── rlhf_experiment.py     # Experiment framework (MLE, Pessimistic MLE, K-wise)
│   ├── linear_bandit_env.py   # Linear bandit RLHF environment
│   ├── k_random_bandit_env.py # K-wise bandit RLHF environment
│   ├── mle.py                 # MLE estimator implementation
│   ├── pessimistic_mle.py     # Pessimistic MLE estimator implementation
│   ├── k_wise_comparison.py   # K-wise comparison experiment logic
├── requirements.txt
├── scripts/
│   ├── setup.sh               # Environment setup (Linux/macOS)
│   ├── setup.ps1              # Environment setup (Windows)
│   ├── run.sh                 # Run main experiment script
├── saves/                     # Saved experiment results
├── figures/                   # Generated plots
```

---

## Dependencies

Defined in `requirements.txt`:

- `torch` (installed separately for CUDA support)
- `gymnasium`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`
- Development tools: `jupyter`, `ipython`, `tqdm`

To install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Experiments

### Linear Bandit RLHF

- Compares MLE and Pessimistic MLE estimators.
- Tracks estimation error and suboptimality as a function of sample size.
- Results can be saved/loaded for reproducibility.

### K-wise Comparison

- Supports K-wise ranking experiments (e.g., 4-wise, 9-wise).
- Provides fitting and evaluation utilities for both standard and K-wise estimators.

---

## Saving & Loading Results

- Results are automatically saved in the `saves/` directory.
- Use the provided `load_results()` method in experiment classes to restore results and plots.
- Results are stored as `.npz` (NumPy) or pickled files for K-wise comparison.

---

## Visualization

- Plots are generated and saved as PDF in the `figures/` directory.
- Includes:
  - MLE estimation error vs. sample size
  - Suboptimality comparison (MLE vs. pessimistic MLE)
  - K-wise comparison results

---

## Contact

For questions or collaboration, reach out to [divyaksh-shukla](https://github.com/divyaksh-shukla).
