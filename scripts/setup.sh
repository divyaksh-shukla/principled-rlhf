#!/bin/bash
# This script sets up the environment for the RLHF experiment.

# Install and setup the environment
conda create -n comprehensive python=3.11 -y

conda activate comprehensive
pip3 install torch --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt
