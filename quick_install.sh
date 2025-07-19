#!/bin/bash

# =============================================================================
# Quest3-Teleop-in-Isaac-Gym Quick Installation Script
# =============================================================================

set -e  # Exit on any error

# Configuration
ENV_NAME="teleop"
PYTHON_VERSION="3.8"

echo "Installing Quest3-Teleop-in-Isaac-Gym..."

# Create conda environment
echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME


# Install Isaac Gym
echo "Installing Isaac Gym..."
cd issac_gym_python
pip install -e .
cd ../

# Step 5: Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

