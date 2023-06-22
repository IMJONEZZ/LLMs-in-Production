#!/bin/bash

# Fix for activate issues
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# Check if conda is installed
if ! command -v conda &>/dev/null; then
  echo "conda not found. Please install Miniconda or Anaconda."
  exit 1
fi

# Check for the requirements.txt file
if [ ! -f "requirements.txt" ]; then
  echo "requirements.txt not found in the current directory. Please run from project root"
  exit 1
fi

# Check if the 'llmbook' environment already exists
if conda info --envs | grep -q 'llmbook'; then
  echo "Environment 'llmbook' already exists. Skipping environment creation."
else
  # Create the conda environment called 'llmbook'
  conda create -n llmbook python=3.10 --yes
fi

# Activate the 'llmbook' conda environment
conda activate llmbook

# Install requirements from the requirements.txt file
echo "Installing requirements from requirements.txt:"
pip install -r requirements.txt

# Deactivate the conda environment after installing packages
echo "Packages installed. Deactivating conda environment..."
conda deactivate
