#!/bin/bash

# Fix for activate issues
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# Check if conda is installed
if ! command -v conda &>/dev/null; then
  echo "conda not found. Please install Miniconda or Anaconda."
  exit 1
fi

# Check for git lfs
if ! command -v git-lfs &> /dev/null; then
  echo "please install git-lfs, e.g. 'brew install git-lfs' or visit https://git-lfs.com/ for your OS instructions"
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

# Create an environment for each chapter
for dir in chapters/*/; do
  chapter=$(basename $dir)
  echo "Setting up environment for $chapter"

  # Check for the requirements.txt file
  if [ ! -f "$dir/requirements.txt" ]; then
    echo "requirements.txt not found in $dir."
    continue
  fi

  # Check if the chapter environment already exists
  if conda info --envs | grep -q $chapter; then
    echo "Environment $chapter already exists. Skipping environment creation."
  else
    # Create the conda environment for the chapter
    echo "Creating a conda environment for $chapter"
    conda create -n $chapter python=3.10 --yes
  fi

  # Activate the conda environment
  conda activate $chapter

  # Install requirements from the requirements.txt file
  echo "Installing requirements for $chapter"
  pip install -r "$dir/requirements.txt"

  # Deactivate the conda environment after installing packages
  echo "Packages installed. Deactivating conda environment..."
  conda deactivate
done

echo -e "\n\n\n"
echo -e "You are now set up.\n\nPlease run 'conda activate chapter_1' to begin.\nWhen ready for the next chapter run 'conda deactivate'\nthen 'conda activate chapter_2' and so on."
