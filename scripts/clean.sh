#!/bin/bash

# Check if conda is installed
if ! command -v conda &>/dev/null; then
  echo "conda not found. Please install Miniconda or Anaconda."
  exit 1
fi

echo "Removing all conda environments related to book."

# Delete the main env
conda env remove -n llmbook

# Delete chapter environments
for dir in chapters/*/; do
  chapter=$(basename $dir)
 
  conda env remove -n $chapter
done

echo "Clean up complete."