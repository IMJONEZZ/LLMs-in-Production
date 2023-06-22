#!/bin/bash

# Check if black and ruff are installed
if ! command -v black &>/dev/null; then
  echo "'black' is not installed. Please install black with 'pip install black'."
  exit 1
fi

if ! command -v ruff &>/dev/null; then
  echo "'ruff' is not installed. Please install ruff with 'pip install ruff'"
  exit 1
fi

# Set your source code folder
src_folder="chapters"

# Run black for formatting
echo "Running black on the $src_folder:"
black $src_folder

# Run ruff for linting
echo "Running ruff on the $src_folder:"
ruff check --fix $src_folder
