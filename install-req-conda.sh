#!/bin/bash

# Conda package installation script

# List of packages to install
PACKAGES=(
  emoji
  fuzzywuzzy
  gensim
  matplotlib
  nltk
  numpy
  pandas
  praw
  pyLDAvis
  scipy
  spacy
  tqdm
  transformers
  wordcloud
  yahoo_fin
  yfinance
  fuzzywuzzy
  python-Levenshtein
  transformers
  datasets
  tensorflow
  tf-keras
)

echo "Starting installation of Python packages using conda..."

# Loop through the packages and install each one
for PACKAGE in "${PACKAGES[@]}"; do
  echo "Installing $PACKAGE..."
  conda install "$PACKAGE" -y
done

echo "Installation of packages completed."
