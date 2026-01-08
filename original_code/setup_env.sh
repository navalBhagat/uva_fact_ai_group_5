#!/bin/bash

set -e

echo ""
echo "Installing environment from environment.yaml..."
conda env create -f environment.yaml

echo ""
echo "Navigating to ./src/autodp..."
cd ./src/autodp

echo "Installing autodp..."
pip install -e .

echo "Installing taming-transformers..."
cd ../taming-transformers
pip install -e .

echo ""