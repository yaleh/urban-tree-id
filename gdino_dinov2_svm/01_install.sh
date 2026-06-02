#!/bin/bash
set -e

pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers accelerate
pip install -q scikit-learn matplotlib seaborn
pip install -q kaggle tqdm pillow joblib

echo "Done."
