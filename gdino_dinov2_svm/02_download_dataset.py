"""
02_download_dataset.py

Download the Urban Street Tree dataset from Kaggle and unzip it.
Requires ~/.kaggle/kaggle.json with valid credentials.

Output: dataset/tree/{train,val,test}/{SpeciesName}/*.jpg
"""

import os
import zipfile
from pathlib import Path

DATASET_SLUG = "erickendric/tree-dataset-of-urban-street-classification-tree"
DEST_DIR = "dataset"

kaggle_cfg = os.path.expanduser("~/.kaggle/kaggle.json")
assert os.path.exists(kaggle_cfg), (
    "Kaggle credentials not found at ~/.kaggle/kaggle.json\n"
    "Create one at https://www.kaggle.com/settings/account → API → Create New Token"
)
os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle")

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

os.makedirs(DEST_DIR, exist_ok=True)
zip_path = Path(DEST_DIR) / "dataset.zip"

if not zip_path.exists():
    print(f"Downloading {DATASET_SLUG}...")
    api.dataset_download_files(DATASET_SLUG, path=DEST_DIR, force=False, quiet=False)
    downloaded = list(Path(DEST_DIR).glob("*.zip"))
    assert downloaded, f"No zip found in {DEST_DIR} after download"
    zip_path = downloaded[0]
else:
    print(f"Zip already exists: {zip_path}")

extracted_marker = Path(DEST_DIR) / "tree"
if not extracted_marker.exists():
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DEST_DIR)
    print("Extracted.")
else:
    print("Already extracted.")

# Quick summary
for split in ("train", "val", "test"):
    split_dir = Path(DEST_DIR) / "tree" / split
    if split_dir.exists():
        n_classes = sum(1 for p in split_dir.iterdir() if p.is_dir())
        n_images = sum(len(list(p.glob("*.jpg"))) for p in split_dir.iterdir() if p.is_dir())
        print(f"  {split}: {n_classes} classes, {n_images} images")

print("Done.")
