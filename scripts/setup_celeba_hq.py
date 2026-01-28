"""
Extract images from CelebA-HQ parquet files and save them as PNGs. This is
based on the dataset structure from:
https://huggingface.co/datasets/korexyz/celeba-hq-256x256/tree/main/data.
"""

import argparse
import io
import pickle
from pathlib import Path
from typing import List, Tuple

import gdown
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image


def download_dataset(dataset_dir) -> None:
    """Download CelebA-HQ dataset from HuggingFace if not already present."""
    print(f"Downloading dataset to {dataset_dir}...")
    snapshot_download(
        repo_id="korexyz/celeba-hq-256x256",
        repo_type="dataset",
        local_dir=str(dataset_dir),
    )


def download_attributes(attr_file) -> None:
    """Download CelebA attributes file from Google Drive if not already
    present."""
    print(f"Downloading attributes to {attr_file}...")
    gdown.download(
        "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pblRyaVFSWGxPY0U",
        str(attr_file),
        quiet=False,
    )


def extract_images(parquet_dir, img_dir):
    """
    Extract images from parquet files and save as PNGs.

    Args:
        parquet_dir: Directory containing parquet files
        img_dir: Directory to save extracted images

    Returns:
        Tuple of (train_indices, test_indices)
    """
    parquet_files = sorted(parquet_dir.glob("*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {parquet_dir}")

    idx_counter = 0
    train_indices = []
    test_indices = []

    for parquet_file in parquet_files:
        print(f"Processing {parquet_file.name}...")

        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        is_train = parquet_file.name.startswith("train")

        for _, row in df.iterrows():
            img_bytes = row["image"]["bytes"]
            fname = f"img{idx_counter:08d}.png"

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img.save(img_dir / fname, format="PNG")

            if is_train:
                train_indices.append(idx_counter)
            else:
                test_indices.append(idx_counter)

            idx_counter += 1

    return train_indices, test_indices


def save_metadata(out_root, train_indices, test_indices):
    """Save image list and train/test splits."""
    total_images = len(train_indices) + len(test_indices)

    # Save image list
    with open(out_root / "image_list.txt", "w") as f:
        f.write("idx celebA_idx split\n")
        for i in range(total_images):
            fname = f"img{i:08d}.png"
            f.write(f"{i} {fname} {i}\n")

    # Save train/test splits
    with open(out_root / "train_filenames.pickle", "wb") as f:
        pickle.dump(train_indices, f)

    with open(out_root / "test_filenames.pickle", "wb") as f:
        pickle.dump(test_indices, f)

    print(f"\nExtracted {total_images} images:")
    print(f"  Train: {len(train_indices)}")
    print(f"  Test: {len(test_indices)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract images from CelebA-HQ parquet files"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path.home() / ".cache" / "CelebAHQ"),
        help="Root directory for dataset cache (default: ~/.cache/CelebAHQ)",
    )
    args = parser.parse_args()

    out_root = Path(args.root).expanduser().resolve()
    dataset_dir = out_root / "dataset"
    parquet_dir = dataset_dir / "data"
    img_dir = out_root / "images"
    attr_file = out_root / "list_attr_celeba.txt"

    img_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    if not attr_file.exists():
        download_attributes(attr_file)

    if not parquet_dir.exists() or not any(parquet_dir.glob("*.parquet")):
        download_dataset(dataset_dir)

    train_indices, test_indices = extract_images(parquet_dir, img_dir)

    save_metadata(out_root, train_indices, test_indices)

    print(f"\nAll files saved to: {out_root}")
