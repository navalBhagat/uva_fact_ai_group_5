"""
Extract images from CelebA-HQ parquet files and save them as PNGs. This is
based on the dataset structure from:
https://huggingface.co/datasets/korexyz/celeba-hq-256x256/tree/main/data.
"""

import argparse
import io
import os
import pickle
from pathlib import Path

import gdown
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path.home() / ".cache" / "CelebAHQ"),
    )
    args = parser.parse_args()

    out_root = Path(args.root).expanduser().resolve()
    dataset_dir = out_root / "dataset"
    parquet_dir = dataset_dir / "data"
    img_dir = out_root / "images"

    img_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    if not parquet_dir.exists():
        snapshot_download(
            repo_id="korexyz/celeba-hq-256x256",
            repo_type="dataset",
            local_dir=str(dataset_dir)
        )

    attr_file = out_root / "list_attr_celeba.txt"
    if not attr_file.exists():
        gdown.download(
            "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pblRyaVFSWGxPY0U",
            str(attr_file),
            quiet=False,
        )

    parquet_files = sorted(p for p in parquet_dir.iterdir()
                           if p.suffix == ".parquet")

    idx_counter = 0
    train_indices = []
    test_indices = []

    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()
        is_train = pf.name.startswith("train")

        for _, row in df.iterrows():
            img_bytes = row["image"]["bytes"]
            fname = f"img{idx_counter:08}.png"
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img.save(img_dir / fname, format="PNG")

            if is_train:
                train_indices.append(idx_counter)
            else:
                test_indices.append(idx_counter)

            idx_counter += 1

    all_filenames = [f"img{i:08}.png" for i in range(idx_counter)]
    with open(out_root / "image_list.txt", "w") as f:
        f.write("idx celebA_idx split\n")
        for i, fname in enumerate(all_filenames):
            f.write(f"{i} {fname} {i}\n")

    with open(out_root / "train_filenames.pickle", "wb") as f:
        pickle.dump(train_indices, f)

    with open(out_root / "test_filenames.pickle", "wb") as f:
        pickle.dump(test_indices, f)


if __name__ == "__main__":
    main()
