"""
Extract images from CelebA-HQ parquet files and save them as PNGs. This is
based on the dataset structure from:
https://huggingface.co/datasets/korexyz/celeba-hq-256x256/tree/main/data.
"""

import pyarrow.parquet as pq
from PIL import Image
import io
import os
import pickle
from huggingface_hub import snapshot_download
import gdown

parquet_dir = "/home/scur0003/.cache/CelebAHQ/dataset/data"
out_root = os.path.expanduser("~/.cache/CelebAHQ")
img_dir = os.path.join(out_root, "images")
os.makedirs(img_dir, exist_ok=True)

# Download CelebAHQ dataset from Hugging Face
dataset_dir = os.path.join(out_root, "dataset")
if not os.path.exists(parquet_dir):
    print(f"Downloading CelebAHQ dataset to {dataset_dir}...")
    try:
        snapshot_download(
            repo_id="korexyz/celeba-hq-256x256",
            repo_type="dataset",
            local_dir=dataset_dir,
            resume_download=True
        )
        print(f"Dataset downloaded successfully to {dataset_dir}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise
else:
    print(f"Dataset already found at {dataset_dir}")

# Download CelebA attributes file
attr_file = os.path.join(out_root, "list_attr_celeba.txt")
if not os.path.exists(attr_file):
    print(f"Downloading CelebA attributes file to {attr_file}...")
    try:
        gdown.download(
            "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pblRyaVFSWGxPY0U",
            attr_file,
            quiet=False
        )
        print(f"Attributes file downloaded successfully")
    except Exception as e:
        print(f"Error downloading attributes file: {e}")
        raise
else:
    print(f"Attributes file already found at {attr_file}")

# Process parquet files and extract images
parquet_files = sorted([os.path.join(parquet_dir, f)
                        for f in os.listdir(parquet_dir) if
                        f.endswith(".parquet")])

idx_counter = 0
train_indices = []
test_indices = []

for pf in parquet_files:
    print(f"Processing {pf} ...")
    table = pq.read_table(pf)
    df = table.to_pandas()

    is_train = os.path.basename(pf).startswith("train")

    for _, row in df.iterrows():
        img_bytes = row["image"]["bytes"]
        fname = f"img{idx_counter:08}.png"
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(os.path.join(img_dir, fname), format="PNG")

        if is_train:
            train_indices.append(idx_counter)
        else:
            test_indices.append(idx_counter)

        idx_counter += 1

all_filenames = [f"img{i:08}.png" for i in range(idx_counter)]
with open(os.path.join(out_root, "image_list.txt"), "w") as f:
    f.write("idx celebA_idx split\n")
    for i, fname in enumerate(all_filenames):
        f.write(f"{i} {fname} {i}\n")

with open(os.path.join(out_root, "train_filenames.pickle"), "wb") as f:
    pickle.dump(train_indices, f)

with open(os.path.join(out_root, "test_filenames.pickle"), "wb") as f:
    pickle.dump(test_indices, f)
