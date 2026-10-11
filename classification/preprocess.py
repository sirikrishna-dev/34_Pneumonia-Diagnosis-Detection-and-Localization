"""
Step 1: Build the dataset.

- Reads the RSNA Pneumonia Detection labels + DICOM images
- Keeps only "Lung Opacity" (pneumonia = 1) and "Normal" (healthy = 0)
  (drops "No Lung Opacity / Not Normal", as in the paper)
- Resizes to 128x128, stratified 70/20/10 train/val/test split
- Saves everything to one compressed .npz (uint8 images + labels + mean/std)

Usage:
    python preprocess.py --data_dir /path/to/rsna --out data/rsna_128.npz

Expected layout of --data_dir (Kaggle RSNA, stage 2 names):
    stage_2_train_labels.csv
    stage_2_detailed_class_info.csv
    stage_2_train_images/<patientId>.dcm
If your folder uses the stage_1 names, pass --prefix stage_1.
"""
import argparse
import os

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SIZE = 128


def load_image(path, size=SIZE):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    # scale to 0-255 per image, then resize
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
    img = Image.fromarray(img.astype(np.uint8)).resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="data/rsna_128.npz")
    ap.add_argument("--prefix", default="stage_2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels = pd.read_csv(os.path.join(args.data_dir, f"{args.prefix}_train_labels.csv"))
    info = pd.read_csv(os.path.join(args.data_dir, f"{args.prefix}_detailed_class_info.csv"))

    # one row per patient (labels csv has one row per bounding box)
    info = info.drop_duplicates("patientId")
    info = info[info["class"].isin(["Lung Opacity", "Normal"])].copy()
    info["label"] = (info["class"] == "Lung Opacity").astype(int)
    print("Class counts:\n", info["label"].value_counts())

    ids = info["patientId"].tolist()
    y = info["label"].values.astype(np.int64)

    img_dir = os.path.join(args.data_dir, f"{args.prefix}_train_images")
    X = np.stack([load_image(os.path.join(img_dir, f"{p}.dcm")) for p in tqdm(ids)])
    ids = np.array(ids)

    # 70 / 20 / 10 stratified split
    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=args.seed)
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=1 / 3, stratify=y[idx_tmp], random_state=args.seed
    )  # 30% * 2/3 = 20% val, 30% * 1/3 = 10% test

    # per-pixel mean/std computed on TRAIN only (paper: normalise "at that location")
    mean = X[idx_tr].astype(np.float32).mean(axis=0)
    std = X[idx_tr].astype(np.float32).std(axis=0) + 1e-6

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        X_train=X[idx_tr], y_train=y[idx_tr], id_train=ids[idx_tr],
        X_val=X[idx_val], y_val=y[idx_val], id_val=ids[idx_val],
        X_test=X[idx_te], y_test=y[idx_te], id_test=ids[idx_te],
        mean=mean, std=std,
    )
    print(f"Saved {args.out}: train={len(idx_tr)} val={len(idx_val)} test={len(idx_te)}")


if __name__ == "__main__":
    main()
