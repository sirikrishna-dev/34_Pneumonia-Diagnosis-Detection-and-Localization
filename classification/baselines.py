"""
Step 2: Baselines - Logistic Regression, SVM, Random Forest on flattened pixels.

Usage:
    python baselines.py --data data/rsna_128.npz --size 64

--size downsamples the images before flattening so the SVM finishes in minutes.
Use --size 128 to match the paper exactly (16,384 features; SVM will be slow).

Output: baseline_results.json  (train/test accuracy for each classifier)
"""
import argparse
import json
import time

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def prep(X, mean, std, size):
    X = (X.astype(np.float32) - mean) / std
    if size != X.shape[1]:
        X = np.stack([np.asarray(Image.fromarray(x).resize((size, size), Image.BILINEAR)) for x in X])
    return X.reshape(len(X), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rsna_128.npz")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", default="baseline_results.json")
    args = ap.parse_args()

    d = np.load(args.data)
    mean, std = d["mean"], d["std"]
    Xtr, ytr = prep(d["X_train"], mean, std, args.size), d["y_train"]
    Xte, yte = prep(d["X_test"], mean, std, args.size), d["y_test"]
    print("Feature vector length:", Xtr.shape[1])

    models = {
        "LR": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="rbf"),
        "RF": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    }
    results = {}
    for name, clf in models.items():
        t = time.time()
        clf.fit(Xtr, ytr)
        results[name] = {
            "train_acc": float(clf.score(Xtr, ytr)),
            "test_acc": float(clf.score(Xte, yte)),
        }
        print(f"{name}: {results[name]}  ({time.time() - t:.0f}s)")

    json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
