"""
Step 3: Train the CNN and evaluate it.

Usage:
    python train.py --data data/rsna_128.npz --epochs 20 --lr 1e-4

Outputs (in --out_dir, default models/):
    cnn_best.pt          best-validation checkpoint (hand this to Person B)
    history.json         per-epoch train/val loss + accuracy
    training_curves.png  accuracy curve for the write-up
    test_report.json     test accuracy, confusion matrix, sensitivity, specificity
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from model import CAMNet


def make_loader(X, y, mean, std, batch_size, shuffle):
    X = (X.astype(np.float32) - mean) / std                # per-pixel normalisation
    X = torch.from_numpy(X).unsqueeze(1)                    # (N, 1, 128, 128)
    y = torch.from_numpy(y)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    preds, gts, total_loss = [], [], 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        if criterion is not None:
            total_loss += criterion(out, yb).item() * len(yb)
        preds.append(out.argmax(1).cpu().numpy())
        gts.append(yb.cpu().numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    return (preds == gts).mean(), total_loss / len(gts), preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rsna_128.npz")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    d = np.load(args.data)
    mean, std = d["mean"], d["std"]
    train_dl = make_loader(d["X_train"], d["y_train"], mean, std, args.batch_size, True)
    val_dl = make_loader(d["X_val"], d["y_val"], mean, std, args.batch_size, False)
    test_dl = make_loader(d["X_test"], d["y_test"], mean, std, args.batch_size, False)

    model = CAMNet(in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    hist = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val = 0.0

    for ep in range(1, args.epochs + 1):
        model.train()
        correct, n, loss_sum = 0, 0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
        tr_acc, tr_loss = correct / n, loss_sum / n
        val_acc, val_loss, _, _ = evaluate(model, val_dl, device, criterion)

        for k, v in zip(hist, [tr_acc, val_acc, tr_loss, val_loss]):
            hist[k].append(float(v))
        print(f"Epoch {ep:02d}/{args.epochs} | train acc {tr_acc:.4f} loss {tr_loss:.4f} "
              f"| val acc {val_acc:.4f} loss {val_loss:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "in_channels": 1,
                    "mean": mean,
                    "std": std,
                    "val_acc": float(val_acc),
                },
                os.path.join(args.out_dir, "cnn_best.pt"),
            )

    json.dump(hist, open(os.path.join(args.out_dir, "history.json"), "w"), indent=2)

    # training curves (for the write-up / slides)
    plt.figure(figsize=(6, 4))
    plt.plot(hist["train_acc"], label="train")
    plt.plot(hist["val_acc"], label="validation")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Training and validation accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "training_curves.png"), dpi=150)

    # final evaluation with the best checkpoint
    ckpt = torch.load(os.path.join(args.out_dir, "cnn_best.pt"), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_acc, _, preds, gts = evaluate(model, test_dl, device)
    tn, fp, fn, tp = confusion_matrix(gts, preds, labels=[0, 1]).ravel()
    report = {
        "test_accuracy": float(test_acc),
        "confusion_matrix": {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)},
        "sensitivity_recall": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
    }
    json.dump(report, open(os.path.join(args.out_dir, "test_report.json"), "w"), indent=2)
    print("\nTEST:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
