
#!/usr/bin/env python3
"""
Simple SVM baseline for NPZ spike-like features with label 0/1.

Each .npz file = 1 sample, with:
  - feature array under --x-key (default: "x"), shape (C, T) e.g., (32, 100) or (8, 1000)
  - label scalar 0/1 under --label-key (default: "label")

By default this script accepts ANY 2D (C, T) feature and flattens to C*T features.
Optionally enforce a specific shape via --channels and/or --time with --strict-shape.

Usage (example for 32x100):
  python svm_npz_dir_baseline.py --npz-dir ./data_npz \
      --x-key x --label-key label \
      --channels 32 --time 100 --strict-shape \
      --report ./svm_npz_report

Outputs:
  - metrics.json, classification_report.txt
  - roc_curve.png, pr_curve.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             RocCurveDisplay, PrecisionRecallDisplay)

import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Simple SVM on NPZ (C,T) w/ label 0/1")
    p.add_argument("--npz-dir", type=str, required=True, help="Directory containing .npz files")
    p.add_argument("--x-key", type=str, default="x", help="Key for feature array (default: x)")
    p.add_argument("--label-key", type=str, default="label", help="Key for label (default: label)")
    p.add_argument("--channels", type=int, default=None, help="Expected number of channels C (optional)")
    p.add_argument("--time", type=int, default=None, help="Expected time length T (optional)")
    p.add_argument("--strict-shape", action="store_true",
                   help="If set, enforce exact shape (channels x time) and skip mismatches")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split size (default: 0.2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--C", type=float, default=1.0, help="LinearSVC C parameter (default 1.0)")
    p.add_argument("--report", type=str, default="./svm_npz_report", help="Output directory for artifacts")
    return p.parse_args()

def load_npz_folder(folder, x_key="x", label_key="label", C=None, T=None, strict=False):
    folder = Path(folder)
    files = sorted([p for p in folder.glob("*.npz")])
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")
    X_list, y_list = [], []
    for f in files:
        try:
            data = np.load(f, allow_pickle=False)
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: cannot read ({e})")
            continue
        if x_key not in data or label_key not in data:
            print(f"[WARN] Skipping {f.name}: keys not found (need {x_key}, {label_key}); has {list(data.keys())}")
            continue
        x = data[x_key]
        y = data[label_key]
        # Accept label as scalar or 1-element array
        if isinstance(y, np.ndarray):
            if y.size != 1:
                print(f"[WARN] Skipping {f.name}: label must be scalar or size-1, got shape {y.shape}")
                continue
            y = int(y.reshape(-1)[0])
        else:
            y = int(y)
        if x.ndim != 2:
            print(f"[WARN] Skipping {f.name}: expected 2D array (C,T), got ndim={x.ndim} shape={x.shape}")
            continue
        Cx, Tx = x.shape
        if strict:
            if (C is not None and Cx != C) or (T is not None and Tx != T):
                print(f"[WARN] Skipping {f.name}: expected shape ({C},{T}), got ({Cx},{Tx})")
                continue
        if y not in (0, 1):
            print(f"[WARN] Skipping {f.name}: label must be 0/1, got {y}")
            continue
        X_list.append(x.astype(np.float32).reshape(-1))  # flatten to (C*T,)
        y_list.append(y)
    if not X_list:
        raise RuntimeError("No valid samples loaded. Please check keys and shapes.")
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    return X, y

def to_jsonable_metrics(metrics):
    def py(v):
        if isinstance(v, (np.floating, np.float32, np.float64)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    out = {}
    for k, v in metrics.items():
        out[str(k)] = py(v)
    return out

def main():
    args = parse_args()
    out_dir = Path(args.report)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data (each NPZ = 1 sample)
    X, y = load_npz_folder(args.npz_dir, args.x_key, args.label_key,
                           C=args.channels, T=args.time, strict=args.strict_shape)
    print(f"Loaded {X.shape[0]} samples, feature dim={X.shape[1]} (flattened CxT)")
    uniques, counts_arr = np.unique(y, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(uniques, counts_arr)}
    print(f"Class counts: {class_counts}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # Simple linear SVM pipeline
    pipe = Pipeline([
        ("scale", MaxAbsScaler()),  # good for binary/sparse-like features
        ("clf", LinearSVC(C=args.C, class_weight="balanced", dual="auto", random_state=args.seed))
    ])

    pipe.fit(X_train, y_train)

    # Evaluate using decision function
    scores = pipe.decision_function(X_test)
    y_pred = (scores >= 0).astype(int)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).astype(int).tolist()

    # AUCs
    roc_auc = None
    pr_auc = None
    try:
        roc_auc = roc_auc_score(y_test, scores)
    except Exception:
        pass
    try:
        pr_auc = average_precision_score(y_test, scores)
    except Exception:
        pass

    # Save metrics (ensure JSON serializable types)
    metrics = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_counts": class_counts,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "pr_auc": None if pr_auc is None else float(pr_auc),
        "confusion_matrix": cm,
        "C": float(args.C)
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable_metrics(metrics), f, indent=2)

    # Classification report
    cls_report = classification_report(y_test, y_pred, digits=3)
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(cls_report)

    # Curves
    try:
        RocCurveDisplay.from_predictions(y_test, scores)
        plt.title("ROC Curve (LinearSVC)")
        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png", dpi=150)
        plt.close()
    except Exception:
        pass

    try:
        PrecisionRecallDisplay.from_predictions(y_test, scores)
        plt.title("Precision-Recall Curve (LinearSVC)")
        plt.tight_layout()
        plt.savefig(out_dir / "pr_curve.png", dpi=150)
        plt.close()
    except Exception:
        pass

    print("=== Simple SVM Results ===")
    print(f"Accuracy: {acc:.3f} | F1: {f1:.3f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.3f}")
    if pr_auc is not None:
        print(f"PR AUC: {pr_auc:.3f}")
    print("\nClassification report:\n")
    print(cls_report)
    print(f"Artifacts saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
