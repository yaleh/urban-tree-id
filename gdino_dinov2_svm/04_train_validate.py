"""
04_train_validate.py

Load DINOv2 embeddings from 03_extract_embeddings.py,
train an RBF-SVM, evaluate on val and test splits,
and write a report + confusion matrix.

Output: results/report.txt, results/confusion_val.png, results/confusion_test.png,
        results/svm_model.joblib
"""

import os
import time
import logging
import argparse
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["labels"].astype(str)


def train_svm(X, y):
    species, counts = np.unique(y, return_counts=True)
    log.info(f"Training RBF-SVM on {len(X)} samples, {len(species)} classes")
    for sp, cnt in zip(species, counts):
        if cnt < 5:
            log.warning(f"  Low sample count: {sp} ({cnt})")
    clf = SVC(kernel="rbf", gamma="scale", class_weight="balanced", probability=False)
    t0 = time.time()
    clf.fit(X, y)
    log.info(f"  Train acc: {clf.score(X, y):.4f}  ({time.time()-t0:.1f}s)")
    return clf


def evaluate(clf, X, y_true, name):
    y_pred = clf.predict(X)
    acc  = accuracy_score(y_true, y_pred)
    wf1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mf1  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    rep  = classification_report(y_true, y_pred, zero_division=0)
    log.info(f"  [{name}] acc={acc:.4f}  weighted-F1={wf1:.4f}  macro-F1={mf1:.4f}")
    return {"acc": acc, "wf1": wf1, "mf1": mf1, "report": rep,
            "y_true": y_true, "y_pred": y_pred}


def plot_confusion(y_true, y_pred, title, out_path):
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(16, 14))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.3, ax=ax)
    except ImportError:
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=5)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=12)
    plt.xticks(fontsize=6, rotation=45, ha="right")
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Confusion matrix → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir",  default="embeddings")
    parser.add_argument("--out-dir",  default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    emb_dir = Path(args.emb_dir)

    log.info("Loading embeddings...")
    X_tr, y_tr = load_npz(emb_dir / "train.npz")
    X_va, y_va = load_npz(emb_dir / "val.npz")
    X_te, y_te = load_npz(emb_dir / "test.npz")
    log.info(f"  train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}  dim={X_tr.shape[1]}")

    clf = train_svm(X_tr, y_tr)

    log.info("Evaluating...")
    val_res  = evaluate(clf, X_va, y_va,  "val")
    test_res = evaluate(clf, X_te, y_te,  "test")

    plot_confusion(val_res["y_true"],  val_res["y_pred"],
                   "Validation Confusion Matrix",
                   f"{args.out_dir}/confusion_val.png")
    plot_confusion(test_res["y_true"], test_res["y_pred"],
                   "Test Confusion Matrix",
                   f"{args.out_dir}/confusion_test.png")

    report_path = f"{args.out_dir}/report.txt"
    with open(report_path, "w") as f:
        f.write("=== GDino bbox crop → DINOv2 ViT-S/14 → RBF-SVM ===\n\n")
        f.write(f"Val  acc={val_res['acc']:.4f}  "
                f"weighted-F1={val_res['wf1']:.4f}  "
                f"macro-F1={val_res['mf1']:.4f}\n")
        f.write(f"Test acc={test_res['acc']:.4f}  "
                f"weighted-F1={test_res['wf1']:.4f}  "
                f"macro-F1={test_res['mf1']:.4f}\n\n")
        f.write("--- Val Classification Report ---\n")
        f.write(val_res["report"] + "\n")
        f.write("--- Test Classification Report ---\n")
        f.write(test_res["report"] + "\n")

    log.info(f"Report → {report_path}")

    model_path = f"{args.out_dir}/svm_model.joblib"
    joblib.dump(clf, model_path)
    log.info(f"SVM model → {model_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()
