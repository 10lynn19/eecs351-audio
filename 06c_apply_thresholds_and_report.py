# 06c_apply_thresholds_and_report.py
# Apply per-class thresholds to probs, produce classification report + confusion matrix.
import argparse, json, os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

def parse_args():
    ap = argparse.ArgumentParser(description="Apply per-class thresholds and evaluate")
    ap.add_argument("--thr", default="outputs_2d/eval/thresholds.json",
                    help="Path to thresholds JSON (supports {'thresholds': {...}} or flat {...})")
    ap.add_argument("--split", choices=["val","test"], default="test",
                    help="Which split to evaluate on (default: test)")
    ap.add_argument("--probs", default="", help="Override path to *_probs.npz; if empty, use outputs_2d/eval/{split}_probs.npz")
    ap.add_argument("--outdir", default="outputs_2d/eval", help="Output directory for report and CM image")
    ap.add_argument("--fallback", choices=["other","argmax"], default="other",
                    help="If no class exceeds its threshold: use 'other' if present, else 'argmax'")
    ap.add_argument("--cm-normalize", choices=["none","true","pred","all"], default="none",
                    help="Normalization for confusion matrix plot (sklearn style)")
    return ap.parse_args()

def load_thresholds(path: Path):
    with open(path, "r") as f:
        d = json.load(f)
    # support {"thresholds": {...}} or flat {...}
    thresholds = d.get("thresholds", d)
    if not isinstance(thresholds, dict):
        raise ValueError("Invalid thresholds JSON structure.")
    return thresholds

def plot_cm(cm, classes, out_path: Path, normalize="none", title="Confusion Matrix"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes, title=title, ylabel="True label", xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize != "none" else str(int(cm[i, j]))
            ax.text(j, i, txt, ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

def apply_thresholds(probs: np.ndarray, classes, thresholds: dict, fallback="other"):
    """
    probs: [N, C], classes: list[str], thresholds: dict[class_name->float]
    returns y_pred (int indices) after threshold gating + fallback.
    """
    N, C = probs.shape
    masked = probs.copy()

    # Apply per-class thresholds: set below threshold to -inf to exclude from argmax
    # (use -1.0 works too since probs are in [0,1], but -np.inf is cleaner)
    for k, cls in enumerate(classes):
        t = float(thresholds.get(cls, 0.5))
        mask = masked[:, k] < t
        masked[mask, k] = -np.inf

    # Argmax on masked probs
    y_pred = np.argmax(masked, axis=1)

    # Fallback: rows where all entries were masked become -inf; detect them
    all_masked = np.all(~np.isfinite(masked), axis=1)
    if np.any(all_masked):
        if fallback == "other" and ("other" in classes):
            other_idx = classes.index("other")
            y_pred[all_masked] = other_idx
        else:
            # fallback to original argmax without thresholds
            orig_argmax = np.argmax(probs, axis=1)
            y_pred[all_masked] = orig_argmax[all_masked]

    return y_pred

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load thresholds
    thr_path = Path(args.thr)
    thresholds = load_thresholds(thr_path)
    print("[DEBUG] thresholds file:", str(thr_path))
    print("[DEBUG] thresholds loaded:", thresholds)

    # 2) Load probs / y_true / classes
    npz_path = Path(args.probs) if args.probs else Path(f"outputs_2d/eval/{args.split}_probs.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing probs file: {npz_path}")
    D = np.load(npz_path, allow_pickle=True)
    probs = D["probs"]            # [N, C], float
    y_true = D["y_true"]          # [N], int
    classes = list(D["classes"])  # list[str]

    # 3) Apply thresholds + fallback
    y_pred = apply_thresholds(probs, classes, thresholds, fallback=args.fallback)

    # 4) Metrics
    acc = float(accuracy_score(y_true, y_pred))
    # per-class precision/recall/f1/support
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
    # macro / weighted
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # 5) Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    # Optional normalize for plotting only
    norm_flag = args.cm_normalize
    cm_for_plot = cm.astype(float)
    if norm_flag == "true":
        cm_for_plot = cm_for_plot / (cm_for_plot.sum(axis=1, keepdims=True) + 1e-12)
    elif norm_flag == "pred":
        cm_for_plot = cm_for_plot / (cm_for_plot.sum(axis=0, keepdims=True) + 1e-12)
    elif norm_flag == "all":
        cm_for_plot = cm_for_plot / (cm_for_plot.sum() + 1e-12)

    cm_png = outdir / "cm_thresholded.png"
    plot_title = f"Confusion Matrix ({args.split}, thresholds)"
    plot_cm(cm_for_plot, classes, cm_png, normalize=norm_flag, title=plot_title)

    # 6) Save JSON report
    report = {
        "split": args.split,
        "n_samples": int(len(y_true)),
        "accuracy": acc,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            cls: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": float(sup[i]),
            } for i, cls in enumerate(classes)
        },
        "thresholds": thresholds,
        "confusion_matrix_shape": [int(cm.shape[0]), int(cm.shape[1])],
    }
    out_json = outdir / f"{args.split}_thresholded_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] accuracy={acc:.6f}  macroF1={macro_f1:.6f}  weightedF1={weighted_f1:.6f}")
    print(f"[OK] saved report -> {out_json}")
    print(f"[OK] saved CM     -> {cm_png}")

if __name__ == "__main__":
    main()