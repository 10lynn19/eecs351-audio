# 05a_train_logreg_1d.py
# Train a multinomial Logistic Regression on 1D aggregated features.
# Uses leak-safe splits from data/meta/features_index.split.csv

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, average_precision_score,
                             roc_auc_score)
import matplotlib.pyplot as plt
from joblib import dump

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/meta/features_index.split.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--Cs", type=float, nargs="+", default=[0.1, 1.0, 3.0])
    return ap.parse_args()

def load_split(df, split):
    sub = df[df["split"]==split].copy()
    sub = sub[sub["feat1d_path"].notna()]
    X = [np.load(p, allow_pickle=False) for p in sub["feat1d_path"]]
    X = np.stack(X).astype(np.float32)
    y = sub["label"].astype(str).values
    return X, y, sub

def plot_conf_mat(cm, labels, path):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

def plot_pr_curves(y_true_bin, y_prob, classes, path):
    fig, ax = plt.subplots(figsize=(6,5))
    ap_micro = average_precision_score(y_true_bin, y_prob, average="micro")
    for i, cls in enumerate(classes):
        p, r, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(r, p, label=f"{cls} (AP={ap:.2f})")
    ax.plot([0,1],[ap_micro,ap_micro], linestyle="--", label=f"micro AP={ap_micro:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR Curves")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

def main():
    args = parse_args()
    outdir = Path(args.outdir); (outdir / "figs").mkdir(parents=True, exist_ok=True)
    (outdir / "models").mkdir(parents=True, exist_ok=True); (outdir / "reports").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index)
    # 加载各 split
    Xtr, ytr, _ = load_split(df, "train")
    Xva, yva, _ = load_split(df, "val")
    Xte, yte, test_rows = load_split(df, "test")

    # 标签编码（全局一致）
    le = LabelEncoder()
    all_labels = np.unique(np.concatenate([ytr, yva, yte], axis=0))
    le.fit(all_labels)
    ytr_i, yva_i, yte_i = le.transform(ytr), le.transform(yva), le.transform(yte)
    classes = list(le.classes_)

    # 标准化：仅用 train 拟合，避免泄漏
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    # 简单超参搜索（按 val macro-F1 选 C）
    best = {"C": None, "f1": -1, "model": None}
    for C in args.Cs:
        clf = LogisticRegression(
            penalty="l2", C=C, solver="saga", max_iter=2000,
            multi_class="multinomial", class_weight="balanced",
            random_state=args.seed, n_jobs=-1
        )
        clf.fit(Xtr_s, ytr_i)
        yva_pred = clf.predict(Xva_s)
        f1 = f1_score(yva_i, yva_pred, average="macro")
        if f1 > best["f1"]:
            best = {"C": C, "f1": f1, "model": clf}

    model = best["model"]

    # Test 集评估
    yte_pred = model.predict(Xte_s)
    yte_prob = model.predict_proba(Xte_s)
    report = classification_report(yte_i, yte_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte_i, yte_pred)

    # PR / ROC（PR 更适合不平衡）
    y_true_bin = label_binarize(yte_i, classes=np.arange(len(classes)))
    pr_path = outdir / "figs" / "pr_1d.png"
    plot_pr_curves(y_true_bin, yte_prob, classes, pr_path)

    # 混淆矩阵图
    cm_path = outdir / "figs" / "cm_1d.png"
    plot_conf_mat(cm, classes, cm_path)

    # 保存模型与报告
    dump(model, outdir / "models" / "baseline1d.joblib")
    np.savez(outdir / "models" / "agg_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    with open(outdir / "reports" / "baseline1d_metrics.json", "w") as f:
        json.dump({
            "val_macro_f1": best["f1"],
            "C": best["C"],
            "test_report": report,
            "classes": classes
        }, f, indent=2)

    print(f"[OK] best C={best['C']}  val macro-F1={best['f1']:.3f}")
    print(f"[OK] saved model -> {outdir/'models'/'baseline1d.joblib'}")
    print(f"[OK] PR curves   -> {pr_path}")
    print(f"[OK] Confusion   -> {cm_path}")
    print(f"[OK] metrics     -> {outdir/'reports'/'baseline1d_metrics.json'}")

if __name__ == "__main__":
    main()