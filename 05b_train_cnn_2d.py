# 05b_train_cnn_2d.py
# Train a small 2D CNN on log-Mel spectrograms (.npy) listed in features_index.split.csv

import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser(description="2D CNN on log-Mel npy features")
    ap.add_argument("--index", default="data/meta/features_index.split.csv")
    ap.add_argument("--outdir", default="outputs_2d")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--augment", action="store_true",
                    help="enable light time/freq masking and jitter")
    return ap.parse_args()

# ---------- small utils ----------
def compute_global_mean_std(paths):
    tot = 0.0
    totsq = 0.0
    n = 0
    for p in paths:
        X = np.load(p).astype(np.float32)  # [n_mels, frames]
        tot += X.sum()
        totsq += (X * X).sum()
        n += X.size
    mean = tot / n
    var = max(1e-8, (totsq / n) - mean * mean)
    std = np.sqrt(var)
    return float(mean), float(std)

def pad_or_trim(X, target_frames):
    # X: [n_mels, T]
    T = X.shape[1]
    if T == target_frames:
        return X
    if T > target_frames:
        return X[:, :target_frames]
    # pad using edge value
    pad = np.repeat(X[:, -1:], target_frames - T, axis=1)
    return np.concatenate([X, pad], axis=1)

def numpy_augment(X):
    # very light augment: random time shift +/-5 frames, optional mask, tiny noise
    T = X.shape[1]
    if T > 10:
        shift = np.random.randint(-5, 6)
        if shift != 0:
            X = np.roll(X, shift, axis=1)
    if np.random.rand() < 0.30:  # time mask
        w = np.random.randint(5, min(20, T//4)+1)
        s = np.random.randint(0, max(1, T - w))
        X[:, s:s+w] = X.mean()
    if np.random.rand() < 0.30:  # freq mask
        F = X.shape[0]
        w = np.random.randint(3, min(10, F//4)+1)
        s = np.random.randint(0, max(1, F - w))
        X[s:s+w, :] = X.mean()
    if np.random.rand() < 0.30:  # tiny jitter
        X = X + 0.01 * X.std() * np.random.randn(*X.shape).astype(np.float32)
    return X

def make_dataset(paths, y_idx, mean, std, target_frames, batch, training, augment):
    def gen():
        for p, yi in zip(paths, y_idx):
            X = np.load(p).astype(np.float32)                         # [F, T]
            X = pad_or_trim(X, target_frames)
            if augment and training:
                X = numpy_augment(X)
            X = (X - mean) / (std + 1e-8)
            X = X[..., np.newaxis]                                    # [F, T, 1]
            yield X, yi

    F = np.load(paths[0]).shape[0]
    sig_X = tf.TensorSpec(shape=(F, target_frames, 1), dtype=tf.float32)
    sig_y = tf.TensorSpec(shape=(), dtype=tf.int32)
    ds = tf.data.Dataset.from_generator(gen, output_signature=(sig_X, sig_y))
    if training:
        ds = ds.shuffle(4096)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(input_shape, n_classes, lr=3e-4):
    inputs = keras.Input(shape=input_shape)  # (F, T, 1)

    x = inputs
    for c in [32, 64, 128]:
        x = keras.layers.Conv2D(c, (3,3), padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
        x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv2D(192, (3,3), padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

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
    ax.plot([0,1], [ap_micro, ap_micro], linestyle="--", label=f"micro AP={ap_micro:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR Curves")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

def main():
    args = parse_args()
    outdir = Path(args.outdir); (outdir/"figs").mkdir(parents=True, exist_ok=True)
    (outdir/"models").mkdir(parents=True, exist_ok=True); (outdir/"reports").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index)
    df = df[df["feat2d_path"].notna()].copy()
    # 只保留存在的文件
    df = df[df["feat2d_path"].apply(lambda p: Path(p).exists())]

    # 标签编码（全局一致）
    le = LabelEncoder()
    le.fit(df["label"].astype(str).values)
    classes = list(le.classes_)

    # 拆分路径与标签
    paths_tr = df[df["split"]=="train"]["feat2d_path"].tolist()
    paths_va = df[df["split"]=="val"]["feat2d_path"].tolist()
    paths_te = df[df["split"]=="test"]["feat2d_path"].tolist()
    ytr = le.transform(df[df["split"]=="train"]["label"].astype(str).values)
    yva = le.transform(df[df["split"]=="val"]["label"].astype(str).values)
    yte = le.transform(df[df["split"]=="test"]["label"].astype(str).values)

    if len(paths_tr)==0 or len(paths_va)==0 or len(paths_te)==0:
        raise SystemExit("Empty split! Check features_index.split.csv.")

    # 统一帧长
    F, T = np.load(paths_tr[0]).shape
    target_frames = T  # 你在03里一般固定过，这里按训练集第一个的 T 来

    # 计算训练集的全局 mean/std 做标准化（避免泄漏）
    mean, std = compute_global_mean_std(paths_tr)
    json.dump({"mean":mean,"std":std,"n_mels":F,"frames":target_frames,"classes":classes},
              open(outdir/"models"/"norm_stats.json","w"), indent=2)

    # 类别权重（缓解不平衡）
    counts = np.bincount(ytr, minlength=len(classes)).astype(np.float32)
    weights = (counts.sum() / np.maximum(1.0, counts) / len(classes)).tolist()
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    # tf.data
    ds_tr = make_dataset(paths_tr, ytr, mean, std, target_frames, args.batch, True, args.augment)
    ds_va = make_dataset(paths_va, yva, mean, std, target_frames, args.batch, False, False)
    ds_te = make_dataset(paths_te, yte, mean, std, target_frames, args.batch, False, False)

    # 模型
    model = build_model(input_shape=(F, target_frames, 1), n_classes=len(classes), lr=args.lr)
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(outdir/"models"/"cnn2d_best.keras"),
                                        monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs,
                        class_weight=class_weight, callbacks=cbs, verbose=1)

    # 测试评估
    yprob = model.predict(ds_te, verbose=0)
    ypred = yprob.argmax(axis=1)

    report = classification_report(yte, ypred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, ypred)

    # 保存报告与图
    with open(outdir/"reports"/"cnn2d_metrics.json","w") as f:
        json.dump({"classes":classes, "report":report}, f, indent=2)

    plot_conf_mat(cm, classes, outdir/"figs"/"cm_2d.png")
    y_true_bin = label_binarize(yte, classes=np.arange(len(classes)))
    plot_pr_curves(y_true_bin, yprob, classes, outdir/"figs"/"pr_2d.png")

    print("[OK] saved best model ->", outdir/"models"/"cnn2d_best.keras")
    print("[OK] metrics/json     ->", outdir/"reports"/"cnn2d_metrics.json")
    print("[OK] figs             ->", outdir/"figs")

if __name__ == "__main__":
    main()