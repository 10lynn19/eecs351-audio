# 06a_predict_probs_2d.py
# Batch predict probabilities on val/test and save to NPZ files.
# Defaults now point to outputs_2d/models/norm_stats.json.

import os, json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf


def load_norm_stats(p: Path):
    if not Path(p).exists():
        raise SystemExit(f"[ERR] norm_stats not found: {p}\n"
                         f"       Expected JSON with keys: mean, std, n_mels, frames, classes")
    with open(p, "r") as f:
        return json.load(f)


def iter_batches(df: pd.DataFrame, classes, norm, batch: int = 128):
    """Yield (X, y, metas) in batches. X normalized and padded/trimmed to (n_mels, frames, 1)."""
    mean, std = float(norm["mean"]), float(norm["std"])
    H, W = int(norm["n_mels"]), int(norm["frames"])
    lab2idx = {lab: i for i, lab in enumerate(classes)}

    Xs, Ys, metas = [], [], []
    for _, r in df.iterrows():
        path = r["feat2d_path"]
        if not Path(path).exists():
            # skip missing feature
            continue
        x = np.load(path).astype(np.float32)  # (H?, W?)
        # enforce H
        if x.shape[0] != H:
            # if mel bins mismatch, try to crop or pad by edge
            if x.shape[0] > H:
                x = x[:H, :]
            else:
                pad_h = np.repeat(x[-1:, :], H - x.shape[0], axis=0)
                x = np.concatenate([x, pad_h], axis=0)
        # enforce W
        if x.shape[1] < W:
            pad = np.repeat(x[:, -1:], W - x.shape[1], axis=1)
            x = np.concatenate([x, pad], axis=1)
        elif x.shape[1] > W:
            x = x[:, :W]

        # normalize with train stats
        x = (x - mean) / (std + 1e-8)
        x = x[..., np.newaxis]  # (H, W, 1)

        Xs.append(x)
        Ys.append(lab2idx.get(str(r["label"]), -1))
        metas.append(path)

        if len(Xs) == batch:
            yield np.stack(Xs, 0), np.array(Ys), metas
            Xs, Ys, metas = [], [], []

    if Xs:
        yield np.stack(Xs, 0), np.array(Ys), metas


def run(index_csv="data/meta/features_index.split.csv",
        norm_stats="outputs_2d/models/norm_stats.json",
        model_path="outputs_2d/models/cnn2d_best.keras",
        out_dir="outputs_2d/eval"):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load metadata/index
    index_csv = Path(index_csv)
    if not index_csv.exists():
        raise SystemExit(f"[ERR] index CSV not found: {index_csv}")
    df = pd.read_csv(index_csv)

    # load norm stats and classes
    norm = load_norm_stats(Path(norm_stats))
    classes = list(norm["classes"])
    if not classes:
        raise SystemExit("[ERR] 'classes' is empty in norm_stats.json")

    # load model
    model_path = Path(model_path)
    if not model_path.exists():
        raise SystemExit(f"[ERR] model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # process val/test
    for split in ["val", "test"]:
        sub = df[df["split"] == split].copy()
        if sub.empty:
            print(f"[WARN] no rows for split={split}, skip")
            continue

        P_list, Y_list, meta_list = [], [], []
        total = 0
        for X, Y, metas in iter_batches(sub, classes, norm, batch=128):
            total += len(Y)
            P = model.predict(X, verbose=0)  # (B, C)
            P_list.append(P)
            Y_list.append(Y)
            meta_list += metas

        if not P_list:
            print(f"[WARN] no feature files found for split={split}, skip")
            continue

        P = np.concatenate(P_list, 0)
        Y = np.concatenate(Y_list, 0)

        np.savez_compressed(out_dir / f"{split}_probs.npz",
                            probs=P,
                            y_true=Y,
                            meta=np.array(meta_list, dtype=object),
                            classes=np.array(classes, dtype=object))
        print(f"[OK] saved {split}: probs={P.shape}, rows={total} -> {out_dir}/{split}_probs.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict per-class probabilities on val/test and save NPZ.")
    parser.add_argument("--index", default="data/meta/features_index.split.csv",
                        help="features_index with split/label/feat2d_path")
    parser.add_argument("--norm-stats", default="outputs_2d/models/norm_stats.json",
                        help="path to norm_stats.json (mean/std/n_mels/frames/classes)")
    parser.add_argument("--model", default="outputs_2d/models/cnn2d_best.keras",
                        help="path to trained keras model")
    parser.add_argument("--outdir", default="outputs_2d/eval",
                        help="output dir for NPZ files")
    args = parser.parse_args()

    run(index_csv=args.index,
        norm_stats=args.norm_stats,
        model_path=args.model,
        out_dir=args.outdir)