
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- CLI ----------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Make 3 progress plots from pipeline outputs.")
    ap.add_argument("--index", default="data/meta/features_index.csv",
                    help="features index CSV produced by 03_extract_features.py")
    ap.add_argument("--label-map", default="data/meta/label_map.json",
                    help="label_to_idx json (from 03)")
    ap.add_argument("--outdir", default="outputs/figs", help="where to save PNGs")
    ap.add_argument("--classes", nargs="*", default=["siren","car_horn","engine","construction","other"],
                    help="classes to visualize (order matters)")
    ap.add_argument("--examples-per-class", type=int, default=1,
                    help="num of example spectrograms per class")
    ap.add_argument("--max-per-class", type=int, default=400,
                    help="max slices per class for mean profile plot")
    ap.add_argument("--fmin", type=float, default=50.0, help="mel fmin used in 03")
    ap.add_argument("--fmax", type=float, default=8000.0, help="mel fmax used in 03")
    return ap.parse_args()

# ---------------------- helpers ----------------------
def _safe_read_npy(path: str):
    arr = np.load(path)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr

def _mel_freqs(n_mels: int, fmin: float, fmax: float):
    # avoid importing librosa just for frequencies (让脚本更轻)
    try:
        import librosa
        return librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    except Exception:
        
        return np.linspace(fmin, fmax, n_mels)

# ---------------------- plot 1 ----------------------
def plot_label_counts(df_idx: pd.DataFrame, out_png: Path):
    counts = df_idx["label"].fillna("other").value_counts().sort_values(ascending=False)
    plt.figure(figsize=(8, 4.2), dpi=160)
    counts.plot(kind="bar")
    plt.ylabel("Slices")
    plt.title("Label distribution (slices)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[OK] fig1 -> {out_png}")

# ---------------------- plot 2 ----------------------
def plot_logmel_examples(df_idx: pd.DataFrame, classes, k_per, out_png: Path):
    
    picks = []
    for c in classes:
        sub = df_idx[df_idx["label"] == c]
        if len(sub) == 0:
            continue
        picks.append(sub.head(k_per))
    if not picks:
        print("[WARN] no matching classes for examples")
        return
    pick_df = pd.concat(picks, ignore_index=True)

    rows = len(classes)
    cols = k_per
    plt.figure(figsize=(3.2*cols, 2.6*rows), dpi=180)

    for i, row in pick_df.iterrows():
        r = i // cols
        c = i % cols
        ax = plt.subplot(rows, cols, i+1)
        f2d = Path(row["feat2d_path"])
        try:
            S = _safe_read_npy(str(f2d))
            im = ax.imshow(S, origin="lower", aspect="auto")  # log-Mel dB
            ax.set_title(f"{row['label']}", fontsize=10)
            ax.set_xlabel("frames")
            ax.set_ylabel("mel bins")
        except Exception as e:
            ax.text(0.5, 0.5, f"read fail\n{f2d.name}", ha="center", va="center")
            print(f"[SKIP] {f2d} ({e})")
    plt.tight_layout(h_pad=1.0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    print(f"[OK] fig2 -> {out_png}")

# ---------------------- plot 3 ----------------------
def plot_mean_mel_profiles(df_idx: pd.DataFrame, classes, max_per_class, out_png: Path, fmin, fmax):
    
    groups = {}
    n_mels_ref = None
    for lab in classes:
        sub = df_idx[df_idx["label"] == lab]
        if len(sub) == 0:
            continue
        sub = sub.head(max_per_class)
        vecs = []
        for _, r in sub.iterrows():
            try:
                S = _safe_read_npy(str(r["feat2d_path"]))  # [n_mels, T]
                if n_mels_ref is None:
                    n_mels_ref = S.shape[0]
                if S.shape[0] != n_mels_ref:
                   
                    m = min(n_mels_ref, S.shape[0])
                    S = S[:m, :]
                vecs.append(S.mean(axis=1))  # [n_mels]
            except Exception:
                continue
        if vecs:
            V = np.stack(vecs, axis=0)  # [N, n_mels]
            groups[lab] = (V.mean(axis=0), V.std(axis=0), V.shape[0])

    if not groups:
        print("[WARN] no data for mean profiles")
        return

    
    n_mels = n_mels_ref if n_mels_ref else list(groups.values())[0][0].shape[0]
    freqs = _mel_freqs(n_mels, fmin, fmax)

    plt.figure(figsize=(8, 4.2), dpi=160)
    for lab in classes:
        if lab not in groups:
            continue
        mu, sd, n = groups[lab]
        plt.plot(freqs, mu, label=f"{lab} (n={n})")
        plt.fill_between(freqs, mu - sd, mu + sd, alpha=0.15)

    plt.xlabel("Frequency (Hz, mel bins)")
    plt.ylabel("log-Mel (dB)")
    plt.title("Mean mel profiles with ±1 std")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[OK] fig3 -> {out_png}")

# ---------------------- main ----------------------
def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index)

    df = df[df["feat2d_path"].map(lambda p: Path(str(p)).exists())].copy()
    df["label"] = df["label"].fillna("other").astype(str)

    plot_label_counts(df, outdir / "fig1_label_counts.png")
    plot_logmel_examples(df, args.classes, args.examples_per_class, outdir / "fig2_logmel_examples.png")
    plot_mean_mel_profiles(df, args.classes, args.max_per_class,
                           outdir / "fig3_mean_mel_profiles.png",
                           fmin=args.fmin, fmax=args.fmax)

if __name__ == "__main__":
    main()