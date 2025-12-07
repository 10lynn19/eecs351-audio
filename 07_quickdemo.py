# 07_quickdemo.py  -- minimal runnable demo with proper normalization
import argparse, json, os, glob, numpy as np
from pathlib import Path
import librosa, soundfile as sf
from tensorflow import keras

SR=16000; N_MELS=64; WIN_MS=25; HOP_MS=10; FIX_FRAMES=100

def load_norm_stats(p: Path):
    """Load mean/std; support scalar, [n_mels], or [n_mels*frames]."""
    if not p.exists():
        print(f"[WARN] norm stats not found: {p} (proceed w/o normalization)")
        return None, None
    d = json.load(open(p))
    mean = np.array(d.get("mean", d.get("mel_mean", [])), dtype=np.float32)
    std  = np.array(d.get("std" , d.get("mel_std" , [])), dtype=np.float32)

    def reshape_vec(v):
        if v.size == 0:
            return None
        if v.size == 1:
            return float(v.reshape(()))  # scalar
        if v.size == N_MELS:
            return v.reshape(N_MELS, 1)  # per-mel band
        if v.size == N_MELS*FIX_FRAMES:
            return v.reshape(N_MELS, FIX_FRAMES)  # full 2D
        # fallback: try per-mel
        return v.reshape(N_MELS, 1)

    return reshape_vec(mean), reshape_vec(std)

def logmel(y, sr=SR, n_mels=N_MELS, win_ms=WIN_MS, hop_ms=HOP_MS, fix_frames=FIX_FRAMES,
           norm_mean=None, norm_std=None):
    win = int(round(win_ms*1e-3*sr)); hop=int(round(hop_ms*1e-3*sr))
    n_fft=1
    while n_fft < win: n_fft <<= 1
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
                                       n_mels=n_mels, fmin=50.0, fmax=min(8000.0, sr/2), power=2.0, center=False)
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    T = S_db.shape[1]
    if T < fix_frames:
        pad = np.repeat(S_db[:, -1:], fix_frames-T, axis=1)
        S_db = np.concatenate([S_db, pad], axis=1)
    elif T > fix_frames:
        S_db = S_db[:, :fix_frames]

    # apply dataset normalization exactly like training
    if norm_mean is not None and norm_std is not None:
        eps = 1e-6
        if isinstance(norm_mean, float) and isinstance(norm_std, float):
            S_db = (S_db - norm_mean) / max(norm_std, eps)
        else:
            # broadcast to [n_mels, T]
            m = norm_mean if np.ndim(norm_mean)>0 else np.array(norm_mean, dtype=np.float32)
            s = norm_std  if np.ndim(norm_std)>0  else np.array(norm_std , dtype=np.float32)
            S_db = (S_db - m) / np.maximum(s, eps)
    return S_db  # [n_mels, T]

def load_classes():
    npz = Path("outputs_2d/eval/test_probs.npz")
    if npz.exists():
        D=np.load(npz, allow_pickle=True)
        return list(D["classes"])
    mp = Path("data/meta/label_map.json")
    d = json.load(open(mp))
    l2i = d["label_to_idx"]
    inv = sorted([(v,k) for k,v in l2i.items()])
    return [k for _,k in inv]

def apply_thresholds(probs_row, classes, thresholds, fallback_other=True):
    x = probs_row.copy()
    for k,cls in enumerate(classes):
        t=float(thresholds.get(cls,0.5))
        if x[k] < t: x[k] = -np.inf
    if np.isneginf(x).all() and fallback_other and "other" in classes:
        return classes.index("other")
    return int(np.nanargmax(x))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wavdir", default="data/demo/wavs", help="Folder of wavs for quick demo")
    ap.add_argument("--model", default="outputs_2d/models/cnn2d_best.keras")
    ap.add_argument("--norm",  default="outputs_2d/models/norm_stats.json", help="norm stats json")
    ap.add_argument("--thr",   default="outputs_2d/eval/thresholds_balanced.json")
    ap.add_argument("--topk",  type=int, default=3)
    args=ap.parse_args()

    classes = load_classes()
    thr = json.load(open(args.thr)); thr = thr.get("thresholds", thr)
    mean, std = load_norm_stats(Path(args.norm))
    model = keras.models.load_model(args.model)

    wavs = sorted(glob.glob(str(Path(args.wavdir)/"*.wav")))
    assert wavs, f"No wav found in {args.wavdir}. Run scripts/make_mini_subset.py first."

    print(f"[INFO] classes: {classes}")
    print(f"[INFO] thresholds from: {args.thr}")
    print(f"[INFO] model: {args.model}")
    print(f"[INFO] norm:  {args.norm} (found={mean is not None and std is not None})")
    print(f"[INFO] {len(wavs)} demo files")

    y_true, y_pred = [], []
    for p in wavs:
        y, sr = sf.read(p, always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim==2: y=y.mean(axis=1)
        if sr != SR: y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
        S = logmel(y, norm_mean=mean, norm_std=std)
        X = np.expand_dims(S, axis=(0,-1))   # [1, n_mels, T, 1]
        probs = model.predict(X, verbose=0)[0]  # [C]
        pred_idx = apply_thresholds(probs, classes, thr)

        topk_idx = probs.argsort()[-args.topk:][::-1]
        print(f"\n== {Path(p).name} ==")
        for k in topk_idx:
            print(f"  {classes[k]:15s}  {probs[k]:.3f}")
        print(f"--> predicted: {classes[pred_idx]}")

        # optional: parse true label from filename
        name = Path(p).name
        true = None
        for c in classes:
            tag=f"__label_{c}__"
            if tag in name: true=c
        if true:
            y_true.append(classes.index(true))
            y_pred.append(pred_idx)

    if y_true:
        from sklearn.metrics import classification_report, accuracy_score
        print("\n[DEMO REPORT]")
        print(classification_report(y_true, y_pred, target_names=classes, digits=3))
        print("acc=", accuracy_score(y_true, y_pred))

if __name__=="__main__":
    main()