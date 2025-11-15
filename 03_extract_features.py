import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

def parse_args():
    ap = argparse.ArgumentParser(description="Extract log-Mel and aggregated features from slices")
    ap.add_argument("--slices", default="data/meta/slices.csv", help="Input slices CSV")
    ap.add_argument("--out-index", default="data/meta/features_index.csv", help="Output features index CSV")
    ap.add_argument("--featdir-2d", default="data/features/logmel_2d", help="Dir for 2D log-Mel npy")
    ap.add_argument("--featdir-1d", default="data/features/agg_1d", help="Dir for aggregated 1D features")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate for loading audio")
    ap.add_argument("--n-mels", type=int, default=64, help="Number of mel bands")
    ap.add_argument("--fmin", type=float, default=50.0, help="Mel fmin (Hz)")
    ap.add_argument("--fmax", type=float, default=8000.0, help="Mel fmax (Hz); <= sr/2")
    ap.add_argument("--win-ms", type=float, default=25.0, help="STFT window length (ms)")
    ap.add_argument("--hop-ms", type=float, default=10.0, help="STFT hop length (ms)")
    ap.add_argument("--n-fft", type=int, default=512, help="FFT size (>= win_length)")
    ap.add_argument("--fix-frames", type=int, default=100,
                    help="Pad/trim log-Mel to this frame count (None to keep variable)")
    ap.add_argument("--use-proc", choices=["auto","norm","orig","raw"], default="auto",
                    help="Pick which audio to load if slice_path is missing. auto tries slice_path first.")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N rows (0 = all)")
    ap.add_argument("--verbose", action="store_true", help="Print per-file info")
    return ap.parse_args()

def _load_audio(path, sr=16000):
    """Load mono audio, resampling to sr if needed."""
    try:
        y, file_sr = sf.read(path, always_2d=False)
        if y is None or (isinstance(y, np.ndarray) and y.size == 0):
            raise ValueError("empty audio")
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = y.mean(axis=1)
        if file_sr != sr:
            y = librosa.resample(y.astype(np.float32), orig_sr=file_sr, target_sr=sr)
        return y.astype(np.float32), sr
    except Exception:
        # fallback via librosa
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32), sr

def _pad_trim_frames(X, target_frames: int):
    """Pad/trim along time axis (axis=1) to target_frames using edge-padding."""
    if target_frames is None:
        return X
    T = X.shape[1]
    if T == target_frames:
        return X
    if T > target_frames:
        return X[:, :target_frames]
    # pad
    pad_width = target_frames - T
    if T == 0:
        # Edge case: no frames; create zeros
        return np.zeros((X.shape[0], target_frames), dtype=np.float32)
    last = X[:, -1:]
    pad = np.repeat(last, pad_width, axis=1)
    return np.concatenate([X, pad], axis=1)

def _logmel(y, sr, n_fft, win_length, hop_length, n_mels, fmin, fmax):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_mels=n_mels, fmin=fmin, fmax=min(fmax, sr/2), power=2.0, center=False
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # log scale dB
    return S_db.astype(np.float32)    

def _agg_1d_from_logmel(S_db):
    """Aggregate to 1D: [mel_mean, mel_std, rms_db, zcr_mean]."""
    # mean/std per mel band
    mean = S_db.mean(axis=1)                  # [n_mels]
    std = S_db.std(axis=1)                    # [n_mels]
    # simple extras from the dB spectrogram proxy
    rms_db = float(S_db.mean())               # overall loudness proxy
    # fake ZCR from sign changes is not ideal on spectrogram; compute on time domain later if needed
    return np.concatenate([mean, std, np.array([rms_db], dtype=np.float32)], axis=0).astype(np.float32)

def main():
    args=parse_args()
    slices_csv=Path(args.slices)
    assert slices_csv.exists(), f"Missing slices CSV: {slices_csv}"

    feat2d_dir=Path(args.featdir_2d); feat2d_dir.mkdir(parents=True,exist_ok=True)
    feat1d_dir=Path(args.featdir_1d); feat1d_dir.mkdir(parents=True,exist_ok=True)
    
    df=pd.read_csv(slices_csv)
    if args.limit and args.limit >0:
        df=df.head(args.limit).copy()
    
    win_length = int(round(args.win_ms * 1e-3 * args.sr))
    hop_length = int(round(args.hop_ms * 1e-3 * args.sr))
    n_fft = max(args.n_fft, 1 if win_length <= 1 else 1)
    if args.n_fft < win_length:
        n_fft = 1
        while n_fft < win_length:
            n_fft <<= 1  # next power-of-two >= win_length

    # Label map
    labels = sorted(set(str(l).strip() for l in df["label"].fillna("other").tolist()))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    rows = []
    # streaming stats for 1D features
    sum_vec, sumsq_vec, count_vec = None, None, 0

    for i, row in df.iterrows():
        spath = str(row.get("slice_path", "")).strip()
        if not spath:
            if args.verbose:
                print(f"[SKIP] No slice_path at row {i}")
            continue

        # load audio
        try:
            y, sr = _load_audio(spath, sr=args.sr)
        except Exception as e:
            if args.verbose:
                print(f"[SKIP] Read fail: {spath} ({e})")
            continue
        if y.size == 0:
            if args.verbose:
                print(f"[SKIP] Empty audio: {spath}")
            continue

        # log-Mel 2D
        S_db = _logmel(y, sr, n_fft, win_length, hop_length, args.n_mels, args.fmin, args.fmax)
        S_db = _pad_trim_frames(S_db, args.fix_frames)

        # aggregated 1D
        feat1d = _agg_1d_from_logmel(S_db)  # dim = n_mels*2 + 1
        if sum_vec is None:
            sum_vec = np.zeros_like(feat1d, dtype=np.float64)
            sumsq_vec = np.zeros_like(feat1d, dtype=np.float64)
        sum_vec += feat1d
        sumsq_vec += feat1d ** 2
        count_vec += 1

        # file names
        base = Path(spath).with_suffix("").name
        npy2d = feat2d_dir / f"{base}_mel{args.n_mels}_f{args.fix_frames}.npy"
        npy1d = feat1d_dir / f"{base}_agg.npy"

        # save
        np.save(npy2d, S_db.astype(np.float32))
        np.save(npy1d, feat1d.astype(np.float32))

        # record
        lab = str(row.get("label", "other")).strip() or "other"
        rows.append({
            "slice_path": spath,
            "label": lab,
            "label_idx": label_to_idx.get(lab, -1),
            "feat2d_path": str(npy2d),
            "feat1d_path": str(npy1d),
            "n_mels": args.n_mels,
            "frames": int(S_db.shape[1]),
            "sr": int(sr),
            "win_ms": args.win_ms,
            "hop_ms": args.hop_ms,
        })

        if args.verbose and (i + 1) % 500 == 0:
            print(f"[INFO] processed {i+1}/{len(df)} slices")

    # save index
    out_idx = Path(args.out_index)
    pd.DataFrame(rows).to_csv(out_idx, index=False)

    # save global mean/std for 1D features
    if count_vec > 0:
        mean = (sum_vec / count_vec).astype(np.float32)
        var = (sumsq_vec / count_vec - mean.astype(np.float64) ** 2).astype(np.float32)
        std = np.sqrt(np.clip(var, 1e-8, None)).astype(np.float32)
        stats_path = Path("data/features/agg_stats.npz")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stats_path, mean=mean, std=std)
    else:
        mean = std = None
        stats_path = None

    # save label map
    label_map_path = Path("data/meta/label_map.json")
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump({"label_to_idx": label_to_idx}, f, indent=2)

    print(f"[OK] features_index -> {out_idx}")
    print(f"[OK] logmel_2d dir  -> {feat2d_dir.resolve()}")
    print(f"[OK] agg_1d dir     -> {feat1d_dir.resolve()}")
    if stats_path:
        print(f"[OK] agg_stats      -> {stats_path.resolve()}")
    print(f"[OK] labels         -> {label_map_path.resolve()}")

if __name__ == "__main__":
    main()