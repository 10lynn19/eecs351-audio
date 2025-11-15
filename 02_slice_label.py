# -*- coding: utf-8 -*-
# src/02_slice_and_label.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

def _norm_str(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan","none","null"):
        return None
    return s

def _safe_label(row, prefer="target_class"):
    t = _norm_str(row.get(prefer, None))
    s = _norm_str(row.get("source_class", None))
    return t or s or "other"

def _pick_audio_path(row):
    # prefer processed audios
    for k in ["proc_norm","proc_orig","filepath"]:
        p = _norm_str(row.get(k, None))
        if p and Path(p).exists():
            return p
    return None

def main():
    ap = argparse.ArgumentParser(description="Slide-window slicing and write slices.csv with labels")
    ap.add_argument("--meta",
        default="data/meta/data_dict_processed_labeled.csv",
        help="Input meta (prefer *labeled*). Falls back to data_dict_processed.csv if missing")
    ap.add_argument("--fallback-meta", default="data/meta/data_dict_processed.csv",
        help="Fallback meta if labeled not found")
    ap.add_argument("--out-csv", default="data/meta/slices.csv", help="Output slices index")
    ap.add_argument("--out-dir", default="data/processed/slices", help="Output wav dir for slices")
    ap.add_argument("--sr", type=int, default=16000, help="Resample rate for slices")
    ap.add_argument("--win-sec", type=float, default=1.0, help="Slice window length (seconds)")
    ap.add_argument("--hop-sec", type=float, default=0.5, help="Slice hop length (seconds)")
    ap.add_argument("--min-dur", type=float, default=0.3, help="Minimum tail duration to keep as final slice")
    ap.add_argument("--label-col", default="target_class", help="Preferred label column")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N rows (0 = all)")
    ap.add_argument("--verbose", action="store_true", help="Print progress")
    args = ap.parse_args()

    meta_p = Path(args.meta)
    if not meta_p.exists():
        meta_p = Path(args.fallback_meta)
        print(f"[WARN] {args.meta} not found; using fallback {meta_p}")
    assert meta_p.exists(), f"Missing meta: {meta_p}"

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_p)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    rows = []
    N = len(df)
    win = None  # compute after knowing sr
    hop = None

    for i, r in df.iterrows():
        apath = _pick_audio_path(r)
        if not apath:
            if args.verbose:
                print(f"[SKIP] No audio path for row {i}")
            continue

        try:
            y, _ = librosa.load(apath, sr=args.sr, mono=True)
        except Exception as e:
            if args.verbose:
                print(f"[SKIP] read fail: {apath} ({e})")
            continue

        n = y.shape[0]
        if win is None:
            win = int(round(args.win_sec * args.sr))
            hop = int(round(args.hop_sec * args.sr))
        if n < max(int(args.min_dur*args.sr), 1):
            if args.verbose:
                print(f"[SKIP] too short: {apath}")
            continue

        # Slide windows
        starts = list(range(0, max(n - win, 0) + 1, hop))
        if len(starts) == 0:
            starts = [0]
        if starts[-1] + win < n and (n - (starts[-1] + win)) >= int(args.min_dur*args.sr):
            starts.append(n - win)  # keep tail if long enough

        # meta fields
        lab = _safe_label(r, prefer=args.label_col)
        src = _norm_str(r.get("source", None)) or ""
        split = r.get("split_hint", "")
        try:
            split = int(split)
        except Exception:
            pass

        base = Path(apath).with_suffix("").name
        for st in starts:
            ed = min(st + win, n)
            seg = y[st:ed]
            # zero-pad tail shorter than win for uniform length
            if seg.shape[0] < win:
                seg = np.pad(seg, (0, win - seg.shape[0]), mode="constant")

            name = f"{base}_{int(1000*st/args.sr)}ms_{int(1000*ed/args.sr)}ms.wav"
            out_p = out_dir / name
            sf.write(out_p, seg, args.sr, subtype="PCM_16")

            rows.append({
                "slice_path": str(out_p),
                "label": lab,
                "sr": int(args.sr),
                "start_sec": round(st / args.sr, 3),
                "end_sec": round(ed / args.sr, 3),
                "source": src,
                "split_hint": split
            })

        if args.verbose and (i+1) % 200 == 0:
            print(f"[INFO] sliced {i+1}/{N}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}  (slices={len(rows)})")

if __name__ == "__main__":
    main()