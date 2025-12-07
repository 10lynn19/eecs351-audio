# scripts/make_mini_subset.py
import pandas as pd, numpy as np, shutil, os
from pathlib import Path
import argparse

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--slices", default="data/meta/slices.csv")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--outdir", default="data/demo/wavs")
    ap.add_argument("--use-split", choices=["","train","val","test"], default="test",
                    help="If set, sample from split_{split}.csv instead of raw slices.csv")
    args=ap.parse_args()

    if args.use_split:
        sp = Path(f"data/meta/split_{args.use_split}.csv")
        assert sp.exists(), f"{sp} not found; run 04_make_split.py first or use --use-split \"\""
        df = pd.read_csv(sp)
        # 需要有 slice_path/label 两列；若没有，则 join features_index.split.csv
        if "slice_path" not in df.columns:
            idx = pd.read_csv("data/meta/features_index.split.csv")
            df = df.merge(idx[["slice_path","label"]], on="slice_path", how="left")
    else:
        df = pd.read_csv(args.slices)

    assert "slice_path" in df.columns and "label" in df.columns, "slices.csv must contain slice_path and label"
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    taken=[]
    for cls, g in df.groupby("label"):
        g = g.dropna(subset=["slice_path"])
        if len(g)==0: continue
        take = g.sample(min(args.per_class, len(g)), random_state=42)
        for _,r in take.iterrows():
            src = Path(r["slice_path"])
            if not src.exists(): continue
            # 文件名加上真实标签，便于 quickdemo 自动记真值
            dst = out / f"{src.stem}__label_{cls}__.wav"
            shutil.copy2(src, dst)
            taken.append({"dst":str(dst), "label":cls})
    pd.DataFrame(taken).to_csv(out.parent/"demo_list.csv", index=False)
    print(f"[OK] wrote {len(taken)} files to {out}")

if __name__=="__main__":
    main()