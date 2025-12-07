# 04_make_split.py
# Leak-safe split grouped by parent recording, stratified by class.

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    ap = argparse.ArgumentParser(description="Make leak-safe train/val/test splits")
    ap.add_argument("--features-index", default="data/meta/features_index.csv")
    ap.add_argument("--slices", default="data/meta/slices.csv")
    ap.add_argument("--out-index", default="data/meta/features_index.split.csv")
    ap.add_argument("--out-dir", default="data/meta")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-per-class", type=int, default=1)
    return ap.parse_args()

_PATTERNS = [
    r'[_-]st\d+.*$', r'[_-]start\d+.*$', r'[_-]t\d+.*$',
    r'[_-]seg\d+.*$', r'[_-]slice\d+.*$', r'_\d{5,}$'
]

def robust_parent_from_name(slice_path: str) -> str:
    stem = Path(slice_path).stem
    for pat in _PATTERNS:
        stem = re.sub(pat, '', stem)
    return stem

def choose_group_key(df: pd.DataFrame) -> str:
    for c in ["parent_path", "session_id", "parent_id"]:
        if c in df.columns:
            return c
    return "_parent_id"

def ensure_min_per_class(groups_df: pd.DataFrame, want_split: str, min_per_class: int):
    cur = groups_df[groups_df["split"]==want_split]["label"].value_counts().to_dict()
    need = [c for c in groups_df["label"].unique() if cur.get(c,0) < min_per_class]
    for cls in need:
        cand = groups_df[(groups_df["split"]=="train") & (groups_df["label"]==cls)]
        if cand.empty:
            continue
        row = cand.sort_values("n_slices").iloc[0]
        groups_df.loc[groups_df["group_id"]==row["group_id"], "split"] = want_split
    return groups_df

def main():
    args = parse_args()
    feats = pd.read_csv(args.features_index)
    sl = pd.read_csv(args.slices)

    key = choose_group_key(sl)
    if key == "_parent_id":
        sl["_parent_id"] = sl["slice_path"].apply(robust_parent_from_name)

    merged = pd.merge(
        feats[["slice_path","label","feat1d_path","feat2d_path"]],
        sl[["slice_path", key]],
        on="slice_path", how="left"
    )
    if merged[key].isna().any():
        # fallback for missing
        mask = merged[key].isna()
        merged.loc[mask, key] = merged.loc[mask, "slice_path"].apply(robust_parent_from_name)

    merged = merged.rename(columns={key: "group_id"})

    # sanity: print compression ratio
    n_groups = merged["group_id"].nunique()
    n_slices = len(merged)
    print(f"[INFO] grouping key = {('parent_path/session_id/parent_id' if key!='_parent_id' else 'derived from name')}")
    print(f"[INFO] slices={n_slices} -> groups={n_groups}  (ratio={n_groups/n_slices:.3f})")

    # build group table (label per group)
    g = merged.groupby("group_id").agg(
        label=("label","first"),
        n_slices=("slice_path","count")
    ).reset_index()

    # group-level stratified split
    train_g, temp_g = train_test_split(
        g, test_size=(args.val_size+args.test_size),
        random_state=args.seed, stratify=g["label"]
    )
    rel = args.test_size/(args.val_size+args.test_size)
    val_g, test_g = train_test_split(
        temp_g, test_size=rel, random_state=args.seed,
        stratify=temp_g["label"]
    )

    groups_df = g.copy()
    groups_df["split"] = "train"
    groups_df.loc[groups_df["group_id"].isin(val_g["group_id"]), "split"] = "val"
    groups_df.loc[groups_df["group_id"].isin(test_g["group_id"]), "split"] = "test"

    groups_df = ensure_min_per_class(groups_df, "val", args.min_per_class)
    groups_df = ensure_min_per_class(groups_df, "test", args.min_per_class)

    # back to slice level
    split_map = groups_df.set_index("group_id")["split"]
    merged["split"] = merged["group_id"].map(split_map)

    # write
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    feats_split = pd.merge(feats, merged[["slice_path","split"]], on="slice_path", how="left")
    feats_split.to_csv(args.out_index, index=False)
    for sp in ["train","val","test"]:
        feats_split[feats_split["split"]==sp].to_csv(out_dir / f"split_{sp}.csv", index=False)

    print("[OK] Saved:", args.out_index, "and split_{train,val,test}.csv")
    for sp in ["train","val","test"]:
        sub = merged[merged["split"]==sp]
        print(f"  {sp}: slices={len(sub)}  groups={sub['group_id'].nunique()}")
        print("    per-class:", sub["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()