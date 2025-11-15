# -*- coding: utf-8 -*-
# src/01b_apply_class_map.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def _norm_str(v):
    if v is None:
        return None
    if isinstance(v, float):
        try:
            if np.isnan(v):
                return None
        except Exception:
            pass
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    return s

def main():
    ap = argparse.ArgumentParser(description="Apply class_map onto meta to produce target_class")
    ap.add_argument("--meta-in", default="data/meta/data_dict_processed.csv",
                    help="Input meta (prefer processed version)")
    ap.add_argument("--class-map", default="data/meta/class_map.csv",
                    help="CSV with at least columns: source_class,target_class")
    ap.add_argument("--meta-out", default="data/meta/data_dict_processed_labeled.csv",
                    help="Output meta with target_class (and hazard fields if exist)")
    ap.add_argument("--fallback", choices=["source","other"], default="source",
                    help="If a source_class not in class_map: use source as target, or 'other'")
    args = ap.parse_args()

    meta_p = Path(args.meta_in); cmap_p = Path(args.class_map)
    assert meta_p.exists(), f"Missing meta: {meta_p}"
    assert cmap_p.exists(), f"Missing class_map: {cmap_p}"

    meta = pd.read_csv(meta_p)
    assert "source_class" in meta.columns, "meta missing 'source_class'"

    cmap = pd.read_csv(cmap_p)
    assert "source_class" in cmap.columns and "target_class" in cmap.columns, \
        "class_map must have 'source_class' and 'target_class'"

    # 统一清洗字符串，避免空格/大小写/字符串'nan'问题
    meta["source_class"] = meta["source_class"].apply(_norm_str)
    if "target_class" in meta.columns:
        meta["target_class"] = meta["target_class"].apply(_norm_str)

    cmap["source_class"] = cmap["source_class"].apply(_norm_str)
    cmap["target_class"] = cmap["target_class"].apply(_norm_str)

    # 只带我们需要的列去合并（右表），并用后缀避免同名冲突
    opt_cols = [c for c in ["is_hazard","risk_weight"] if c in cmap.columns]
    right = cmap[["source_class","target_class"] + opt_cols].copy()

    merged = meta.merge(right, on="source_class", how="left", suffixes=("", "_map"))

    # 生成最终 target_class：优先映射，其次原列，再次按策略回退
    def _choose_target(row):
        t_map = _norm_str(row.get("target_class_map"))
        t_raw = _norm_str(row.get("target_class"))
        if args.fallback == "source":
            return t_map or t_raw or _norm_str(row.get("source_class")) or "other"
        else:
            return t_map or t_raw or "other"

    merged["target_class_final"] = merged.apply(_choose_target, axis=1)
    merged.drop(columns=[c for c in ["target_class_map"] if c in merged.columns], inplace=True)
    merged["target_class"] = merged["target_class_final"]
    merged.drop(columns=["target_class_final"], inplace=True)

    # 合并 is_hazard / risk_weight：优先采用映射(_map)，其次保留原列，最后补默认
    if "is_hazard_map" in merged.columns:
        if "is_hazard" not in merged.columns:
            merged["is_hazard"] = np.nan
        merged["is_hazard"] = merged["is_hazard_map"].combine_first(merged["is_hazard"])
        merged.drop(columns=["is_hazard_map"], inplace=True)
    if "is_hazard" in merged.columns:
        merged["is_hazard"] = merged["is_hazard"].fillna(0).astype(int)

    if "risk_weight_map" in merged.columns:
        if "risk_weight" not in merged.columns:
            merged["risk_weight"] = np.nan
        merged["risk_weight"] = merged["risk_weight_map"].combine_first(merged["risk_weight"])
        merged.drop(columns=["risk_weight_map"], inplace=True)
    if "risk_weight" in merged.columns:
        merged["risk_weight"] = merged["risk_weight"].fillna(0.0).astype(float)

    out_p = Path(args.meta_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_p, index=False)
    print(f"[OK] wrote {out_p}  (rows={len(merged)})")

if __name__ == "__main__":
    main()