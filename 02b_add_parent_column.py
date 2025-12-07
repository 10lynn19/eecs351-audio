# 02b_add_parent_column_auto.py
# Auto-derive a robust parent_id for grouping slices by their original recording.
# It evaluates multiple heuristics and picks the one that yields "reasonable" grouping.

import re
from pathlib import Path
import pandas as pd
from collections import OrderedDict
import numpy as np

SRC = Path("data/meta/slices.csv")
OUT = Path("data/meta/slices.with_parent.csv")

def parent_dir_id(p: Path) -> str:
    # If your layout has subfolders under .../slices/<PARENT>/file.wav, this works well.
    parts = p.parts
    if "slices" in parts:
        i = parts.index("slices")
        if i + 2 < len(parts):
            return parts[i + 1]
    return p.parent.name  # often "slices" -> will collapse to 1 group; we'll penalize it later.

_PATTERNS_MARKER = r'[_-](?:st|start|t|seg|slice|part|chunk|idx|fr|s)\d+.*$'
def stem_candidates(stem: str) -> OrderedDict:
    """Generate several filename-based candidates for parent id."""
    s0 = re.sub(r'^(norm_|orig_)', '', stem)  # drop processing prefix
    c = OrderedDict()
    # cut at markers like _st123, -slice12, etc.
    cut_marker = re.split(_PATTERNS_MARKER, s0, maxsplit=1)[0]
    c["cut_marker"]       = cut_marker
    # cut trailing ..._123ms or ..._123ms_456ms
    c["cut_ms"]           = re.sub(r'[_-]?\d+ms(?:[_-]\d+ms)?$', '', c["cut_marker"])
    # cut long digit tails ..._123456 or ...-123456
    c["cut_longdigits"]   = re.sub(r'[_-]?\d{6,}$', '', c["cut_ms"])
    # drop last pure-numeric token ..._<num>
    c["drop_last_numtok"] = re.sub(r'[_-]\d+$', '', c["cut_longdigits"])
    # as-is (after prefix removal)
    c["stem"]             = s0
    return c

def pick_best_parent(df: pd.DataFrame):
    paths = df["slice_path"].astype(str).tolist()
    p_objs = [Path(p) for p in paths]
    stems  = [pp.stem for pp in p_objs]

    # build candidates
    cand_map = OrderedDict()
    cand_map["dir"] = [parent_dir_id(pp) for pp in p_objs]
    # filename-based
    cands = [stem_candidates(s) for s in stems]
    for key in cands[0].keys():
        cand_map[key] = [c[key] for c in cands]

    # score each candidate
    scores = []
    for name, vals in cand_map.items():
        n_unique = len(set(vals))
        ratio = n_unique / len(vals)
        # group sizes
        tmp = pd.DataFrame({"pid": vals, "label": df["label"]})
        grp = tmp.groupby("pid").size().values
        med = float(np.median(grp)) if len(grp) else 0.0
        # label conflicts (ideally 0)
        confl = tmp.groupby("pid")["label"].nunique()
        conflicts = int((confl > 1).sum())
        # score: prefer some grouping (0.01<=ratio<=0.5), no conflicts, larger median size
        ok_range = 0.01 <= ratio <= 0.5
        score = (
            (0 if conflicts==0 else -1000) +           # hard penalty on conflicts
            (50 if ok_range else -50) +                 # prefer reasonable ratio
            (min(50, med)) -                            # prefer bigger typical group (cap at 50)
            (abs(ratio-0.1)*10)                         # prefer around 0.1
        )
        scores.append((name, n_unique, ratio, med, conflicts, score))

    # choose best by score; break ties by fewer conflicts, then ratio closeness to 0.1
    scores.sort(key=lambda x: (-x[5], x[4], abs(x[2]-0.1)))
    best = scores[0]
    best_name = best[0]
    print("[CANDIDATES]")
    for row in scores:
        print(f"  {row[0]:>16s} | uniques={row[1]:6d}  ratio={row[2]:.3f}  med_size={row[3]:.1f}  conflicts={row[4]:3d}  score={row[5]:6.1f}")
    print(f"[PICK] best='{best_name}'")

    parent_id = cand_map[best_name]
    return best_name, parent_id

def main():
    df = pd.read_csv(SRC)
    # quick check
    need_cols = {"slice_path","label"}
    missing = need_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in slices.csv: {missing}")

    best_name, parent_id = pick_best_parent(df)
    df["parent_id"] = parent_id
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[OK] wrote {OUT}  | slices={len(df)}  unique parent_id={df['parent_id'].nunique()}  (from='{best_name}')")

if __name__ == "__main__":
    main()