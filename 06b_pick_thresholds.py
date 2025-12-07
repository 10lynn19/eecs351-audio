# 06b_pick_thresholds.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

# 你可以在这里设定“安全偏好”：关键类追求 recall，易误报类追求 precision
RECALL_MIN = {"siren": 0.80, "car_horn": 0.85}
PREC_MIN   = {"glass_breaking": 0.85, "impulse_bang": 0.75}

def metrics_at(y_true_bin, y_score, t):
    pred = (y_score >= t).astype(np.int32)
    tp = int(((pred == 1) & (y_true_bin == 1)).sum())
    fp = int(((pred == 1) & (y_true_bin == 0)).sum())
    fn = int(((pred == 0) & (y_true_bin == 1)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1, tp, fp, fn

def choose_threshold(y_true, probs, classes, out_dir="outputs_2d/eval"):
    out = []
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    n, C = probs.shape
    for k, cls in enumerate(classes):
        y_bin = (y_true == k).astype(np.int32)
        best = {"cls": cls, "t": 0.5, "precision":0, "recall":0, "f1":0, "picked_by":"f1"}
        # 先找满足“安全偏好”的阈值，否则退回F1最大
        pref = None
        if cls in RECALL_MIN: pref = ("recall", RECALL_MIN[cls])
        if cls in PREC_MIN:   pref = ("precision", PREC_MIN[cls]) if pref is None else pref
        best_pref = None
        for t in thresholds:
            p, r, f1, tp, fp, fn = metrics_at(y_bin, probs[:, k], t)
            if pref:
                name, target = pref
                val = r if name == "recall" else p
                if val >= target and (best_pref is None or (r if name=="recall" else p) > (best_pref[name])):
                    best_pref = {"cls":cls,"t":float(t),"precision":float(p),"recall":float(r),"f1":float(f1),"picked_by":name}
            if f1 > best["f1"]:
                best = {"cls":cls,"t":float(t),"precision":float(p),"recall":float(r),"f1":float(f1),"picked_by":"f1"}
        out.append(best_pref if best_pref else best)
    df = pd.DataFrame(out).sort_values("cls")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "per_class_thresholds.csv", index=False)
    with open(Path(out_dir) / "thresholds.json", "w") as f:
        json.dump({"thresholds": {r["cls"]: r["t"] for _, r in df.iterrows()},
                   "picked": out}, f, indent=2)
    print(df)
    print(f"[OK] saved thresholds -> {out_dir}/thresholds.json")
    return df

def main(val_npz="outputs_2d/eval/val_probs.npz", out_dir="outputs_2d/eval"):
    data = np.load(val_npz, allow_pickle=True)
    P = data["probs"]; y = data["y_true"]; classes = list(data["classes"])
    choose_threshold(y, P, classes, out_dir=out_dir)

if __name__ == "__main__":
    main()