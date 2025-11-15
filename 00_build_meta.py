#align audio in data/raw with dataset CSVs to create a unified data_dict.csv and a label mapping class_map.csv.
import pandas as pd, numpy as np
from pathlib import Path
import csv

RAW=Path("data/raw")
META_DIR=Path("data/meta"); META_DIR.mkdir(parents=True,exist_ok=True)

def find_csv(root:Path,name_keywords):
    cands=list(root.rglob("*.csv"))
    for p in cands:
        low=p.name.lower()
        if any(k in low for k in name_keywords):
            return p
        return None

def load_urbansound_meta(base:Path):
    meta_p=find_csv(base,["urbansound8k"])
    if not meta_p:return {}
    df=pd.read_csv(meta_p)
    return {row["slice_file_name"]:{"source_class":row["class"],"fold":int(row["fold"])} for _,row in df.iterrows()}

def load_esc50_meta(base:Path):
    meta_p=find_csv(base,["esc50"])
    if not meta_p:return {}
    df=pd.read_csv(meta_p)
    return {row["filename"]:{"source_class":row["category"],"fold":int(row["fold"])} for _,row in df.iterrows()}

def guess_source(path:Path):
    p=str(path).lower()
    if "urbansound" in p: return "urbansound8k"
    if "esc50" in p : return "esc50"
    if "self_collected" in p: return "self_collected"
    return "unknown"

def main():
    us_meta=load_urbansound_meta(RAW/"urbansound8k")
    esc_meta=load_esc50_meta(RAW/"esc50")
    rows=[]
    for wav in RAW.rglob("*.wav"):
        src=guess_source(wav)
        name=wav.name
        source_class,fold="",""
        if src=="urbansound8k" and name in us_meta:
            source_class=us_meta[name]["source_class"]
            fold=us_meta[name]["fold"]
        elif src=="esc50" and name in esc_meta:
            source_class=esc_meta[name]["source_class"]
            fold=esc_meta[name]["fold"]

        rows.append({
            "filepath": str(wav),
            "source": src,
            "split_hint": fold,              
            "start_time": "", "end_time": "",
            "source_class": source_class if source_class else "unknown",
            "target_class": "",              
            "device": "", "location_type": "", "spl_db": "",
            "session_id": "", "annotator": "", "license": ""
        })
    
    df=pd.DataFrame(rows).sort_values("filepath").reset_index(drop=True)
    df.to_csv(META_DIR / "data_dict.csv", index=False)

    uniq = sorted(df["source_class"].unique().tolist())
    map_p = META_DIR / "class_map.csv"
    if not map_p.exists():
        with open(map_p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source_class","target_class","notes"])
            for c in uniq:
                w.writerow([c, c, "edit target_class to your unified label"])
    else:
        pass

    print(f"[OK] Wrote {len(df)} rows to data/meta/data_dict.csv")
    print(f"[OK] Class map template at data/meta/class_map.csv (edit it next)")

if __name__ == "__main__":
    main()


