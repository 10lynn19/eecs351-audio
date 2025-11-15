import librosa, soundfile as sf, numpy as np
from scipy.signal import butter, lfilter
from pathlib import Path
import pandas as pd

SR=16000
HPF_CUT=100.0
TARGET_RMS=0.1

def butter_highpass(cut,fs,order=4):
    from scipy.signal import butter
    b,a =butter(order,cut/(fs/2),btype='highpass')
    return b,a

def apply_hpf(x,fs,cut=HPF_CUT):
    b,a=butter_highpass(cut,fs)
    return lfilter(b,a,x)

def rms_norm(x,target_rms=TARGET_RMS,eps=1e-8):
    rms=np.sqrt(np.mean(x**2)+eps)
    return x*(target_rms/max(rms,eps))

def main():
    meta_in=Path("data/meta/data_dict.csv")
    assert meta_in.exists(), "run 00" 
    meta=pd.read_csv(meta_in)

    outdir=Path("data/processed")
    outdir.mkdir(parents=True,exist_ok=True)

    rows=[]
    for i, r in meta.iterrows():
        f = r["filepath"]
        try:
            y, sr = librosa.load(f, sr=SR, mono=True)
        except Exception as e:
            print(f"[SKIP] Failed to read: {f}  ({e})")
            continue

        y = apply_hpf(y, SR, HPF_CUT)

        # Quality flag (clipping)
        clip_ratio = float(np.mean(np.abs(y) >= 0.999))
        quality = "ok" if clip_ratio < 0.01 else "clipped"

        base = Path(f).with_suffix("").name
        p_orig = outdir / f"orig_{base}.wav"
        p_norm = outdir / f"norm_{base}.wav"

        sf.write(p_orig, y, SR, subtype="PCM_16")
        sf.write(p_norm, rms_norm(y), SR, subtype="PCM_16")

        rows.append({**r,
            "proc_orig": str(p_orig),
            "proc_norm": str(p_norm),
            "quality": quality
        })

    pd.DataFrame(rows).to_csv("data/meta/data_dict_processed.csv", index=False)
    print(f"[OK] Wrote {len(rows)} rows to data/meta/data_dict_processed.csv")

if __name__ == "__main__":
    main()