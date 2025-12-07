#!/usr/bin/env bash
set -e
NAME="eecs351_audio_release_$(date +%Y%m%d)"
OUT="${NAME}.zip"
echo "[*] creating ${OUT}"

zip -r "${OUT}" \
  00_build_meta.py 01_preprocess_audio.py 01b_apply_class_map.py 02_slice_label.py \
  03_extract_features.py 04_make_split.py 05a_train_logreg_1d.py \
  06a_predict_probs_2d.py 06b_pick_thresholds.py 06c_apply_thresholds_and_report.py \
  07_quickdemo.py \
  scripts/ \
  outputs_2d/models/cnn2d_best.keras \
  outputs_2d/models/norm_stats.json \
  outputs_2d/eval/thresholds_balanced.json \
  outputs_2d/eval/thresholds_safety.json \
  outputs_2d/final/ \
  data/demo/ \
  data/meta/class_map.csv \
  requirements.txt requirements.lock.txt README.md \
  -x "*.DS_Store" \
  -x ".git/*" \
  -x ".venv/*" \
  -x "data/raw/*" \
  -x "data/processed/*" \
  -x "data/features/*" \
  -x "data-processed-*.tar.*" \
  -x "outputs/*" \
  -x "outputs_2d/eval/*probs.npz"

echo "[OK] wrote ${OUT}"
