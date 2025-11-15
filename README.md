run the script below to download the processed audio bundle from Google Drive verify integrity (SHA256), and extract files to the correct folders.
```bash
bash scripts/download_processed.sh
What this script does
	•	Downloads the archive and its .sha256 from Google Drive via gdown
	•	Verifies the archive with SHA256
	•	Extracts into:
	•	data/processed/
	•	data/meta/slices.csv
	•	data/meta/features_index.csv
	•	data/meta/label_map.json
	•	data/features/agg_stats.npz

Prerequisites
	•	If prompted: install zstd
	•	macOS: brew install zstd
	•	Ubuntu: sudo apt-get install zstd
git add README.md
git commit -m "docs: add bilingual instructions to download processed data"
git push
