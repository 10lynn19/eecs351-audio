bash -n scripts/download_processed.sh
set -euo pipefail
FILE_ID_TAR="14sAfXy34U1Qq_4f2057XF760tt5ajYl6"
FILE_ID_SHA="11rUBHMb9WVrYKVetUX5U3vqDDBbkElV1"
PKG="data-processed-v1.tar.zst"  # 如果你用的是 gzip，就改为 data-processed-v1.tar.gz

python -m pip install -q --upgrade pip gdown
gdown --fuzzy "https://drive.google.com/uc?id=${FILE_ID_TAR}" -O "${PKG}"
gdown --fuzzy "https://drive.google.com/uc?id=${FILE_ID_SHA}" -O "data-processed-v1.sha256"
shasum -a 256 -c data-processed-v1.sha256

# 解压（zstd 包的通用写法；若是 .tar.gz，用 tar -xzf）
if [[ "${PKG}" == *.tar.zst ]]; then
  if ! command -v zstd >/dev/null; then
    echo "zstd not found. Install it: brew install zstd"
    exit 1
  fi
  zstd -d -c "${PKG}" | tar -xf -
else
  tar -xzf "${PKG}" -C .
fi

echo "[OK] Ready: data/processed + d
echo "[OK] Ready: data/processed + data/meta/*.csv + data/features/agg_stats.npz"
