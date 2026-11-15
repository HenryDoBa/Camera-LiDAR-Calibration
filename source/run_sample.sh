#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SAMPLE_ZIP="${1:-$HOME/Downloads/camera_lidar_sample.zip}"
if [[ ! -f "$SAMPLE_ZIP" ]]; then echo "Cannot find sample ZIP: $SAMPLE_ZIP"; exit 1; fi
mkdir -p sample_data
python - "$SAMPLE_ZIP" <<'PY'
import zipfile,sys
from pathlib import Path
with zipfile.ZipFile(sys.argv[1]) as z:
    for member in z.infolist():
        dest=(Path('sample_data')/member.filename).resolve()
        if not dest.is_relative_to(Path('sample_data').resolve()):
            raise ValueError('Unsafe ZIP filename')
    z.extractall('sample_data')
PY
python run.py lidar --lidar-dir sample_data/1_cartesians --out results_sample --seed-csv config/sample_lidar_seeds.csv --roi-radius 0.65 --min-inliers 18 --manual-only
python run.py camera --camera-dir sample_data/1_pictures --out results_sample --yellow-auto --edge-threshold 3.5
python run.py review --out results_sample --pair-mode frameid
printf '\nCompleted 15-frame diagnostic. Open results_sample/review.html.\n'
printf 'There are only five complete camera/LiDAR detection pairs in this sample.\n'
printf 'Do NOT attempt registration from fewer than six manually validated pairs.\n'
