#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BASE="${1:-$HOME/Downloads}"
if [[ ! -d "$BASE/1_cartesians" || ! -d "$BASE/1_pictures" ]]; then
  echo "Expected $BASE/1_cartesians and $BASE/1_pictures"; exit 1
fi
python run.py lidar --lidar-dir "$BASE/1_cartesians" --out results_full --seed-csv config/sample_lidar_seeds.csv --roi-radius 0.65 --threshold 0.022 --min-inliers 18
python run.py camera --camera-dir "$BASE/1_pictures" --out results_full --yellow-auto --edge-threshold 3.5
python run.py review --out results_full --pair-mode frameid
printf '\nOpen results_full/review.html and inspect BOTH images and colored PLY.\n'
printf 'Create selected_frames.txt with >=6 manually validated LiDAR stems. Then run:\n'
printf 'python run.py register --out results_full --selected selected_frames.txt\n'
printf 'python run.py validate --out results_full\n'
