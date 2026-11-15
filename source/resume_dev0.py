#!/usr/bin/env python3
"""Repair an already completed full run: keep only Dev0 camera detections.

Safe to re-run. Does not reprocess original photos or LiDAR PLYs.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        return list(rdr.fieldnames or []), list(rdr)


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fields)
        wr.writeheader()
        wr.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results_full'))
    parser.add_argument('--camera-device', default='Dev0')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = (root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    camera_csv = output / 'camera_centers.csv'
    lidar_csv = output / 'lidar_centers.csv'
    if not camera_csv.is_file() or not lidar_csv.is_file():
        raise SystemExit(f'Missing camera_centers.csv or lidar_centers.csv in {output}')
    backup = output / 'camera_centers_ALL_DEVICES_BACKUP.csv'
    if not backup.exists():
        shutil.copy2(camera_csv, backup)
        print('Backup created:', backup)
    fields, rows = read_csv(backup)
    keep = [r for r in rows if r.get('stem', '').startswith(args.camera_device + '_')]
    if not keep:
        raise SystemExit(f'No {args.camera_device} rows found in backup! Left original CSV unchanged.')
    write_csv(camera_csv, fields, keep)
    print(f'Camera rows: {len(rows)} total -> {len(keep)} {args.camera_device} rows')
    script = root / 'run.py'
    if not script.is_file():
        raise SystemExit('Put resume_dev0.py in the SAME folder as run.py')
    subprocess.run([sys.executable, str(script), 'review', '--out', str(output), '--pair-mode', 'frameid'], check=True, cwd=root)
    pairs_path = output / 'paired_centers.csv'
    _, pairs = read_csv(pairs_path)
    bad = [p for p in pairs if not p['camera_stem'].startswith(args.camera_device + '_')]
    if bad:
        raise RuntimeError(f'BUG: {len(bad)} matched pairs are not from {args.camera_device}!')
    ids = set()
    for p in pairs:
        m = re.search(r'fn(\d+)$', p['lidar_stem'])
        if m:
            ids.add(int(m.group(1)))
    recommended = [48, 52, 58, 64, 69, 75, 81, 87, 93, 98]
    candidate_lines = [f'test_fn{i}' for i in recommended if i in ids]
    path = root / 'CANDIDATES_TO_CHECK_NOT_YET_APPROVED.txt'
    path.write_text('# These are only candidates; inspect BOTH camera ellipse and colored LiDAR PLY.\n' +
                    '\n'.join(candidate_lines) + '\n', encoding='utf-8')
    print(f'\nDev0 LiDAR+camera candidate pairs: {len(pairs)}')
    print('Candidate list (NOT manually approved):', path)
    # Small QC bundle the student can share for human review. No full dataset needed.
    qc_zip = root / 'Dev0_QC_upload.zip'
    chosen = set(candidate_lines)
    files = [output / name for name in ('paired_centers.csv', 'camera_centers.csv',
                                       'lidar_centers.csv', 'review.html')]
    files.append(path)
    for pair in pairs:
        if pair['lidar_stem'] in chosen:
            files.append(output / 'lidar_colored_ply' / (pair['lidar_stem'] + '.ply'))
            files.append(output / 'camera_overlays' / (pair['camera_stem'] + '.jpg'))
    unique = list(dict.fromkeys(files))
    with ZipFile(qc_zip, 'w', ZIP_DEFLATED, compresslevel=5) as z:
        for f in unique:
            if f.is_file():
                z.write(f, f.relative_to(root) if f.is_relative_to(root) else f.name)
    print('QC bundle for sharing:', qc_zip)
    print('HTML to review:', output / 'review.html')
    print('Next: manually create selected_frames.txt with >=6 checked frames, then register/validate.')

if __name__ == '__main__':
    main()
