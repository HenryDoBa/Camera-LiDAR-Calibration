"""Point-cloud I/O, frame timestamp matching, and CSV helpers."""
from __future__ import annotations
from pathlib import Path
import csv
import re
import struct
import numpy as np

LIDAR_EXT = {'.ply', '.pcd', '.bin', '.npy', '.npz', '.xyz', '.txt', '.csv'}
IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def list_files(root, extensions):
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(f'Folder does not exist: {root}')
    files = [p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in extensions]
    def natural_key(p):
        match = re.search(r'fn(\d+)', p.stem, re.IGNORECASE)
        # Keep a single sequence in numeric frame order (fn1, fn8, fn9, fn10).
        return (str(p.parent), int(match.group(1)) if match else 10**18, p.name)
    return sorted(files, key=natural_key)


def _load_ply(path):
    """Standard ASCII/binary little endian vertex PLY with arbitrary extra fields."""
    types = {'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
             'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
             'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
             'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8'}
    with open(path, 'rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError(f'Invalid PLY: {path}')
        mode = None
        vertices, properties, in_vertices = None, [], False
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Truncated PLY header')
            tokens = line.decode('ascii', 'replace').strip().split()
            if not tokens:
                continue
            if tokens[0] == 'format':
                mode = tokens[1]
            if tokens[0] == 'element':
                in_vertices = tokens[1] == 'vertex'
                if in_vertices:
                    vertices = int(tokens[2])
            elif tokens[0] == 'property' and in_vertices:
                if tokens[1] == 'list':
                    raise ValueError('List property in PLY vertex is unsupported; install open3d')
                properties.append((tokens[2], types[tokens[1]]))
            if tokens[0] == 'end_header':
                break
        if vertices is None or not {'x', 'y', 'z'} <= {p[0] for p in properties}:
            raise ValueError('PLY vertex must have x, y, z fields')
        if mode == 'ascii':
            data = np.loadtxt(f, dtype=float, max_rows=vertices)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            names = [p[0] for p in properties]
            return data[:, [names.index(v) for v in ('x', 'y', 'z')]]
        if mode == 'binary_little_endian':
            dtype = np.dtype([(n, '<' + d) for n, d in properties])
            arr = np.fromfile(f, dtype=dtype, count=vertices)
            return np.column_stack([arr[a] for a in ('x', 'y', 'z')])
        raise ValueError(f'Unsupported PLY encoding: {mode}')


def read_points(path, bin_dim=4):
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == '.ply':
        try:
            arr = _load_ply(p)
        except (ValueError, KeyError) as exc:
            try:
                import open3d as o3d
            except ImportError:
                raise ValueError(f'{exc}. Install open3d for other PLY flavors') from exc
            arr = np.asarray(o3d.io.read_point_cloud(str(p)).points)
    elif suffix == '.pcd':
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError('For .pcd clouds, install open3d: pip install open3d') from exc
        arr = np.asarray(o3d.io.read_point_cloud(str(p)).points)
    elif suffix == '.bin':
        raw = np.fromfile(p, dtype='<f4')
        if len(raw) % bin_dim:
            raise ValueError(f'Binary point file {p} incompatible with --bin-dim {bin_dim}')
        arr = raw.reshape(-1, bin_dim)[:, :3]
    elif suffix == '.npy':
        arr = np.load(p)
    elif suffix == '.npz':
        with np.load(p) as data:
            key = 'points' if 'points' in data else next(iter(data.files))
            arr = data[key]
    elif suffix in {'.csv', '.txt', '.xyz'}:
        arr = np.loadtxt(p, delimiter=',' if suffix == '.csv' else None)
    else:
        raise ValueError(f'Unsupported point cloud: {p}')
    arr = np.asarray(arr, dtype=np.float64).reshape(-1, np.asarray(arr).shape[-1])
    if arr.shape[1] < 3:
        raise ValueError('Need at least three point coordinates')
    return arr[:, :3][np.isfinite(arr[:, :3]).all(axis=1)]


def write_colored_ply(path, points, mask):
    """Gray points for background, red for sphere inliers (binary little endian PLY)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float32)
    mask = np.asarray(mask, bool)
    if len(mask) != len(pts):
        raise ValueError('Color mask length mismatch')
    data = np.zeros(len(pts), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, c in enumerate('xyz'):
        data[c] = pts[:, i]
    data['red'], data['green'], data['blue'] = 135, 135, 135
    data['red'][mask], data['green'][mask], data['blue'][mask] = 255, 0, 0
    with path.open('wb') as f:
        header = (f'ply\nformat binary_little_endian 1.0\n'
                  f'element vertex {len(data)}\n'
                  'property float x\nproperty float y\nproperty float z\n'
                  'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                  'end_header\n')
        f.write(header.encode('ascii'))
        data.tofile(f)


def timestamp_from_name(file):
    """Parse Unix epoch seconds (10 digits), ms(13), us(16), ns(19); fractions allowed.

    Names with other layouts need matching by stem/index or custom matching.
    """
    name = Path(file).stem
    tokens = re.findall(r'(?<!\d)(\d{10,19})(?:\.(\d+))?(?!\d)', name)
    if not tokens:
        return None
    main, frac = max(tokens, key=lambda e: len(e[0]))
    digits = len(main)
    if digits >= 18:
        secs = int(main) / 1e9
    elif digits >= 15:
        secs = int(main) / 1e6
    elif digits >= 12:
        secs = int(main) / 1e3
    else:
        secs = float(main)
        if frac:
            secs += float('0.' + frac)
    return secs


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        out = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        out.writeheader()
        out.writerows(rows)


def read_overrides(path, keys):
    if not path:
        return {}
    rows = read_csv(path)
    return {Path(row['filename']).stem: tuple(float(row[k]) for k in keys)
            for row in rows}


def pair_rows(lidar_rows, camera_rows, mode='timestamp', max_dt=0.06, offset=0.0):
    """Pair nearest available camera frame; positive offset shifts camera timestamp."""
    lidar = [r for r in lidar_rows if r['status'] == 'ok']
    camera = [r for r in camera_rows if r['status'] == 'ok']
    if mode == 'frameid':
        # IMPORTANT: same fn identifier means nominally corresponding recording frame,
        # not independently verified timestamp synchronization.
        def fid(name):
            m = re.search(r'fn(\d+)(?:$|[^\d])', name, re.IGNORECASE)
            return int(m.group(1)) if m else None
        by_id = {fid(c['stem']): c for c in camera if fid(c['stem']) is not None}
        return [(l, by_id[fid(l['stem'])], None)
                for l in lidar if fid(l['stem']) is not None and fid(l['stem']) in by_id]
    if mode == 'index':
        return [(l, c, None) for l, c in zip(lidar, camera)]
    if mode == 'name':
        by_stem = {c['stem']: c for c in camera}
        return [(l, by_stem[l['stem']], 0.0) for l in lidar if l['stem'] in by_stem]
    if mode != 'timestamp':
        raise ValueError('pair mode must be timestamp, frameid, name or index')
    valid_l = [l for l in lidar if timestamp_from_name(l['stem']) is not None]
    valid_c = [c for c in camera if timestamp_from_name(c['stem']) is not None]
    if not valid_l or not valid_c:
        raise ValueError('Could not parse timestamps. Check filenames or use --pair-mode name/index')
    cam_sorted = sorted([(timestamp_from_name(c['stem']) + offset, c) for c in valid_c],
                        key=lambda x: x[0])
    times = np.asarray([a[0] for a in cam_sorted])
    used = set()
    pairs = []
    for l in sorted(valid_l, key=lambda r: timestamp_from_name(r['stem'])):
        t = timestamp_from_name(l['stem'])
        i = int(np.searchsorted(times, t))
        choices = range(max(0, i - 3), min(len(times), i + 4))
        choices = [j for j in choices if j not in used]
        if not choices:
            continue
        j = min(choices, key=lambda x: abs(times[x] - t))
        delta = float(times[j] - t)
        if abs(delta) <= max_dt:
            used.add(j)
            pairs.append((l, cam_sorted[j][1], delta))
    return pairs
