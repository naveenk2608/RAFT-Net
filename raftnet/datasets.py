# Part of RAFT-Net. The notebooks write this file into the models/ folder of a ForMa clone
# (https://github.com/multimediaFor/ForMa) and import it from there.

# ── Dataset loaders: every entry is (image_path, mask_path_or_None, label, mask_kind) ──────
# mask_kind tells load_gt() how to turn the mask file into {0,1}:
#   'binary'   -> grayscale > 127 (CASIA v1/v2, IMD2020)
#   'columbia' -> colour-coded region map: tampered = pixels where the COLUMBIA_TAMPERED
#                 colour channel dominates (NOT grayscale>127, which keeps only the thin
#                 bright-green boundary band -- the bug behind the old Columbia numbers)
import os, glob, json, random
import numpy as np
import cv2

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
COLUMBIA_TAMPERED = 'green'   # verify against the README printed in the Columbia cell


def load_gt(mask_path, mask_kind, target_hw):
    """Returns a {0,1} uint8 mask at target_hw=(H, W) of the ORIGINAL image, or None."""
    if mask_path is None:
        return None
    if mask_kind == 'columbia':
        bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        g = bgr[..., 1].astype(np.int16)
        r = bgr[..., 2].astype(np.int16)
        m = (g > r) if COLUMBIA_TAMPERED == 'green' else (r > g)
        m = m.astype(np.uint8)
    else:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        m = (m > 127).astype(np.uint8)
    if m.shape[:2] != tuple(target_hw):
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m


# ── CASIA v2 in-distribution split ─────────────────────────────────────────────────────
# 'validated' = the split used by every RAFT-Net stage, MPC and LoMa (unreadable files such as
#               Thumbs.db dropped BEFORE shuffling -> 12472 samples -> 11225/1247).
# 'cfl'       = CFL-Net's split (nothing dropped -> 12473 samples incl. Thumbs.db -> 11226/1247).
# The two share only ~41% of their validation images, so CFL-Net must be scored on its OWN
# split (the other one contains ~59% of CFL-Net's training images).
def casia2_file_list(au_dir, tp_dir, gt_dir):
    samples = []
    for f in sorted(os.listdir(au_dir)):
        p = os.path.join(au_dir, f)
        if os.path.isfile(p) and not f.startswith('.'):
            samples.append((p, None))
    gt_by_stem = {}
    for f in os.listdir(gt_dir):
        if f.startswith('.'):
            continue
        stem = os.path.splitext(f)[0]
        if stem.endswith('_gt'):
            stem = stem[:-3]
        gt_by_stem[stem] = os.path.join(gt_dir, f)
    for f in sorted(os.listdir(tp_dir)):
        p = os.path.join(tp_dir, f)
        if not (os.path.isfile(p) and not f.startswith('.')):
            continue
        m = gt_by_stem.get(os.path.splitext(f)[0])
        if m is not None:
            samples.append((p, m))
    return samples


def casia2_val_split(au_dir, tp_dir, gt_dir, variant, seed=1221, val_fraction=0.1,
                     readable_check=None):
    samples = casia2_file_list(au_dir, tp_dir, gt_dir)
    if variant == 'validated':
        assert readable_check is not None
        samples = [s for s in samples if readable_check(s)]
    elif variant != 'cfl':
        raise ValueError(variant)
    random.Random(seed).shuffle(samples)
    n_val = int(len(samples) * val_fraction)
    val, train = samples[:n_val], samples[n_val:]
    # CFL's list can contain an unreadable file in val; it is skipped at load time exactly as
    # CFL-Net's own evaluation did (reported as skipped).
    as_entries = lambda lst: [(img, m, 0 if m is None else 1, 'binary') for img, m in lst]
    return as_entries(val), as_entries(train)


# ── Columbia ────────────────────────────────────────────────────────────────────────────
def columbia_entries(root):
    splc_img = [d for d in glob.glob(os.path.join(root, '**', '4cam_splc'), recursive=True)
                if any(f.lower().endswith(('.tif', '.tiff')) for f in os.listdir(d))]
    auth_img = [d for d in glob.glob(os.path.join(root, '**', '4cam_auth'), recursive=True)
                if any(f.lower().endswith(('.tif', '.tiff')) for f in os.listdir(d))]
    edge_dirs = [d for d in glob.glob(os.path.join(root, '**', 'edgemask'), recursive=True)
                 if '4cam_splc' in d]
    assert splc_img and auth_img and edge_dirs, (splc_img, auth_img, edge_dirs)
    edge = {}
    for d in edge_dirs:
        for f in os.listdir(d):
            stem = os.path.splitext(f)[0]
            if stem.endswith('_edgemask'):          # '_edgemask_3' variants are ignored
                edge[stem[:-len('_edgemask')]] = os.path.join(d, f)
    entries, unmatched = [], []
    for f in sorted(os.listdir(splc_img[0])):
        if not f.lower().endswith(('.tif', '.tiff')):
            continue
        stem = os.path.splitext(f)[0]
        m = edge.get(stem)
        if m is None:
            unmatched.append(f)
        entries.append((os.path.join(splc_img[0], f), m, 1, 'columbia'))
    for f in sorted(os.listdir(auth_img[0])):
        if f.lower().endswith(('.tif', '.tiff')):
            entries.append((os.path.join(auth_img[0], f), None, 0, 'binary'))
    return entries, unmatched


# ── IMD2020 ─────────────────────────────────────────────────────────────────────────────
def imd2020_entries(root):
    auth_dir, img_dir, mask_dir = (os.path.join(root, d) for d in ('Authentic', 'img', 'mask'))
    masks = {}
    for f in os.listdir(mask_dir):
        stem = os.path.splitext(f)[0]
        if stem.lower().endswith('_mask'):
            masks[stem[:-5]] = os.path.join(mask_dir, f)
    entries, unmatched = [], []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith(IMG_EXTS):
            continue
        m = masks.get(os.path.splitext(f)[0])
        if m is None:
            unmatched.append(f)
        entries.append((os.path.join(img_dir, f), m, 1, 'binary'))
    for f in sorted(os.listdir(auth_dir)):
        if f.lower().endswith(IMG_EXTS):
            entries.append((os.path.join(auth_dir, f), None, 0, 'binary'))
    return entries, unmatched


# ── CASIA v1 ────────────────────────────────────────────────────────────────────────────
# The attached image mirror ships NO masks. Masks come from a separate groundtruth dataset
# (e.g. the community "CASIA 1.0 groundtruth" set: Sp_..._gt.png). Discovery is by NAME,
# not by folder layout: any image file under /kaggle/input whose stem (minus an optional
# '_gt' suffix) equals a CASIA1 tampered-image stem is taken as that image's mask.
def casia1_entries(casia1_root, search_root='/kaggle/input'):
    sp_dir = os.path.join(casia1_root, 'Sp')
    au_dir = os.path.join(casia1_root, 'Au')
    tamp = {os.path.splitext(f)[0]: os.path.join(sp_dir, f)
            for f in sorted(os.listdir(sp_dir)) if f.lower().endswith(IMG_EXTS)}
    masks = {}
    for r, dirs, files in os.walk(search_root):
        if os.path.abspath(r).startswith(os.path.abspath(sp_dir)):
            continue
        for f in files:
            if not f.lower().endswith(IMG_EXTS):
                continue
            stem = os.path.splitext(f)[0]
            base = stem[:-3] if stem.endswith('_gt') else stem
            if base in tamp and (stem.endswith('_gt') or 'truth' in r.lower() or 'mask' in r.lower()
                                 or 'gt' in os.path.basename(r).lower()):
                masks[base] = os.path.join(r, f)
    entries = [(p, masks.get(s), 1, 'binary') for s, p in tamp.items()]
    entries += [(os.path.join(au_dir, f), None, 0, 'binary')
                for f in sorted(os.listdir(au_dir)) if f.lower().endswith(IMG_EXTS)]
    return entries, sum(1 for s in tamp if s not in masks)
