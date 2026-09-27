# Part of RAFT-Net. The notebooks write this file into the models/ folder of a ForMa clone
# (https://github.com/multimediaFor/ForMa) and import it from there.

# common.py -- the single protocol shared by EVERY full-run job (ForMa, RAFT-Net, all ablations).
import os, math, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

cv2.setNumThreads(0)
EPOCHS, BATCH, LR, WD = 40, 8, 1e-4, 1e-2
WARMUP_ITERS, MIN_LR_RATIO = 500, 0.01
INPUT = 512
LAMBDA_BOUNDARY = 1.0
ACR_START_EPOCH, ACR_RAMP_EPOCHS, ACR_LAMBDA, ACR_SUBBATCH = 20, 4, 0.05, 2   # 0-based: off for epochs 1-20
NUM_WORKERS = 2                     # two jobs share the 4 CPU cores
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def acr_lambda(epoch):
    """epoch = 0-based index of the epoch about to be trained."""
    if epoch < ACR_START_EPOCH:
        return 0.0
    return ACR_LAMBDA * min(1.0, (epoch - ACR_START_EPOCH + 1) / ACR_RAMP_EPOCHS)


# ── data (identical to the pilot, without the dropped DCT stream) ────────────────────────────
def _jpeg(img, q):
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def augment(img, mask):
    if random.random() >= 0.75:
        return img, mask
    if random.random() < 0.75:
        k = random.randrange(4)
        if k == 0:   img, mask = img[:, ::-1], mask[:, ::-1]
        elif k == 1: img, mask = img[::-1], mask[::-1]
        elif k == 2: img, mask = np.rot90(img), np.rot90(mask)
        else:        img, mask = img.transpose(1, 0, 2), mask.T
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
    if random.random() < 0.75:
        if random.random() < 0.5:
            img = _jpeg(img, random.randint(50, 95))
        else:
            h, w = img.shape[:2]
            s = random.uniform(0.5, 0.9)
            small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            img = cv2.resize(_jpeg(small, random.randint(50, 85)), (w, h), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.25:
        k = random.randrange(3)
        if k == 0:   img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 1.5))
        elif k == 1: img = cv2.medianBlur(img, 3)
        else:        img = np.clip(img.astype(np.float32) + np.random.normal(0, random.uniform(2, 8), img.shape), 0, 255).astype(np.uint8)
    return img, mask


def load_sample(ip, mp, train):
    img = cv2.imread(ip, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if mp else None
    m = np.zeros((h, w), np.uint8) if m is None else (m > 127).astype(np.uint8)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    if train:
        img, m = augment(img, m)
    x = cv2.resize(img, (INPUT, INPUT))[..., ::-1].astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)
    m = cv2.resize(m, (INPUT, INPUT), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return np.ascontiguousarray(x), m


class CASIASet(Dataset):
    def __init__(self, samples, train):
        self.samples, self.train = samples, train
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        x, m = load_sample(*self.samples[i], self.train)
        return torch.from_numpy(x), torch.from_numpy(m)


def _worker_init(wid):
    s = torch.initial_seed() % (2 ** 32)
    random.seed(s); np.random.seed(s)


def train_loader(samples, seed, epoch):
    """Order and augmentation are seeded per (seed, epoch): a resumed run sees exactly the batches an
    uninterrupted run would."""
    g = torch.Generator()
    g.manual_seed(seed * 1000 + epoch)
    return DataLoader(CASIASet(samples, True), batch_size=BATCH, shuffle=True, generator=g, drop_last=True,
                      num_workers=NUM_WORKERS, worker_init_fn=_worker_init, pin_memory=True)


def val_loader(samples):
    return DataLoader(CASIASet(samples, False), batch_size=8, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# ── losses: ForMa's official Dice + softmax-focal for EVERY model; RAFT-Net adds boundary + ACR ──
def init_forma_loss(repo_dir):
    import importlib.util as ilu
    spec = ilu.spec_from_file_location('forma_losses', os.path.join(repo_dir, 'losses.py'))
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MyLoss()


def boundary_target(m):
    m4 = m.unsqueeze(1)
    return (F.max_pool2d(m4, 5, 1, 2) + F.max_pool2d(-m4, 5, 1, 2)).clamp(0, 1)


def boundary_loss(logits, m):
    t = boundary_target(m)
    pos = t.mean().clamp(min=1e-4)
    bce = F.binary_cross_entropy_with_logits(logits.float(), t, pos_weight=((1 - pos) / pos).clamp(max=20.0))
    p = torch.sigmoid(logits.float())
    return bce + 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)


def main_loss(model, x, m, forma_loss):
    with torch.amp.autocast('cuda'):
        out = model(x)
    loss = forma_loss(out['mask'].float(), m)
    parts = {'seg': float(loss.detach())}
    if 'boundary' in out:
        lb = boundary_loss(out['boundary'], m)
        loss = loss + LAMBDA_BOUNDARY * lb
        parts['boundary'] = float(lb.detach())
    return loss, parts


def saliency_second_order(logits, feat):
    score = logits[:, 1].sum()
    grads = torch.autograd.grad(score, feat, create_graph=True, retain_graph=True)[0]
    return F.relu((grads.mean(dim=(2, 3), keepdim=True) * feat).sum(dim=1, keepdim=True))


def acr_tv_loss(S, prob, threshold=0.5):
    if S.shape[-2:] != prob.shape[-2:]:
        S = F.interpolate(S, size=prob.shape[-2:], mode='bilinear', align_corners=True)
    m_hat = (prob > threshold).float()
    dh, dw = (S[:, :, 1:] - S[:, :, :-1]).abs(), (S[:, :, :, 1:] - S[:, :, :, :-1]).abs()
    mh = ((m_hat[:, :, 1:] + m_hat[:, :, :-1]) > 0).float()
    mw = ((m_hat[:, :, :, 1:] + m_hat[:, :, :, :-1]) > 0).float()
    return (((dh * mh).sum((1, 2, 3)) + (dw * mw).sum((1, 2, 3))) / m_hat.sum((1, 2, 3)).clamp(min=1.0)).mean()


def acr_subbatch_loss(model, x, n=ACR_SUBBATCH):
    """Fixed ACR (verified by the diagnostic + pilot smoke test): second-order Grad-CAM with AMP OFF on n
    images; BatchNorm uses running statistics in this pass so the tiny sub-batch cannot corrupt them."""
    was_training = model.training
    model.eval()
    try:
        with torch.amp.autocast('cuda', enabled=False):
            out = model(x[:n].float(), None, return_feat=True)
            logits = F.interpolate(out['mask'], size=x.shape[-2:], mode='bilinear', align_corners=True)
            return acr_tv_loss(saliency_second_order(logits, out['feat']), F.softmax(logits, dim=1)[:, 1:2])
    finally:
        model.train(was_training)


def make_schedule(opt, steps_per_epoch):
    total = steps_per_epoch * EPOCHS
    def f(step):
        if step < WARMUP_ITERS:
            return (step + 1) / WARMUP_ITERS
        t = min(1.0, (step - WARMUP_ITERS) / max(1, total - WARMUP_ITERS))
        return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * t))
    return optim.lr_scheduler.LambdaLR(opt, f)


def param_sq(model):
    return sum(float(p.detach().float().pow(2).sum()) for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def validate(model, loader, device, acc_cls):
    """CASIA v2 validation, per-image F1 @0.5 (checkpoint selection only -- test sets are never used here)."""
    model.eval()
    acc = acc_cls('casia2_val')
    for x, m in loader:
        with torch.amp.autocast('cuda'):
            out = model(x.to(device, non_blocking=True))['mask']
        d = F.interpolate((out[:, 1:2] - out[:, 0:1]).float(), (INPUT, INPUT), mode='bilinear', align_corners=True)
        for p, g in zip(torch.sigmoid(d)[:, 0].cpu().numpy(), m.numpy()):
            acc.add(p, g if g.any() else None, int(g.any()))
    model.train()
    return acc.summary()
