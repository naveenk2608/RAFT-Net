# Part of RAFT-Net. The notebooks write this file into the models/ folder of a ForMa clone
# (https://github.com/multimediaFor/ForMa) and import it from there.

# raftnet.py -- RAFT-Net and the official ForMa baseline, built on ForMa's own classes.
import torch
import torch.nn as nn
import torch.nn.functional as F
from vmamba_pixelshuf_modals import (Backbone_VSSM, SegFormerHead, ModalMixer, ModalitiesExtractor, ConvModule)

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


class Denorm(nn.Module):
    """Model input is ImageNet-normalised (for the backbone); forensic extractors need raw [0,1] pixels
    (Noiseprint++ was trained on [0,1] RGB). Applied identically in ForMa and RAFT-Net."""
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor(IMNET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMNET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return x * self.std + self.mean


def conv_bn_relu(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


# ── Official ForMa baseline ───────────────────────────────────────────────────────────────────
class ForMaOfficial(nn.Module):
    """Forensic_Vmamba as released: VMamba-T backbone; Noiseprint++ (frozen) + Bayar + SRM on the image
    shrunk to 1/4, mixed to 3 channels, fed to the decoder only."""
    def __init__(self, np_weights):
        super().__init__()
        self.denorm = Denorm()
        self.backbone = Backbone_VSSM()
        self.decode = SegFormerHead()
        self.modalmixer = ModalMixer()
        self.modal_extractor = ModalitiesExtractor(['noiseprint', 'bayar', 'srm'], np_weights)

    def train(self, mode=True):
        super().train(mode)
        self.modal_extractor.noiseprint.eval()     # ForMa's model.train() would silently re-enable BN updates
        return self

    def forward(self, x, dct=None, return_feat=False):
        n, _, h, w = x.shape
        y = self.backbone(x)
        xm = F.interpolate(self.denorm(x), size=(h // 4, w // 4), mode='bilinear', align_corners=False)
        xm = self.modalmixer(self.modal_extractor.multi_output(xm))
        return {'mask': self.decode(y, xm)}


class ForMaWide(ForMaOfficial):
    """Official ForMa with ONE change: the width w of its forensic pathway. ForMa's mixer compresses 96 channels of
    noise features to 3 (then linearly expands them to 16 before fusion with 384 RGB channels). Here the mixer outputs
    w channels and w channels enter the fusion layer; backbone, extractors, mixer blocks and decoder are unchanged."""
    def __init__(self, np_weights, width):
        super().__init__(np_weights)
        self.modalmixer = ModalMixer(out_channels=width)
        self.decode.conv_modals = nn.Conv2d(width, width, kernel_size=1)
        self.decode.linear_fuse = ConvModule(c1=96 * 4 + width, c2=96, k=1)


# ── RAFT-Net components ───────────────────────────────────────────────────────────────────────
class ForensicStem(nn.Module):
    """Forensic evidence at every backbone scale (1/4, 1/8, 1/16, 1/32).
    fullres=True : SRM + Bayar residuals computed on the FULL-resolution image, downsampled by learned
                   stride-2 convolutions (ForMa computes them on a bilinearly shrunk image, which blurs away
                   the fine residuals these filters exist to capture).
    Noiseprint++ : frozen camera-fingerprint net, at 1/4 as in ForMa (full resolution is ~16x too slow on a T4).
    dct=True     : native-grid block-DCT map (computed in the data loader on the original-resolution image,
                   JPEG grid intact) injected at 1/8."""
    def __init__(self, np_weights, fullres=True, dct=True, widths=(64, 128, 192, 256)):
        super().__init__()
        self.fullres, self.use_dct, self.widths = fullres, dct, widths
        self.hp = ModalitiesExtractor(['bayar', 'srm'])
        self.npp = ModalitiesExtractor(['noiseprint'], np_weights)
        self.hp_down = (nn.Sequential(conv_bn_relu(6, 32, 3, 2), conv_bn_relu(32, 48, 3, 2)) if fullres
                        else conv_bn_relu(6, 48, 3, 1))
        self.np_conv = conv_bn_relu(3, 16, 3, 1)
        self.f4 = conv_bn_relu(48 + 16, widths[0], 3, 1)
        self.d8 = conv_bn_relu(widths[0], widths[1], 3, 2)
        if dct:
            self.dct_enc = nn.Sequential(conv_bn_relu(64, widths[1], 1, 1), conv_bn_relu(widths[1], widths[1], 3, 1))
        self.d16 = conv_bn_relu(widths[1], widths[2], 3, 2)
        self.d32 = conv_bn_relu(widths[2], widths[3], 3, 2)

    def forward(self, x01, dct=None):
        h, w = x01.shape[-2:]
        q = F.interpolate(x01, size=(h // 4, w // 4), mode='bilinear', align_corners=False)
        hp = torch.cat(self.hp.multi_output(x01 if self.fullres else q), dim=1)
        f4 = self.f4(torch.cat([self.hp_down(hp), self.np_conv(self.npp.multi_output(q)[0])], dim=1))
        f8 = self.d8(f4)
        if self.use_dct:
            if dct is None:
                raise ValueError('RAFT-Net with dct=True needs the native-grid DCT map as input')
            if dct.shape[-2:] != f8.shape[-2:]:
                dct = F.interpolate(dct, size=f8.shape[-2:], mode='nearest')
            f8 = f8 + self.dct_enc(dct.to(f8.dtype))
        f16 = self.d16(f8)
        return [f4, f8, f16, self.d32(f16)]


class FCSS(nn.Module):
    """Forensic-Conditioned Selective Scan. For every VMamba block, a ZERO-initialised 1x1 conv maps the stage's
    forensic features to a bias on the SSM step size (dt) at every position, bounded by SCALE*tanh(.). The patched
    SS2D.forward_corev2 adds it (cross-scanned into all 4 scan directions) before the selective scan. Zero init =>
    the network starts exactly as the ImageNet-pretrained VMamba; training learns how far forensic evidence
    should steer the scan."""
    SCALE = 2.0

    def __init__(self, backbone, in_widths):
        super().__init__()
        convs, self._ops = [], []
        for s, layer in enumerate(backbone.layers):
            for blk in layer.blocks:
                d_inner = blk.op.dt_projs_weight.shape[1]
                c = nn.Conv2d(in_widths[s], d_inner, 1)
                nn.init.zeros_(c.weight)
                nn.init.zeros_(c.bias)
                convs.append(c)
                self._ops.append((s, blk.op))
        self.convs = nn.ModuleList(convs)

    def set(self, feats):
        for (s, op), conv in zip(self._ops, self.convs):
            op.forensic_bias = self.SCALE * torch.tanh(conv(feats[s]))

    def clear(self):
        for _, op in self._ops:
            op.forensic_bias = None


class WideHead(SegFormerHead):
    """ForMa's pixel-shuffle decoder, with the forensic pathway widened (ForMa: 3 -> 16 channels next to 384 RGB
    channels). Same computation as SegFormerHead.forward; optionally also returns the forensic features (for ACR)."""
    def __init__(self, modal_in=64, modal_out=32, embedding_dim=96):
        super().__init__(embedding_dim=embedding_dim)
        self.conv_modals = nn.Conv2d(modal_in, modal_out, kernel_size=1)
        self.linear_fuse = ConvModule(c1=embedding_dim * 4 + modal_out, c2=embedding_dim, k=1)

    def forward(self, inputs, x_modals, return_feat=False):
        c1, c2, c3, c4 = inputs
        m = self.conv_modals(x_modals)
        n = c4.shape[0]
        _c4 = F.pixel_shuffle(self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3]), 8)
        _c3 = F.pixel_shuffle(self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3]), 4)
        _c2 = F.pixel_shuffle(self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3]), 2)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        out = self.linear_pred(self.dropout(self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, m], dim=1))))
        return (out, m) if return_feat else out


class BoundaryModule(nn.Module):
    """BASNet-style boundary-motivated module (from the RAFT-Net prototype, unchanged)."""
    def __init__(self, c_low1=96, c_low2=192, c_high=768, mid_channels=64):
        super().__init__()
        self.conv_f1 = nn.Sequential(nn.Conv2d(c_low1, mid_channels, 1), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.conv_f2 = nn.Sequential(nn.Conv2d(c_low2, mid_channels, 1), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.gate = nn.Sequential(nn.Conv2d(mid_channels * 2, mid_channels, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(mid_channels, mid_channels, 1))
        self.attn = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, 1), nn.Sigmoid())
        self.conv_high = nn.Sequential(nn.Conv2d(c_high, mid_channels, 1), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))

    def forward(self, f1, f2, f_high):
        hw = f1.shape[-2:]
        F1 = self.conv_f1(f1)
        F2 = self.conv_f2(F.interpolate(f2, size=hw, mode='bilinear', align_corners=True))
        FL = self.attn(self.gate(torch.cat([F1, F2], dim=1)))
        FH = self.conv_high(F.interpolate(f_high, size=hw, mode='bilinear', align_corners=True))
        return FL * FH


class EdgeEnhance(nn.Module):
    """BASNet-style boundary-enhanced module (from the RAFT-Net prototype, unchanged)."""
    def __init__(self, channels=64):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(4)
        self.atrous = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.swish = nn.SiLU(inplace=True)

    def forward(self, x):
        h, w = x.shape[-2:]
        p1 = F.interpolate(self.pool1(x), size=(h, w), mode='bilinear', align_corners=True)
        p2 = F.interpolate(self.pool2(x), size=(h, w), mode='bilinear', align_corners=True)
        return x + self.swish(self.atrous(x)) * torch.sigmoid(self.fuse(torch.cat([p1, p2], dim=1)))


class BoundaryBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.bmm = BoundaryModule()
        self.bem = EdgeEnhance(64)
        self.head = nn.Sequential(conv_bn_relu(64, 32, 3, 1), nn.Conv2d(32, 1, 1))

    def forward(self, c1, c2, c4, out_hw):
        b = self.head(self.bem(self.bmm(c1, c2, c4)))
        return F.interpolate(b, size=out_hw, mode='bilinear', align_corners=True)


# ── RAFT-Net ──────────────────────────────────────────────────────────────────────────────────
class RAFTNet(nn.Module):
    def __init__(self, np_weights, fcss=True, dct=True, fullres=True, boundary=True):
        super().__init__()
        self.cfg = dict(fcss=fcss, dct=dct, fullres=fullres, boundary=boundary)
        self.denorm = Denorm()
        self.backbone = Backbone_VSSM()
        self.stem = ForensicStem(np_weights, fullres=fullres, dct=dct)
        self.fcss = FCSS(self.backbone, self.stem.widths) if fcss else None
        self.decode = WideHead(modal_in=self.stem.widths[0], modal_out=32)
        self.boundary = BoundaryBranch() if boundary else None

    def train(self, mode=True):
        super().train(mode)
        self.stem.npp.noiseprint.eval()
        return self

    def forward(self, x, dct=None, return_feat=False):
        h, w = x.shape[-2:]
        feats = self.stem(self.denorm(x), dct)
        if self.fcss is not None:
            self.fcss.set(feats)
        try:
            y = self.backbone(x)
        finally:
            if self.fcss is not None:
                self.fcss.clear()
        out = {}
        if return_feat:
            out['mask'], out['feat'] = self.decode(y, feats[0], return_feat=True)
        else:
            out['mask'] = self.decode(y, feats[0])
        if self.boundary is not None:
            out['boundary'] = self.boundary(y[0], y[1], y[3], (h, w))
        return out


VARIANTS = {
    # name                  builder kwargs (None = official ForMa)                                    needs DCT input
    'forma':               (None,                                                                     False),
    'raftnet':             (dict(fcss=True,  dct=False, fullres=True,  boundary=True),                False),  # FINAL
    'raftnet_no_fcss':     (dict(fcss=False, dct=False, fullres=True,  boundary=True),                False),
    'raftnet_no_boundary': (dict(fcss=True,  dct=False, fullres=True,  boundary=False),               False),
    'raftnet_no_fullres':  (dict(fcss=True,  dct=False, fullres=False, boundary=True),                False),
    'raftnet_lite_nofullres': (dict(fcss=False, dct=False, fullres=False, boundary=True),             False),  # final candidate
    'forma_w12':           (dict(width=12),                                                            False),  # bottleneck study
    'forma_w48':           (dict(width=48),                                                            False),  # bottleneck study
}


def build_model(variant, np_weights):
    kw, _ = VARIANTS[variant]
    if kw is None:
        return ForMaOfficial(np_weights)
    if 'width' in kw:
        return ForMaWide(np_weights, kw['width'])
    return RAFTNet(np_weights, **kw)


def load_imagenet_backbone(model, sd, prefix='backbone.'):
    """ForMa train.py logic: copy checkpoint tensors into model[prefix + name] when shapes match."""
    msd = model.state_dict()
    n = 0
    for k, v in sd.items():
        mk = prefix + k
        if mk in msd and tuple(msd[mk].shape) == tuple(v.shape):
            msd[mk] = v
            n += 1
    model.load_state_dict(msd)
    bb = [k for k in msd if k.startswith(prefix)]
    return n, len(sd), len(bb)
