# Part of RAFT-Net. The notebooks write this file into the models/ folder of a ForMa clone
# (https://github.com/multimediaFor/ForMa) and import it from there.

# ── Unified metric engine ─────────────────────────────────────────────────────────
# Pure numpy, histogram-based: every pixel's tampered-probability is binned into 1000 bins, so
# counts at ANY threshold k/1000 are exact cumulative sums -- no pixel subsampling, bounded
# memory regardless of image size, and pixel ROC-AUC / AP computed on ALL pixels.
#
# Threshold semantics: a pixel is predicted tampered at threshold t=k/1000 iff prob >= t,
# which is exactly floor(prob*1000) >= k. At k=500 this is the standard fixed 0.5 threshold.
#
# Reported per dataset (tampered images WITH a ground-truth mask only, the Protocol-1
# convention used by MPC / PSCC-Net / TruFor tables):
#   F1/IoU/P/R per-image @0.5, averaged over images   <- HEADLINE (matches MPC Table II/III)
#   F1/IoU pooled over all pixels @0.5
#   F1/IoU pooled at the dataset-best threshold       (optimistic: threshold chosen on test data)
#   F1 per-image at each image's own best threshold   (oracle upper bound, clearly labelled)
#   pixel ROC-AUC and AP (all pixels of tampered images)
# plus image-level detection AUC (max-pooled prob), using authentic images as negatives.
import numpy as np
from sklearn.metrics import roc_auc_score

N_BINS = 1000
K_05 = 500  # bin index for threshold 0.5


def _safe_div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.zeros(np.broadcast(a, b).shape, dtype=np.float64)
    np.divide(a, b, out=out, where=b > 0)
    return out


class MetricAccumulator:
    def __init__(self, name):
        self.name = name
        self.hist_pos = np.zeros(N_BINS, dtype=np.int64)   # GT-tampered pixels, all tampered images
        self.hist_neg = np.zeros(N_BINS, dtype=np.int64)   # GT-authentic pixels of tampered images
        self.rows = []            # per tampered image with a usable mask
        self.img_labels = []      # image-level: 1 tampered, 0 authentic
        self.img_scores = []      # image-level score: max prob
        self.n_empty_gt = 0       # "tampered" images whose mask has zero tampered pixels
        self.n_no_mask = 0        # tampered images with no mask file (image-level only)

    def add(self, prob, gt, label, key=None):
        """prob: HxW float in [0,1] at the SAME resolution as gt.
        gt: HxW {0,1} array, or None (authentic image, or tampered image without a mask).
        label: 1 tampered / 0 authentic (image level)."""
        prob = np.asarray(prob, dtype=np.float32)
        self.img_labels.append(int(label))
        self.img_scores.append(float(prob.max()))
        if gt is None:
            if label == 1:
                self.n_no_mask += 1
            return
        g = np.asarray(gt).astype(bool)
        assert g.shape == prob.shape, f"shape mismatch prob {prob.shape} vs gt {g.shape}"
        n_pos = int(g.sum())
        if n_pos == 0:
            self.n_empty_gt += 1
            return
        idx = np.minimum((prob * N_BINS).astype(np.int64), N_BINS - 1)
        hp = np.bincount(idx[g], minlength=N_BINS)
        hn = np.bincount(idx[~g], minlength=N_BINS)
        self.hist_pos += hp
        self.hist_neg += hn
        tp = np.cumsum(hp[::-1])[::-1]          # tp[k]: GT-pos pixels with prob >= k/1000
        fp = np.cumsum(hn[::-1])[::-1]
        fn = n_pos - tp
        f1_all = _safe_div(2 * tp, 2 * tp + fp + fn)
        TP, FP, FN = int(tp[K_05]), int(fp[K_05]), int(fn[K_05])
        self.rows.append({
            'key': key,
            'f1': float(_safe_div(2 * TP, 2 * TP + FP + FN)),
            'iou': float(_safe_div(TP, TP + FP + FN)),
            'precision': float(_safe_div(TP, TP + FP)),
            'recall': float(_safe_div(TP, TP + FN)),
            'f1_best_own_thr': float(f1_all.max()),
            'gt_frac': n_pos / g.size,
            'f1_all_tampered': 2 * n_pos / (2 * n_pos + (g.size - n_pos)),   # trivial baseline: every pixel predicted tampered
            'pred_frac': (TP + FP) / g.size,
            'tp': TP, 'fp': FP, 'fn': FN,
        })

    def summary(self):
        s = {'dataset': self.name, 'n_images_total': len(self.img_labels),
             'n_tampered_scored': len(self.rows), 'n_empty_gt_skipped': self.n_empty_gt,
             'n_tampered_without_mask': self.n_no_mask,
             'n_authentic': int(sum(1 for l in self.img_labels if l == 0))}
        if self.rows:
            for k in ['f1', 'iou', 'precision', 'recall', 'f1_best_own_thr', 'gt_frac', 'pred_frac', 'f1_all_tampered']:
                s[f'img_mean_{k}'] = float(np.mean([r[k] for r in self.rows]))
            TP = sum(r['tp'] for r in self.rows)
            FP = sum(r['fp'] for r in self.rows)
            FN = sum(r['fn'] for r in self.rows)
            s['pooled_f1@0.5'] = float(_safe_div(2 * TP, 2 * TP + FP + FN))
            s['pooled_iou@0.5'] = float(_safe_div(TP, TP + FP + FN))
            tp = np.cumsum(self.hist_pos[::-1])[::-1]
            fp = np.cumsum(self.hist_neg[::-1])[::-1]
            fn = self.hist_pos.sum() - tp
            f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
            kb = int(np.argmax(f1))
            s['pooled_best_thr'] = kb / N_BINS
            s['pooled_f1@best'] = float(f1[kb])
            s['pooled_iou@best'] = float(_safe_div(tp[kb], tp[kb] + fp[kb] + fn[kb]))
            n_pos, n_neg = self.hist_pos.sum(), self.hist_neg.sum()
            if n_pos > 0 and n_neg > 0:
                # ROC-AUC = P(score_pos > score_neg) + 0.5 * P(tie within a bin)
                neg_below = np.concatenate([[0], np.cumsum(self.hist_neg)[:-1]])
                s['pixel_roc_auc'] = float((self.hist_pos * (neg_below + 0.5 * self.hist_neg)).sum()
                                           / (float(n_pos) * float(n_neg)))
                # AP = sum over thresholds (high -> low) of precision * delta-recall
                prec = _safe_div(tp, tp + fp)
                rec = tp / n_pos
                rec_next = np.concatenate([rec[1:], [0.0]])
                s['pixel_ap'] = float(((rec - rec_next) * prec).sum())
        labels = np.array(self.img_labels)
        if len(set(labels.tolist())) == 2:
            s['image_auc_maxpool'] = float(roc_auc_score(labels, np.array(self.img_scores)))
        return s
