# losses.py
from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCE(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.smoothing = float(smoothing)
    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        B, C = log_probs.shape
        if self.smoothing > 0.0:
            with torch.no_grad():
                true_dist = torch.full_like(log_probs, self.smoothing / (C - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            loss = -(true_dist * log_probs)
            if self.weight is not None:
                w = self.weight[target].unsqueeze(1)
                loss = loss * w
            return loss.sum(dim=1).mean()
        else:
            return F.nll_loss(log_probs, target, weight=self.weight, reduction='mean')

def top1_acc(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()

def top5_acc(logits, targets):
    with torch.no_grad():
        k = min(5, logits.size(1))
        _, idx = logits.topk(k, dim=1)
        return (idx == targets.unsqueeze(1)).any(dim=1).float().mean().item()

def confusion_and_hist(logit_batches, targ_batches, class_names):
    preds = np.concatenate([lb.argmax(1) for lb in logit_batches], axis=0)
    targs = np.concatenate(targ_batches, axis=0)
    C = len(class_names)
    cm = np.zeros((C, C), dtype=int)
    for p, t in zip(preds, targs):
        cm[t, p] += 1
    per_acc = {class_names[c]: (cm[c, c] / max(1, cm[c].sum())) for c in range(C)}
    from collections import Counter as Ctr
    hist = Ctr(preds.tolist())
    return cm, per_acc, hist

def macro_f1_from_cm(cm):
    C = cm.shape[0]
    f1s = []
    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1s.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s))

