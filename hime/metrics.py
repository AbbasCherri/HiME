"""AUPRC (average precision with tie groups, as scikit-learn) and ECE."""
import numpy as np


def average_precision(y, s):
    o = np.argsort(-np.asarray(s), kind="stable")
    y, s = np.asarray(y, float)[o], np.asarray(s)[o]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[last]
    prec = tp / (last + 1)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def ece(y, p, bins=15):
    idx = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(np.mean(y[m]) - np.mean(p[m]))
    return float(tot)
