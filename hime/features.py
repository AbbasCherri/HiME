"""I-spline basis, histogram knots, exact federated standardisation and PCA (plan 2.3)."""
import numpy as np
from scipy.interpolate import BSpline


def ispline_knots(lo, hi, interior):
    inner = np.unique(np.asarray(interior, float))
    inner = inner[(inner > lo) & (inner < hi)]          # merges duplicate / boundary quartiles
    return np.r_[[lo] * 3, inner, [hi] * 3]


def ispline_basis(x, t):
    """Quadratic I-splines: I_i = sum_{j>=i} B_j (quadratic B-splines); values in [0,1], monotone.
    The constant I_0 = 1 is dropped (the intercept carries it)."""
    x = np.clip(np.asarray(x, float), t[0], t[-1])
    B = BSpline.design_matrix(x, t, 2).toarray()
    return np.cumsum(B[:, ::-1], axis=1)[:, ::-1][:, 1:]


def client_range(x):
    return float(np.min(x)), float(np.max(x))


def client_hist(x, lo, hi, bins=1000):
    return np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi))[0]


def hist_quantiles(counts, lo, hi, qs=(0.25, 0.5, 0.75)):
    """Quantiles from pooled histograms; each is the upper edge of the bin holding the exact quantile."""
    edges = np.linspace(lo, hi, len(counts) + 1)
    c = np.cumsum(counts)
    return np.array([edges[min(np.searchsorted(c, q * c[-1]), len(counts) - 1) + 1] for q in qs])


def fed_knots(client_cols, bins=1000):
    """client_cols: list of per-client 1-D arrays of one feature -> I-spline knot vector."""
    rng_ = [client_range(x) for x in client_cols if len(x)]
    lo, hi = min(r[0] for r in rng_), max(r[1] for r in rng_)
    counts = sum(client_hist(x, lo, hi, bins) for x in client_cols)
    return ispline_knots(lo, hi, hist_quantiles(counts, lo, hi))


def suff_stats(Z):
    return Z.sum(0), Z.T @ Z, len(Z)


def fed_standardise_pca(stats, r=None):
    s = sum(x[0] for x in stats); S = sum(x[1] for x in stats); n = sum(x[2] for x in stats)
    mu = s / n
    C = S / n - np.outer(mu, mu)
    sd = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    w, V = np.linalg.eigh(C / np.outer(sd, sd))
    V = V[:, ::-1]
    V = V * np.sign(V[np.abs(V).argmax(0), np.arange(V.shape[1])])   # deterministic signs
    return mu, sd, (V if r is None else V[:, :r])


def transform(Z, mu, sd, P):
    return ((Z - mu) / sd) @ P
