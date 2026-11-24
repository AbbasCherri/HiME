"""Synthetic oracle world with known likelihood ratios (Phase 2). Demonstrates the pipeline runs;
it does not reproduce the paper's results."""
import numpy as np
from scipy.stats import multivariate_normal as mvn

from .experts import sigmoid
from .rng import stream


def make_world(n=2000, N=50, d_s=7, d_i=16, d_d=0, sep_s=0.5, sep_i=0.35, rho=0.0, seed=0,
               extent=40_000.0, direct_frac=0.2, avail="beta", calib=0.2, test_frac=0.2, block=2000.0):
    g = stream(seed, "oracle")
    xy = g.uniform(0, extent, (n, 2))
    epi = np.array([0.3, 0.5]) * extent
    dist = np.hypot(*(xy - epi).T) / extent
    prev = sigmoid(-1.0 - 6.0 * dist)                         # regional prevalence, class-conditionals fixed
    y = (g.random(n) < prev).astype(float)
    mS, mI, mD = np.full(d_s, sep_s), np.full(d_i, sep_i), np.full(d_d, 0.5)
    xS = g.normal(size=(n, d_s)) + y[:, None] * mS
    xI = g.normal(size=(n, d_i)) + y[:, None] * mI
    xD = g.normal(size=(n, d_d)) + y[:, None] * mD
    if rho:                                                   # violates assumption (a) in class 1
        u = g.normal(size=n)
        xS[:, 0] += rho * u * y
        xI[:, 0] += rho * u * y
    pG = np.array([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])         # P(pattern | y): none, I, IP
    pattern = np.array([g.choice(3, p=pG[int(t)]) for t in y]) if d_d else np.ones(n, int)
    # spatial blocks -> test fold
    bid = (xy // block).astype(int) @ np.array([1, 10_000])
    ub = np.unique(bid)
    test_blocks = ub[g.random(len(ub)) < test_frac]
    test = np.isin(bid, test_blocks)
    # clients: nearest of N centres among training buildings
    tr = np.flatnonzero(~test)
    centres = xy[g.choice(tr, size=N, replace=False)]
    lab = np.argmin(((xy[tr, None] - centres[None]) ** 2).sum(-1), 1)
    client = np.full(n, -1); client[tr] = lab
    keep = np.unique(lab)                                     # drop empty clients, reindex
    remap = -np.ones(N, int); remap[keep] = np.arange(len(keep))
    client[tr] = remap[lab]
    N = len(keep)
    is_cal = (~test) & (g.random(n) < calib)
    cxy = np.array([xy[client == c][np.argmin(np.hypot(*(xy[client == c] - xy[client == c].mean(0)).T))] for c in range(N)])
    shake = -np.hypot(*(cxy - epi).T)                         # proxy for medoid MMI
    direct = shake <= np.quantile(shake, direct_frac) if direct_frac > 0 else np.zeros(N, bool)
    if avail == "beta":
        alpha = g.beta(4, 2, size=N)
    else:                                                     # damage-correlated, mean ~ 2/3
        cp = np.array([prev[client == c].mean() for c in range(N)])
        z = (cp - cp.mean()) / (cp.std() + 1e-12)
        alpha = sigmoid(np.log(2) - 1.0 * z)
    return dict(xy=xy, y=y, xS=xS, xI=xI, xD=xD, pattern=pattern, prev=prev, test=test, client=client,
                calib=is_cal, client_xy=cxy, direct=direct, alpha=alpha, N=N, extent=extent,
                params=dict(mS=mS, mI=mI, mD=mD, rho=rho, pG=pG))


def llr_gauss(x, m):
    """log N(x; m, I) - log N(x; 0, I)."""
    return x @ m - 0.5 * m @ m


def llr_joint(w, i):
    """Exact joint log-likelihood ratio of (xS, xI) for rows i, including the rho correlation."""
    p = w["params"]; mS, mI, rho = p["mS"], p["mI"], p["rho"]
    x = np.c_[w["xS"][i], w["xI"][i]]
    d = len(mS) + len(mI)
    S1 = np.eye(d)
    S1[0, 0] += rho ** 2; S1[len(mS), len(mS)] += rho ** 2
    S1[0, len(mS)] = S1[len(mS), 0] = rho ** 2
    return mvn.logpdf(x, np.r_[mS, mI], S1) - mvn.logpdf(x, np.zeros(d), np.eye(d))
