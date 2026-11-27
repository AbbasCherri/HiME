"""Turn a world (oracle now; real data after Phase 1) into federated design matrices."""
import numpy as np

from .experts import LinearExpert, logit
from .features import fed_knots, fed_standardise_pca, ispline_basis, suff_stats, transform


def prepare(w, r=None, lam=1e-4, bins=1000, monotone=6):
    train = (w["client"] >= 0) & ~w["calib"]
    rows = [np.flatnonzero(train & (w["client"] == c)) for c in range(w["N"])]
    xS, xI = w["xS"], w["xI"]
    knots = [fed_knots([xS[r_, f] for r_ in rows], bins) for f in range(xS.shape[1])]
    blocks = [ispline_basis(xS[:, f], t) for f, t in enumerate(knots)]
    B = np.hstack(blocks)
    # Exact federated column standardisation of the I-spline block: positive rescaling keeps beta >= 0
    # equivalent to monotonicity, and removes the ill-conditioning that slows federated GD.
    smu, ssd, _ = fed_standardise_pca([suff_stats(B[r_]) for r_ in rows if len(r_)])
    XS = np.c_[np.ones(len(xS)), (B - smu) / ssd]
    nonneg = np.concatenate([[False]] + [np.full(b.shape[1], f < monotone) for f, b in enumerate(blocks)])
    mu, sd, P = fed_standardise_pca([suff_stats(xI[r_]) for r_ in rows if len(r_)], r)
    XI = np.c_[np.ones(len(xI)), transform(xI, mu, sd, P)]
    if "flag" in w:                                           # imagery-audit flag: pre-event-looking chip
        XI = np.c_[XI, w["flag"]]
    pi = w["y"][train].mean()                                 # equals pooled client label counts
    lp = logit(pi)
    experts = {"S": LinearExpert(XS.shape[1], lp, lam, nonneg), "I": LinearExpert(XI.shape[1], lp, lam)}
    return dict(experts=experts, X={"S": XS, "I": XI}, y=w["y"], client_rows=rows,
                calib_rows=np.flatnonzero(w["calib"] & (w["client"] >= 0)), test_rows=np.flatnonzero(w["test"]),
                client_xy=w["client_xy"], direct=w["direct"], alpha=w["alpha"], logit_pi=lp, knots=knots)
