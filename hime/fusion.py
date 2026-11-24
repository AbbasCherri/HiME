"""Fusion: general multi-pattern rule (Prop 2) and the 4-parameter stacker (plan 2.4, Prop 3')."""
import numpy as np
from scipy.optimize import minimize

from .experts import sigmoid, softplus

PATTERN = {"none": 0, "I": 1, "IP": 2}


def pattern_logit(logit_pi, gS, gI, gD, pattern, delta):
    """pattern: int codes (0 none, 1 I, 2 IP); delta[code] = log P(G|y=1)/P(G|y=0)."""
    return logit_pi + np.asarray(delta)[pattern] + gS + np.where(pattern >= 1, gI, 0.0) + np.where(pattern == 2, gD, 0.0)


def _features(gS, gI, interaction):
    F = np.c_[np.ones_like(gS), gS, gI, gS * gI]
    return F if interaction else F[:, :3]


def stacker_objective(p, gS, gI, y, logit_pi, gamma, interaction=True):
    F = _features(gS, gI, interaction)
    p0 = np.array([0.0, 1.0, 1.0, 0.0])[: F.shape[1]]
    z = logit_pi + F @ p
    d = p - p0
    f = np.mean(softplus(z) - y * z) + 0.5 * gamma * d @ d
    g = F.T @ (sigmoid(z) - y) / len(y) + gamma * d
    return f, g


def fit_stacker(gS, gI, y, logit_pi, gamma=None, interaction=True, p_init=None):
    """Returns (c, wS, wI[, wSI]) minimising calibration log-loss + ridge toward (0,1,1,0); gamma = 10/|V|."""
    gamma = 10.0 / len(y) if gamma is None else gamma
    k = 4 if interaction else 3
    p_init = np.array([0.0, 1.0, 1.0, 0.0])[:k] if p_init is None else p_init
    bounds = [(None, None), (0, None), (0, None), (None, None)][:k]
    r = minimize(stacker_objective, p_init, args=(gS, gI, y, logit_pi, gamma, interaction), jac=True,
                 method="L-BFGS-B", bounds=bounds, options=dict(pgtol=1e-10, ftol=1e-15, maxiter=10000))
    lower = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
    return _newton_polish(r.x, gS, gI, y, logit_pi, gamma, interaction, lower)


def _newton_polish(p, gS, gI, y, logit_pi, gamma, interaction, lower, iters=50):
    """Projected Newton on the (strongly convex, <= 4-parameter) stacker objective: L-BFGS-B can stop
    early on flat ridges; this drives every start to the unique minimiser to machine precision."""
    F = _features(gS, gI, interaction)
    obj = lambda q: stacker_objective(q, gS, gI, y, logit_pi, gamma, interaction)
    for _ in range(iters):
        f, g = obj(p)
        free = ~((p <= lower + 1e-12) & (g > 0))
        if not free.any() or np.abs(g[free]).max() < 1e-13:
            break
        w = sigmoid(logit_pi + F @ p); w = w * (1 - w)
        H = F.T @ (F * w[:, None]) / len(y) + gamma * np.eye(len(p))
        d = np.zeros_like(p)
        d[free] = -np.linalg.solve(H[np.ix_(free, free)], g[free])
        t = 1.0
        while True:
            q = np.maximum(p + t * d, lower)
            if obj(q)[0] <= f + 1e-4 * g @ (q - p) or t < 1e-12:
                break
            t /= 2
        if t < 1e-12:
            break
        p = q
    return p


def stacker_logit(p, gS, gI, logit_pi):
    return logit_pi + _features(gS, gI, len(p) == 4) @ p
