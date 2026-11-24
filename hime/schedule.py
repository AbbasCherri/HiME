"""Routing (Prop 4'), sync periods kappa_m and placement period T_p (plan 2.7, 2.10)."""
import math

import numpy as np


def u_fed(P, participations, K, T, kappa):
    return 2 * float(np.sum(participations)) * P + 2 * K * P * T / kappa


def federate(P, participations, K, T, raw_bytes_one_link):
    """Prop 4': U_fed is non-increasing in kappa, so testing kappa = 1 decides for every kappa."""
    return u_fed(P, participations, K, T, 1) <= 2 * raw_bytes_one_link


def kappas(P, drift, K, budget, p=2, kmax=100):
    """Assumed form of design v1 Prop 7: minimise sum_m drift_m kappa_m^p subject to the per-round
    backhaul load sum_m 2 K P_m / kappa_m <= budget. Lagrangian solution, rounded up (feasible)."""
    P, drift = np.asarray(P, float), np.asarray(drift, float)
    b = 2 * K * P
    if b.sum() <= budget:
        return np.ones(len(P), int)
    # kappa_m = (lam b_m / (p drift_m))^(1/(p+1)); pick lam so sum b/kappa = budget (monotone in lam)
    def load(lam):
        return np.sum(b / np.maximum(1.0, (lam * b / (p * drift)) ** (1 / (p + 1))))
    lo, hi = 0.0, 1.0
    while load(hi) > budget:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if load(mid) > budget else (lo, mid)
    k = np.ceil(np.maximum(1.0, (hi * b / (p * drift)) ** (1 / (p + 1))) - 1e-9).astype(int)
    return np.minimum(k, kmax)


def placement_period(kappa):
    """Smallest common multiple of all kappa_m that is at least max(5, 2 max kappa_m)."""
    L = math.lcm(*[int(k) for k in kappa])
    target = max(5, 2 * max(kappa))
    return L * math.ceil(target / L)
