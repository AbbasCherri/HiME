"""Participation (plan 2.6): importance, allocation, inclusion probabilities, Sampford, HT tools."""
from itertools import combinations

import numpy as np


def importance(ghat, wt, alpha):
    """ghat, wt: [M, n]; alpha: [n].  a_i = (sum_m psi_m (wt ghat / alpha)^2)^(1/2)."""
    b = wt * ghat / alpha
    tot = b.sum(1, keepdims=True)
    psi = np.where(tot > 0, 1.0 / np.where(tot > 0, tot, 1.0) ** 2, 0.0)
    return np.sqrt((psi * b ** 2).sum(0))


def allocate(A, cap, M):
    """Minimise sum A_j^2/s_j s.t. 1 <= s_j <= cap_j on non-empty strata, sum s = min(M, sum cap)."""
    A, cap = np.asarray(A, float), np.asarray(cap, int)
    s = np.zeros(len(A), int)
    open_ = np.flatnonzero(cap > 0)
    M = int(min(M, cap.sum()))
    if M < len(open_):                                     # not enough units for one per stratum
        s[open_[np.argsort(-A[open_], kind="stable")[:M]]] = 1
        return s
    s[open_] = 1
    while s.sum() < M:
        score = np.where(s < cap, A ** 2 / np.maximum(s * (s + 1), 1), -np.inf)
        s[int(np.argmax(score))] += 1
    return s


def inclusion_probs(a, s, eps=0.1):
    """pi*_i = min(1, lam a_i) with sum = s (water-filling), then mix with uniform s/n."""
    a = np.asarray(a, float); n = len(a)
    if s >= n:
        return np.ones(n)
    if s <= 0:
        return np.zeros(n)
    pi = np.zeros(n); free = np.ones(n, bool)
    while True:
        k = s - pi[~free].sum()
        af = a[free].sum()
        if af <= 0:
            pi[free] = k / free.sum()
            break
        cand = k * a / af
        over = free & (cand >= 1.0)
        if not over.any():
            pi[free] = cand[free]
            break
        pi[over] = 1.0
        free &= ~over
    return (1 - eps) * pi + eps * s / n


def sampford(pi, rng, max_tries=1_000_000):
    """Sampford's rejective method: sum(pi) distinct units with inclusion probabilities exactly pi."""
    pi = np.asarray(pi, float)
    cert = pi >= 1 - 1e-12
    idx = np.flatnonzero(~cert & (pi > 0))
    taken = np.flatnonzero(cert)
    p = pi[idx]
    n = int(round(p.sum()))
    if n == 0:
        return taken
    if n == len(idx):
        return np.sort(np.r_[taken, idx])
    q1 = p / n
    lam = p / (1 - p); q2 = lam / lam.sum()
    for _ in range(max_tries):
        draw = np.r_[rng.choice(len(p), p=q1), rng.choice(len(p), size=n - 1, p=q2)]
        if len(np.unique(draw)) == n:
            return np.sort(np.r_[taken, idx[draw]]).astype(int)
    raise RuntimeError("Sampford rejection did not terminate")


def sampford_design(pi):
    """Exact design p(S) ∝ sum_{k in S}(1-pi_k) prod_{k in S} pi_k/(1-pi_k), for 0 < pi < 1."""
    pi = np.asarray(pi, float); n = int(round(pi.sum()))
    lam = pi / (1 - pi)
    sets = list(combinations(range(len(pi)), n))
    w = np.array([np.prod(lam[list(S)]) * (1 - pi[list(S)]).sum() for S in sets])
    return sets, w / w.sum()


def joint_inclusion(sets, probs, N):
    P = np.zeros((N, N))
    for S, p in zip(sets, probs):
        S = list(S)
        P[np.ix_(S, S)] += p
    return P


def ht_variance(pi, pij, V):
    """Exact E||sum_{S} v_i/pi_i - sum v_i||^2 from joint inclusion probabilities (Lemma 2')."""
    G = V @ V.T
    return float(np.sum((pij - np.outer(pi, pi)) / np.outer(pi, pi) * G))


def multinomial_variance(p, s, V):
    """With-replacement (Hansen-Hurwitz) estimator variance, s draws with probabilities p."""
    tot = V.sum(0)
    return float(((V ** 2).sum(1) / p).sum() - tot @ tot) / s


def ema_floor(x, hit, rate, floor):
    return np.maximum(floor, (1 - rate) * x + rate * hit)
