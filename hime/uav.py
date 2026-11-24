"""UAV placement (plan 2.8): Al-Hourani radius, capped altitude, exact candidates, movement-aware
weighted coverage, greedy over the partition matroid, multi-start local search, exact MILP."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, brentq, linear_sum_assignment, milp, minimize_scalar
from scipy.sparse import lil_matrix

C0 = 299_792_458.0
# (a, b, eta_LoS dB, eta_NLoS dB). Values as commonly quoted from Al-Hourani et al. 2014;
# to be checked against the full text before paper use (plan 2.9).
ENV = {"suburban": (4.88, 0.43, 0.1, 21.0), "urban": (9.61, 0.16, 1.0, 20.0)}


def pathloss(h, d, f=2e9, env="suburban"):
    a, b, eL, eN = ENV[env]
    R = np.hypot(h, d)
    th = np.degrees(np.arctan2(h, d))
    P = 1.0 / (1.0 + a * np.exp(-b * (th - a)))
    return 20 * np.log10(4 * np.pi * f * R / C0) + eL * P + eN * (1 - P)


def radius(h, L=117.0, **kw):
    """Largest ground distance with pathloss <= L (L = L_max minus margin). Pathloss rises with d."""
    g = lambda d: pathloss(h, d, **kw) - L
    if g(0.0) > 0:
        return 0.0
    hi = 100.0
    while g(hi) <= 0 and hi < 1e6:
        hi *= 2
    return brentq(g, 0.0, hi, xtol=1e-6)


def best_altitude(hmin=10.0, hmax=150.0, **kw):
    """Lemma 7': r(h) unimodal, so the capped maximiser is found by bounded 1-D search."""
    r = minimize_scalar(lambda h: -radius(h, **kw), bounds=(hmin, hmax), method="bounded",
                        options=dict(xatol=1e-4))
    h = float(r.x)
    for e in (hmin, hmax):                   # bounded search never evaluates endpoints exactly
        if radius(e, **kw) > radius(h, **kw):
            h = e
    return h, radius(h, **kw)


def _circle_x(c1, r1, c2, r2):
    """Intersections of circles (c1[k], r1[k]) and (c2[k], r2[k]), vectorised."""
    d = np.hypot(*(c2 - c1).T)
    ok = (d > 0) & (d <= r1 + r2) & (d >= np.abs(r1 - r2))
    c1, c2, d, r1, r2 = c1[ok], c2[ok], d[ok], np.broadcast_to(r1, ok.shape)[ok], np.broadcast_to(r2, ok.shape)[ok]
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(np.maximum(r1 ** 2 - a ** 2, 0))
    u = (c2 - c1) / d[:, None]
    m = c1 + a[:, None] * u
    perp = np.c_[-u[:, 1], u[:, 0]]
    return np.vstack([m + h[:, None] * perp, m - h[:, None] * perp])


def candidates(clients, r, origin, dmax):
    """Lemma 8 candidate set inside the movement disc. Row 0 is the current position (stay).
    Includes client locations, pairwise client-circle intersections and client-circle/disc intersections."""
    origin = np.asarray(origin, float)
    near = clients[np.hypot(*(clients - origin).T) <= dmax + r]
    pts = [origin[None], near]
    if len(near) > 1:
        i, j = np.triu_indices(len(near), 1)
        pts.append(_circle_x(near[i], r, near[j], r))
    if np.isfinite(dmax) and len(near):
        pts.append(_circle_x(near, r, np.broadcast_to(origin, near.shape), dmax))
    P = np.vstack(pts)
    keep = np.hypot(*(P - origin).T) <= dmax * (1 + 1e-9) + 1e-9
    keep[0] = True
    return P[keep]


def coverage(points, clients, r):
    return np.hypot(points[:, None, 0] - clients[None, :, 0], points[:, None, 1] - clients[None, :, 1]) <= r * (1 + 1e-9) + 1e-9


def f_pairs(pairs, covs, w, Tp):
    """Movement-aware objective on an arbitrary set of (uav, candidate) pairs: f_stay + (Tp-1) f_all."""
    n = len(w)
    allc = np.zeros(n, bool); stay = np.zeros(n, bool)
    for j, c in pairs:
        allc |= covs[j][c]
        if c == 0:
            stay |= covs[j][0]
    return float((Tp - 1) * w @ allc + w @ stay)


def f_sel(sel, covs, w, Tp):
    return f_pairs([(j, c) for j, c in enumerate(sel) if c >= 0], covs, w, Tp)


def greedy(covs, w, Tp):
    K, n = len(covs), len(w)
    sel = [-1] * K
    allc = np.zeros(n, bool); stay = np.zeros(n, bool)
    for _ in range(K):
        best = (-np.inf, None, None)
        for j in range(K):
            if sel[j] >= 0:
                continue
            gain = (Tp - 1) * ((covs[j] & ~allc) @ w)
            gain[0] += (covs[j][0] & ~stay) @ w
            c = int(np.argmax(gain))
            if gain[c] > best[0]:
                best = (gain[c], j, c)
        _, j, c = best
        sel[j] = c
        allc |= covs[j][c]
        if c == 0:
            stay |= covs[j][0]
    return sel


def local_search(sel, covs, w, Tp, max_pass=1000):
    """Single-UAV moves accepted only on strict improvement; terminates (finite, strictly increasing)."""
    sel = list(sel); K, n = len(covs), len(w)
    cur = f_sel(sel, covs, w, Tp)
    for _ in range(max_pass):
        improved = False
        for j in range(K):
            oa = np.zeros(n, bool); os_ = np.zeros(n, bool)
            for k, c in enumerate(sel):
                if k != j:
                    oa |= covs[k][c]
                    if c == 0:
                        os_ |= covs[k][0]
            vals = (Tp - 1) * ((covs[j] | oa) @ w) + w @ os_
            vals[0] += w @ ((covs[j][0] | os_) & ~os_)
            c = int(np.argmax(vals))
            if vals[c] > cur + 1e-12 * max(1.0, abs(cur)):
                sel[j], cur, improved = c, float(vals[c]), True
        if not improved:
            break
    return sel


def kmeans(X, k, w, rng, iters=50):
    k = min(k, len(X))
    C = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        for c in range(k):
            m = lab == c
            if m.any() and w[m].sum() > 0:
                C[c] = (w[m, None] * X[m]).sum(0) / w[m].sum()
    return C


def kmeans_start(cands, uav_xy, clients, w, covered, rng):
    pool = ~covered if (~covered).sum() >= len(uav_xy) else np.ones(len(clients), bool)
    C = kmeans(clients[pool], len(uav_xy), w[pool] + 1e-12, rng)
    rows, cols = linear_sum_assignment(np.hypot(uav_xy[:, None, 0] - C[None, :, 0], uav_xy[:, None, 1] - C[None, :, 1]))
    sel = [0] * len(uav_xy)
    for j, c in zip(rows, cols):
        sel[j] = int(np.argmin(np.hypot(*(cands[j] - C[c]).T)))
    return sel


def place(clients, w, uav_xy, r, dmax, Tp, rng):
    """Greedy, then local search from {greedy, static, k-means}; keep the best (Prop 9)."""
    cands = [candidates(clients, r, o, dmax) for o in uav_xy]
    covs = [coverage(P, clients, r) for P in cands]
    static = [0] * len(uav_xy)
    covered0 = np.any([cv[0] for cv in covs], axis=0) if covs else np.zeros(len(clients), bool)
    starts = {"greedy": greedy(covs, w, Tp), "static": static,
              "kmeans": kmeans_start(cands, uav_xy, clients, w, covered0, rng)}
    info = {k: f_sel(s, covs, w, Tp) for k, s in starts.items()}
    final = {k: local_search(s, covs, w, Tp) for k, s in starts.items()}
    fin = {k: f_sel(s, covs, w, Tp) for k, s in final.items()}
    best = max(fin, key=lambda k: (fin[k], k == "greedy"))
    sel = final[best]
    pos = np.array([cands[j][c] for j, c in enumerate(sel)])
    return pos, np.array(sel), {"start_f": info, "final_f": fin, "best": best, "covs": covs, "cands": cands}


def exact_opt(covs, w, Tp):
    """Exact optimum by MILP (HiGHS). Variables: x_{j,c} (binary), y_i (covered), z_i (covered by a stayer)."""
    K, n = len(covs), len(w)
    off = np.cumsum([0] + [len(c) for c in covs])
    nx = off[-1]
    nv = nx + 2 * n
    c = np.r_[np.zeros(nx), -(Tp - 1) * w, -w]
    A = lil_matrix((K + 2 * n, nv)); lb = np.zeros(K + 2 * n); ub = np.zeros(K + 2 * n)
    for j in range(K):
        A[j, off[j]:off[j + 1]] = 1; lb[j] = ub[j] = 1
        for i in np.flatnonzero(covs[j][0]):
            A[K + n + i, off[j]] = -1
        jj, ii = np.nonzero(covs[j])
        for a_, i in zip(jj, ii):
            A[K + i, off[j] + a_] += -1
    for i in range(n):
        A[K + i, nx + i] = 1; A[K + n + i, nx + n + i] = 1
        lb[K + i] = lb[K + n + i] = -np.inf
    integ = np.r_[np.ones(nx), np.zeros(2 * n)]
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=integ, bounds=Bounds(0, 1))
    x = res.x[:nx]
    sel = [int(np.argmax(x[off[j]:off[j + 1]])) for j in range(K)]
    return f_sel(sel, covs, w, Tp), sel


def calibrate_K(clients, r, target=0.70, Ks=range(4, 31, 2)):
    """Smallest even K whose greedy (unconstrained movement) coverage reaches `target` of clients."""
    P = candidates(clients, r, clients.mean(0), np.inf)
    cov = np.vstack([np.zeros((1, len(clients)), bool), coverage(P, clients, r)])  # dummy row 0: no stay bonus
    w = np.ones(len(clients))
    for K in Ks:
        sel = greedy([cov] * K, w, 2)
        if np.any(cov[sel], axis=0).mean() >= target:
            return K
    return max(Ks)
