"""Hierarchical FL engine for HiME-2 and its controls (plan 2.5-2.8, 2.11)."""
import math
from dataclasses import dataclass

import numpy as np

from .fusion import fit_stacker, stacker_logit
from .metrics import average_precision
from .participation import allocate, ema_floor, importance, inclusion_probs, sampford
from .rng import stream
from .schedule import placement_period
from .uav import best_altitude, place


@dataclass
class Config:
    T: int = 100
    M: int = 10
    C: int | None = None
    tau: int = 5
    eta_l0: float = 1.0
    eta_e: float = 1.0
    eps: float = 0.1
    kappa: tuple = (1, 1)            # per expert, in order of pb["experts"]
    Tp: int | None = None
    correction: bool = True
    L: float = 123.0                 # path-loss limit chosen by the coverage sweep (plan: 117)
    env: str = "suburban"
    hmin: float = 10.0
    hmax: float = 150.0
    dmax: float = 6000.0             # 20 m/s x 300 s
    sampler: str = "sampford"        # sampford | v1 | uniform
    placement: str = "hime"          # hime | static
    use_alpha: bool = True
    use_chi: bool = True
    seed: int = 0
    eval_every: int = 10


def _draw(cfg, a, s, rng):
    """Returns (local indices, HT weight per draw)."""
    if s == 0:
        return np.array([], int), np.array([])
    if cfg.sampler == "uniform":
        pi = np.full(len(a), s / len(a))
    else:
        pi = inclusion_probs(a, s, cfg.eps)
    if cfg.sampler == "v1":                                   # design v1: multinomial with replacement
        p = pi / s
        d = rng.choice(len(a), size=s, p=p)
        return d, 1.0 / (s * p[d])
    S = sampford(pi, rng)
    return S, 1.0 / pi[S]


def run(pb, cfg: Config, uav_xy0, theta_star=None):
    E = pb["experts"]; names = list(E); N = len(pb["client_rows"]); K = len(uav_xy0)
    kap = dict(zip(names, cfg.kappa))
    Tp = cfg.Tp or placement_period(cfg.kappa)
    assert all(Tp % k == 0 for k in cfg.kappa), "T_p must be a multiple of every kappa"
    r_av, r_sm, r_pl = (stream(cfg.seed, k) for k in ("availability", "sampling", "placement"))
    avail = r_av.random((cfg.T, N)) < pb["alpha"][None]      # common random numbers across arms
    Xc = {m: [pb["X"][m][r] for r in pb["client_rows"]] for m in names}
    yc = [pb["y"][r] for r in pb["client_rows"]]
    n_i = np.array([len(r) for r in pb["client_rows"]], float)
    omega = n_i / n_i.sum()
    alpha_hat, chi = np.full(N, 0.5), np.ones(N)
    a_pred = np.sqrt(len(names)) * omega
    h, rad = best_altitude(cfg.hmin, cfg.hmax, L=cfg.L, env=cfg.env)
    C = cfg.C or math.ceil(1.5 * cfg.M / K)
    Pm = {m: E[m].nbytes for m in names}
    theta = {m: np.zeros(E[m].dim) for m in names}
    last = {m: theta[m].copy() for m in names}; H = {m: 0.0 for m in names}
    uav = np.array(uav_xy0, float); edge = np.full(N, -1); moved = np.zeros(K, bool)
    B = dict(access_down=0, access_up=0, backhaul=0)
    hist, placements = [], []
    for t in range(1, cfg.T + 1):
        if (t - 1) % Tp == 0:                                 # placement epoch
            nd = np.flatnonzero(~pb["direct"])
            if cfg.placement == "hime" and len(nd):
                uav, sel, info = place(pb["client_xy"][nd], a_pred[nd], uav, rad, cfg.dmax, Tp, r_pl)
                moved = sel != 0
                placements.append(dict(t=t, best=info["best"], start_f=info["start_f"], final_f=info["final_f"]))
            else:
                moved = np.zeros(K, bool)
            d = np.hypot(pb["client_xy"][:, None, 0] - uav[None, :, 0], pb["client_xy"][:, None, 1] - uav[None, :, 1])
            cov = d <= rad * (1 + 1e-9)
            edge = np.where(cov.any(1), 1 + np.argmin(np.where(cov, d, np.inf), 1), -1)
            edge[pb["direct"]] = 0
            if cfg.use_chi:
                chi = ema_floor(chi, (edge >= 0).astype(float), 0.2, 0.2)
            wt = np.tile(omega / chi, (len(names), 1))
            on = edge >= 0
            W = {m: np.bincount(edge[on], weights=wt[k][on], minlength=K + 1) for k, m in enumerate(names)}
            th_e = {m: np.tile(theta[m], (K + 1, 1)) for m in names}
            yv = {m: np.zeros((K + 1, E[m].dim)) for m in names}
            t0 = t
        reach = edge >= 0
        if t == t0:                                           # a UAV that moved serves nobody this round
            reach &= ~((edge >= 1) & moved[np.maximum(edge - 1, 0)])
        resp = reach & avail[t - 1]
        if cfg.use_alpha:
            alpha_hat[reach] = ema_floor(alpha_hat[reach], resp[reach].astype(float), 0.1, 0.2)
        ah = alpha_hat if cfg.use_alpha else np.ones(N)
        U = np.flatnonzero(resp)
        ghat = np.array([[np.linalg.norm(E[m].grad(th_e[m][edge[i]], Xc[m][i], yc[i])) for i in U] for m in names]).reshape(len(names), len(U))
        a = importance(ghat, wt[:, U], ah[U]) if len(U) else np.zeros(0)
        a_pred[U] = a
        eta_l = cfg.eta_l0 * (1 + t / 10) ** -0.5
        eU = edge[U]
        A = np.bincount(eU, weights=a, minlength=K + 1)
        size = np.bincount(eU, minlength=K + 1)
        s = allocate(A, np.minimum(C, size), cfg.M)
        for j in range(K + 1):
            mem = U[eU == j]
            S, hw = _draw(cfg, a[eU == j], int(s[j]), r_sm) if s[j] else (np.array([], int), np.array([]))
            for k, m in enumerate(names):
                if W[m][j] <= 0:
                    continue
                base = th_e[m][j]
                acc = np.zeros_like(base)
                cache = {}
                for li, c in zip(S, hw):
                    i = mem[li]
                    if i not in cache:
                        cache[i] = E[m].local_steps(base, Xc[m][i], yc[i], cfg.tau, eta_l) - base
                    acc += wt[k, i] * c / ah[i] * cache[i]
                th_e[m][j] = E[m].project(base + cfg.eta_e * (acc / W[m][j] + yv[m][j]))
            B["access_up"] += len(set(mem[S].tolist())) * sum(Pm.values())
        B["access_up"] += 4 * len(names) * len(U)
        B["access_down"] += sum(Pm.values()) * len(np.unique(edge[reach]))
        rec = dict(t=t, available=len(U), sampled=int(s.sum()), reachable=int(reach.sum()))
        for m in names:
            H[m] += cfg.eta_e
            if t % kap[m] == 0:
                act = W[m] > 0
                Wa = W[m][act]
                new = (Wa[:, None] * th_e[m][act]).sum(0) / Wa.sum()
                if cfg.correction:
                    D = th_e[m] - last[m]
                    Db = (Wa[:, None] * D[act]).sum(0) / Wa.sum()
                    yv[m][act] += (Db - D[act]) / H[m]
                    rec[f"wy_{m}"] = float(np.abs((W[m][:, None] * yv[m]).sum(0)).max())
                theta[m] = new; th_e[m][:] = new; last[m] = new.copy(); H[m] = 0.0
                rec[f"kact_{m}"] = int(act[1:].sum())
                B["backhaul"] += Pm[m] * rec[f"kact_{m}"] * (3 if cfg.correction else 2)
        if theta_star is not None and t % cfg.eval_every == 0:
            for m in names:
                rec[f"dist_{m}"] = float(np.linalg.norm(theta[m] - theta_star[m]))
        hist.append(rec)
    # virtual cloud average of the edge models
    final = {}
    for m in names:
        act = W[m] > 0
        final[m] = (W[m][act, None] * th_e[m][act]).sum(0) / W[m][act].sum()
    return dict(theta=final, history=hist, placements=placements, bytes=B, Tp=Tp, altitude=h, radius=rad)


def evaluate(pb, theta, rows_cal=None, rows_test=None):
    E, X, y, lp = pb["experts"], pb["X"], pb["y"], pb["logit_pi"]
    rc = pb["calib_rows"] if rows_cal is None else rows_cal
    rt = pb["test_rows"] if rows_test is None else rows_test
    g = {m: E[m].logit(theta[m], X[m]) for m in E}
    p = fit_stacker(g["S"][rc], g["I"][rc], y[rc], lp)
    z = stacker_logit(p, g["S"][rt], g["I"][rt], lp)
    return dict(stacker=p.tolist(), auprc=average_precision(y[rt], z),
                auprc_S=average_precision(y[rt], g["S"][rt]), auprc_I=average_precision(y[rt], g["I"][rt]),
                prevalence=float(y[rt].mean()))


def central(pb):
    th = {}
    for m, e in pb["experts"].items():
        rows = np.concatenate(pb["client_rows"])
        th[m] = e.fit_central(pb["X"][m][rows], pb["y"][rows])
    return th
