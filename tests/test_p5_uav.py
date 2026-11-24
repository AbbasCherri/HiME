import numpy as np
import pytest

from hime.uav import (best_altitude, candidates, coverage, exact_opt, f_pairs, f_sel, greedy, local_search,
                      pathloss, place, radius)


def test_L7_radius_matches_scan_and_cap():
    for h in (50.0, 120.0):
        r = radius(h)
        d = np.linspace(0, 3 * r, 300_001)
        assert abs(d[pathloss(h, d) <= 117.0].max() - r) < 3 * r / 300_000 + 1e-3
    hs = np.linspace(10, 3000, 600); rs = np.array([radius(h) for h in hs])
    k = rs.argmax(); assert (np.diff(rs[:k + 1]) >= -1e-6).all() and (np.diff(rs[k:]) <= 1e-6).all()   # unimodal
    h, r = best_altitude(10, 150)
    scan = max(radius(x) for x in np.linspace(10, 150, 1401))
    assert r >= scan - 1e-3
    if hs[k] > 150: assert abs(h - 150) < 1e-2                # capped argmax
    assert radius(h, L=114.0) < radius(h, L=117.0)            # margin shrinks r


def _inst(rng, n, degenerate=False):
    X = rng.uniform(0, 4000, (n, 2))
    if degenerate:
        X[1] = X[0]; X[2] = X[0] + [2000.0, 0]; X[3] = (X[0] + X[2]) / 2
    return X


@pytest.mark.parametrize("seed", range(15))
def test_L8_candidate_sufficiency(seed):
    rng = np.random.default_rng(seed); X = _inst(rng, 12, seed % 3 == 0)
    r, o, dmax = 1000.0, np.array([2000.0, 2000.0]), 1800.0
    P = candidates(X, r, o, dmax); cc = coverage(P, X, r)
    g = np.stack(np.meshgrid(np.linspace(200, 3800, 90), np.linspace(200, 3800, 90)), -1).reshape(-1, 2)
    g = g[np.hypot(*(g - o).T) <= dmax]
    for row in coverage(g, X, r * (1 - 1e-6)):
        assert (cc[:, row].all(1)).any()
    assert (np.hypot(*(P - o).T) <= dmax + 1e-6).all() and np.allclose(P[0], o)


def _covs(rng, n=20, K=3, Tp=5):
    X = rng.uniform(0, 5000, (n, 2)); w = rng.random(n)
    uav = rng.uniform(0, 5000, (K, 2))
    cands = [candidates(X, 900.0, u, 2500.0) for u in uav]
    return [coverage(P, X, 900.0) for P in cands], w, Tp, X, uav


def test_L9p_monotone_submodular():
    rng = np.random.default_rng(0)
    for _ in range(100):
        covs, w, Tp, *_ = _covs(rng)
        ground = [(j, c) for j in range(len(covs)) for c in range(len(covs[j]))]
        idx = rng.permutation(len(ground))
        B = [ground[i] for i in idx[:6]]; A = B[:3]; e = ground[idx[6]]
        fa, fb = f_pairs(A, covs, w, Tp), f_pairs(B, covs, w, Tp)
        assert fb >= fa - 1e-12
        assert f_pairs(A + [e], covs, w, Tp) - fa >= f_pairs(B + [e], covs, w, Tp) - fb - 1e-9


def test_T1_greedy_half_opt_and_P9_multistart():
    rng = np.random.default_rng(1); ratios = []
    for k in range(200):
        covs, w, Tp, X, uav = _covs(rng, n=int(rng.integers(8, 40)), K=int(rng.integers(1, 5)))
        opt, _ = exact_opt(covs, w, Tp)
        g = f_sel(greedy(covs, w, Tp), covs, w, Tp)
        ratios.append(g / opt if opt > 0 else 1.0)
        assert g >= 0.5 * opt - 1e-9 and g <= opt + 1e-6
        ls = local_search(greedy(covs, w, Tp), covs, w, Tp)
        assert f_sel(ls, covs, w, Tp) >= g - 1e-12
    import os; os.makedirs("runs", exist_ok=True); np.save("runs/h6_greedy_ratios.npy", np.array(ratios))
    print("greedy/opt min", min(ratios), "mean", np.mean(ratios))


def test_P9_place_dominates_starts_and_respects_dmax():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 20000, (60, 2)); w = rng.random(60); uav = rng.uniform(0, 20000, (4, 2))
    pos, sel, info = place(X, w, uav, 1500.0, 6000.0, 6, rng)
    best = max(info["final_f"].values())
    assert all(best >= v - 1e-12 for v in info["start_f"].values())
    assert (np.hypot(*(pos - uav).T) <= 6000 + 1e-6).all()
