import copy

import numpy as np

from hime.engine import Config, central, evaluate, run
from hime.experts import LinearExpert
from hime.schedule import federate, kappas, placement_period, u_fed


def test_P7_schedule_rules():
    assert placement_period([1, 1]) == 5 and placement_period([2, 3]) == 6 and placement_period([4, 6]) == 12
    assert placement_period([5]) == 10
    P, K = np.array([144.0, 516.0]), 4
    k = kappas(P, np.array([1.0, 1.0]), K, budget=0.25 * 2 * K * P.sum())
    assert (2 * K * P / k).sum() <= 0.25 * 2 * K * P.sum() + 1e-9 and (k >= 1).all()
    assert (kappas(P, np.ones(2), K, budget=1e12) == 1).all()


def test_P4p_routing_monotone():
    part = np.full(100, 23)
    us = [u_fed(516, part, 4, 100, k) for k in range(1, 20)]
    assert all(a >= b for a, b in zip(us, us[1:]))
    assert federate(516, part, 4, 100, 100 * 600 * 513) and federate(144, part, 4, 100, 100 * 600 * 29)


def _free(pb):
    pb = dict(pb); pb["experts"] = {m: LinearExpert(e.dim, e.lp, e.lam) for m, e in pb["experts"].items()}
    return pb


def test_P10_kappa1_correction_is_inert_and_zero_sum(small):
    w, pb, uav = small
    pb = _free(pb)                                             # projection inactive (Prop 10 scope)
    on = run(pb, Config(T=12, M=8, kappa=(1, 1), correction=True, seed=1000), uav)
    off = run(pb, Config(T=12, M=8, kappa=(1, 1), correction=False, seed=1000), uav)
    for m in on["theta"]:
        assert np.allclose(on["theta"][m], off["theta"][m], atol=1e-12, rtol=0)
    assert max(r.get(f"wy_{m}", 0) for r in on["history"] for m in ("S", "I")) < 1e-10


def test_P10_zero_sum_kappa_gt1(small):
    w, pb, uav = small
    out = run(pb, Config(T=16, M=8, kappa=(2, 4), seed=1001), uav)
    assert max(r.get(f"wy_{m}", 0) for r in out["history"] for m in ("S", "I")) < 1e-10


def test_P6_backhaul_bytes_exact(small):
    w, pb, uav = small
    out = run(pb, Config(T=16, M=8, kappa=(2, 4), seed=1002), uav)
    P = {m: e.nbytes for m, e in pb["experts"].items()}
    exp = sum(P[m] * r[f"kact_{m}"] * 3 for r in out["history"] for m in P if f"kact_{m}" in r)
    assert out["bytes"]["backhaul"] == exp


def test_P7_deterministic_and_common_random_numbers(small):
    w, pb, uav = small
    a = run(pb, Config(T=10, M=8, seed=1003), uav); b = run(pb, Config(T=10, M=8, seed=1003), uav)
    assert all(np.array_equal(a["theta"][m], b["theta"][m]) for m in a["theta"]) and a["bytes"] == b["bytes"]
    c = run(pb, Config(T=10, M=8, seed=1003, sampler="uniform"), uav)
    assert [r["available"] for r in a["history"]] == [r["available"] for r in c["history"]]


def test_P7_tiny_run_beats_prevalence(small):
    w, pb, uav = small
    ts = central(pb)
    out = run(pb, Config(T=60, M=10, seed=1004, eval_every=10), uav, theta_star=ts)
    ev = evaluate(pb, out["theta"])
    print("realistic run:", ev, "gap to central (E_I):", [r["dist_I"] for r in out["history"] if "dist_I" in r])
    assert ev["auprc"] > 2 * ev["prevalence"]


def test_P7_converges_to_central_in_controlled_setting(small):
    """All clients direct, always available, full participation: the only gap left is local-step drift,
    which shrinks with the decaying local step size. Realistic gaps are measured, not asserted (O1)."""
    w, pb, uav = small
    N = len(pb["client_rows"])
    pc = dict(pb); pc["direct"] = np.ones(N, bool); pc["alpha"] = np.ones(N)
    ts = central(pc)
    out = run(pc, Config(T=80, M=N, C=N, seed=1005, eval_every=10, use_chi=False), uav, theta_star=ts)
    rows = np.concatenate(pc["client_rows"])
    for m, e in pc["experts"].items():
        d = [r[f"dist_{m}"] for r in out["history"] if f"dist_{m}" in r]
        X, y = pc["X"][m][rows], pc["y"][rows]
        f0, fT, fs = (e.loss(t, X, y) for t in (np.zeros(e.dim), out["theta"][m], ts[m]))
        print(m, "distance to central optimum:", d, "objective gap ratio:", (fT - fs) / (f0 - fs))
        assert all(a > b for a, b in zip(d, d[1:]))          # distance falls at every evaluation
        assert (fT - fs) / (f0 - fs) < 0.2                    # most of the objective gap closed


def test_H10_correction_reduces_drift_at_kappa4(small):
    w, pb, uav = small
    ts = central(pb)
    res = {c: run(pb, Config(T=40, M=10, kappa=(4, 4), correction=c, seed=1000, placement="static"), uav, theta_star=ts)
           for c in (True, False)}
    d = {c: sum(np.linalg.norm(res[c]["theta"][m] - ts[m]) for m in ts) for c in res}
    print("distance to central optimum, with/without correction:", d)
    assert d[True] <= d[False] * 1.05                          # reported; strict claim decided in Phase 11


def test_NC4_NC5_biased_selector_and_no_HT_weights_fail_unbiasedness():
    from hime.participation import inclusion_probs, sampford
    rng = np.random.default_rng(0)
    v = rng.gamma(1.0, size=12); a = v.copy()
    pi = inclusion_probs(a, 4)
    R = 20_000
    ht = np.mean([(v[S] / pi[S]).sum() for S in (sampford(pi, rng) for _ in range(R))])
    noht = np.mean([v[S].sum() * 12 / 4 for S in (sampford(pi, rng) for _ in range(R))])
    top = v[np.argsort(-a)[:4]].sum() * 12 / 4                # deterministic loss-ranked selector
    se = np.std([(v[S] / pi[S]).sum() for S in (sampford(pi, rng) for _ in range(2000))]) / np.sqrt(R)
    assert abs(ht - v.sum()) <= 4 * se
    assert abs(noht - v.sum()) > 4 * se and abs(top - v.sum()) > 4 * se
