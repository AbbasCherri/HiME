import itertools

import numpy as np
from hypothesis import given, settings, strategies as st

from hime.participation import (allocate, ht_variance, importance, inclusion_probs, joint_inclusion,
                                multinomial_variance, sampford, sampford_design)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.floats(0, 100), min_size=1, max_size=30), st.integers(0, 30))
def test_P6_inclusion_probs_valid(a, s):
    s = min(s, len(a)); pi = inclusion_probs(np.array(a), s)
    assert abs(pi.sum() - s) < 1e-9 and (pi <= 1 + 1e-12).all() and (pi >= 0).all()


def _uncapped(rng, N, s):
    while True:
        pi = inclusion_probs(rng.gamma(1.0, size=N), s)
        if pi.max() < 0.97:
            return pi


def test_P6_sampford_design_exact_and_HT_unbiased():
    rng = np.random.default_rng(0)
    for _ in range(30):
        N = int(rng.integers(3, 9)); s = int(rng.integers(1, N)); pi = _uncapped(rng, N, s)
        sets, p = sampford_design(pi)
        P = joint_inclusion(sets, p, N)
        assert np.allclose(np.diag(P), pi, atol=1e-12)
        V = rng.normal(size=(N, 3))
        est = sum(pk * (V[list(S)] / pi[list(S), None]).sum(0) for S, pk in zip(sets, p))
        assert np.allclose(est, V.sum(0), atol=1e-10)


def test_P6_sampler_matches_design_monte_carlo():
    rng = np.random.default_rng(1); pi = _uncapped(rng, 7, 3)
    sets, p = sampford_design(pi); P = joint_inclusion(sets, p, 7)
    R = 40_000; cnt = np.zeros((7, 7)); V = rng.normal(size=(7, 2)); sq = 0.0
    for _ in range(R):
        S = sampford(pi, rng); assert len(S) == 3 and len(set(S)) == 3
        cnt[np.ix_(S, S)] += 1
        e = (V[S] / pi[S, None]).sum(0) - V.sum(0); sq += e @ e
    z = np.abs(cnt / R - P) / np.sqrt(P * (1 - P) / R + 1e-300)
    assert z[P > 0].max() <= 4.5, z.max()
    var = ht_variance(pi, P, V)
    assert abs(sq / R - var) <= 4 * var * np.sqrt(2 / R) * 3


def test_T2_sampford_dominates_multinomial_uncapped_and_O5_capped():
    rng = np.random.default_rng(2); capped_viol = 0; capped_n = 0
    for k in range(300):
        N = int(rng.integers(3, 9)); s = int(rng.integers(1, N))
        a = rng.gamma(0.5 if k % 2 else 2.0, size=N)
        pi = inclusion_probs(a, s)
        V = rng.normal(size=(N, 2)) * a[:, None]
        sets, p = sampford_design(np.minimum(pi, 1 - 1e-12))
        P = joint_inclusion(sets, p, N)
        vht = ht_variance(np.diag(P), P, V)
        star = inclusion_probs(a, s, eps=0.0)
        p_v1 = 0.9 * a / a.sum() + 0.1 / N                   # design v1 draw probabilities
        vmn = multinomial_variance(p_v1, s, V)
        if (star < 1 - 1e-12).all():
            assert vht <= multinomial_variance(pi / s, s, V) * (1 + 1e-9)
            assert vht <= vmn * (1 + 1e-9)
        else:
            capped_n += 1; capped_viol += vht > vmn * (1 + 1e-9)
    print(f"O5 evidence: capped instances {capped_n}, dominance violated {capped_viol}")


def test_L5_allocation_equals_bruteforce():
    rng = np.random.default_rng(3)
    for _ in range(100):
        J = int(rng.integers(1, 5)); A = rng.gamma(1.0, size=J); cap = rng.integers(0, 4, size=J); M = int(rng.integers(1, 9))
        s = allocate(A, cap, M)
        tgt = min(M, cap.sum()); assert s.sum() == tgt and (s <= cap).all()
        if tgt >= (cap > 0).sum():
            best = min((sum(A[j] ** 2 / c[j] for j in range(J) if cap[j] > 0), c)
                       for c in itertools.product(*[range(1 if cap[j] else 0, cap[j] + 1) for j in range(J)]) if sum(c) == tgt)
            assert np.isclose(sum(A[j] ** 2 / s[j] for j in range(J) if cap[j] > 0), best[0])


def test_P11_norm_proportional_is_optimal_with_replacement():
    rng = np.random.default_rng(4)
    for _ in range(100):
        V = rng.normal(size=(10, 3)) * rng.gamma(1.0, size=(10, 1)); b = np.linalg.norm(V, axis=1)
        q = rng.dirichlet(np.ones(10))
        assert multinomial_variance(b / b.sum(), 4, V) <= multinomial_variance(q, 4, V) + 1e-12


def test_P6_importance_edge_cases():
    a = importance(np.zeros((2, 3)), np.ones((2, 3)), np.ones(3))
    assert (a == 0).all()
    pi = inclusion_probs(a, 2); assert np.allclose(pi, 2 / 3)
    assert inclusion_probs(np.array([5.0]), 1)[0] == 1.0
