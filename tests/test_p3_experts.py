import numpy as np
import pytest

from hime.experts import LinearExpert, la_loss, logit
from hime.features import (fed_knots, fed_standardise_pca, hist_quantiles, client_hist, ispline_basis,
                           ispline_knots, suff_stats, transform)


def test_P1_loss_stable_and_gradient():
    g = np.array([-700.0, 0.0, 700.0])
    assert np.isfinite(la_loss(g, np.array([1, 0, 1.0]), logit(0.03))).all()
    rng = np.random.default_rng(0)
    X = np.c_[np.ones(50), rng.normal(size=(50, 3))]; y = (rng.random(50) < .3).astype(float)
    e = LinearExpert(4, logit(.3), lam=1e-2); th = rng.normal(size=4)
    fd = np.array([(e.loss(th + h, X, y) - e.loss(th - h, X, y)) / 2e-6 for h in np.eye(4) * 1e-6])
    assert np.allclose(fd, e.grad(th, X, y), atol=1e-7)


@pytest.mark.parametrize("pi", [0.03, 0.3])
def test_P1_recovers_log_likelihood_ratio(pi):
    rng = np.random.default_rng(1); n = 300_000; m = np.array([0.8, -0.4])
    y = (rng.random(n) < pi).astype(float)
    x = rng.normal(size=(n, 2)) + y[:, None] * m
    X = np.c_[np.ones(n), x]
    th = LinearExpert(3, logit(y.mean()), lam=0.0).fit_central(X, y)
    assert np.allclose(th, np.r_[-0.5 * m @ m, m], atol=0.06), th


def test_P3_ispline_range_monotone_dup_knots_clip():
    t = ispline_knots(0.0, 1.0, [0.25, 0.25, 0.5])           # duplicate merged
    assert len(t) == 3 + 2 + 3
    x = np.linspace(-1, 2, 401)
    I = ispline_basis(x, t)
    assert I.min() >= -1e-12 and I.max() <= 1 + 1e-12
    assert (np.diff(I, axis=0) >= -1e-12).all()
    assert np.allclose(I[x <= 0], I[x <= 0][0]) and np.allclose(I[x >= 1], I[x >= 1][0])


def test_P3_hist_knots_within_one_bin():
    rng = np.random.default_rng(2)
    cols = [rng.lognormal(size=rng.integers(20, 200)) for _ in range(30)]
    allx = np.concatenate(cols); lo, hi = allx.min(), allx.max()
    q = hist_quantiles(sum(client_hist(c, lo, hi) for c in cols), lo, hi)
    assert np.all(np.abs(q - np.quantile(allx, [.25, .5, .75])) <= (hi - lo) / 1000 + 1e-12)
    assert len(fed_knots(cols)) <= 9


def test_P3_fed_standardise_pca_equals_central():
    rng = np.random.default_rng(3)
    Zs = [rng.normal(size=(rng.integers(5, 50), 6)) @ rng.normal(size=(6, 6)) for _ in range(20)]
    Z = np.vstack(Zs)
    mu, sd, P = fed_standardise_pca([suff_stats(z) for z in Zs], 3)
    assert np.allclose(mu, Z.mean(0), atol=1e-12) and np.allclose(sd, Z.std(0), rtol=1e-10)
    w, V = np.linalg.eigh(np.corrcoef(Z.T)); V = V[:, ::-1][:, :3]
    V = V * np.sign(V[np.abs(V).argmax(0), np.arange(3)])
    assert np.allclose(P, V, atol=1e-9)
    assert np.allclose(transform(Z, mu, sd, P), ((Z - Z.mean(0)) / Z.std(0)) @ V, atol=1e-8)


def test_P3_projected_gd_nonneg_and_central_kkt():
    rng = np.random.default_rng(4)
    X = np.c_[np.ones(200), rng.random((200, 5))]; y = (rng.random(200) < .2).astype(float)
    e = LinearExpert(6, logit(.2), lam=1e-3, nonneg=[False] + [True] * 5)
    th = e.local_steps(np.zeros(6) - 1, X, y, 20, 0.5)
    assert (th[1:] >= 0).all()
    ts = e.fit_central(X, y); g = e.grad(ts, X, y)
    free = ~e.nonneg | (ts > 1e-8)
    assert np.abs(g[free]).max() < 1e-6 and (g[~free] >= -1e-6).all()
