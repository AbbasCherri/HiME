"""Logit-adjusted loss and convex linear experts with projected full-batch GD (plan 2.3, 2.5)."""
import numpy as np
from scipy.optimize import minimize


def softplus(z):
    return np.logaddexp(0.0, z)


def sigmoid(z):
    return np.exp(-softplus(-z))


def logit(p):
    return np.log(p) - np.log1p(-p)


def la_loss(g, y, logit_pi):
    """Log-loss on z = g + logit(pi); its population minimiser g is the log-likelihood ratio (Prop 1)."""
    z = g + logit_pi
    return softplus(z) - y * z


class LinearExpert:
    """g(x) = theta . x with x[:,0] = 1.  F = mean la_loss + lam/2 ||theta[1:]||^2."""

    def __init__(self, dim, logit_pi, lam=1e-4, nonneg=None):
        self.dim, self.lp, self.lam = dim, float(logit_pi), lam
        self.nonneg = np.zeros(dim, bool) if nonneg is None else np.asarray(nonneg, bool)

    @property
    def nbytes(self):                      # fp32 payload
        return 4 * self.dim

    def project(self, th):
        th = np.array(th, float)
        th[self.nonneg] = np.maximum(th[self.nonneg], 0.0)
        return th

    def logit(self, th, X):
        return X @ th

    def loss(self, th, X, y):
        return la_loss(X @ th, y, self.lp).mean() + 0.5 * self.lam * th[1:] @ th[1:]

    def grad(self, th, X, y):
        g = X.T @ (sigmoid(X @ th + self.lp) - y) / len(y)
        g[1:] += self.lam * th[1:]
        return g

    def local_steps(self, th, X, y, tau, lr):
        for _ in range(tau):
            th = self.project(th - lr * self.grad(th, X, y))
        return th

    def fit_central(self, X, y, th0=None):
        th0 = np.zeros(self.dim) if th0 is None else th0
        b = [(0, None) if c else (None, None) for c in self.nonneg]
        r = minimize(lambda t: (self.loss(t, X, y), self.grad(t, X, y)), th0, jac=True,
                     method="L-BFGS-B", bounds=b, options=dict(maxiter=5000, gtol=1e-10, ftol=1e-15))
        return r.x
