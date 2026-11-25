"""GPU fits for unconstrained convex linear experts. The CPU scipy path (LinearExpert.fit_central) stays the
reference; tests/test_gpu.py checks the two agree. Torch is imported lazily (it ships in the `embed` extra)."""
import numpy as np


def available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def fit_newton(X, y, logit_pi, lams, device="cuda", gtol=1e-10, maxit=100):
    """Minimise mean softplus(z) - y z, z = X th + logit_pi, plus lam/2 ||th[1:]||^2, for each lam.
    Damped Newton: gradient, objective and iterate in fp64; Hessian built and solved in fp32 (an inexact
    Newton step still converges to the fp64 optimum because the gradient is exact). Warm-started across
    lams in ascending order. Returns an array (len(lams), d) in the order of `lams`."""
    import torch
    Xd = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float64, device=device)
    X32 = Xd.float()
    yd = torch.as_tensor(y, dtype=torch.float64, device=device)
    n, d = Xd.shape
    pen = torch.ones(d, dtype=torch.float64, device=device)
    pen[0] = 0.0
    zero = torch.zeros((), dtype=torch.float64, device=device)

    def obj(th, lam):
        z = Xd @ th + logit_pi
        return (torch.logaddexp(zero, z) - yd * z).mean() + 0.5 * lam * (pen * th * th).sum()

    out, th = {}, torch.zeros(d, dtype=torch.float64, device=device)
    for lam in sorted(lams):
        for _ in range(maxit):
            p = torch.sigmoid(Xd @ th + logit_pi)
            g = Xd.T @ (p - yd) / n + lam * pen * th
            if g.abs().max().item() < gtol:
                break
            w = (p * (1 - p)).float()
            H = (X32.T * w) @ X32 / n
            H.diagonal().add_((lam * pen).float() + 1e-10)
            step = torch.linalg.solve(H, g.float()).double()
            f0, slope, t = obj(th, lam), (g @ step).item(), 1.0
            while t > 1e-10 and obj(th - t * step, lam) > f0 - 1e-4 * t * slope:
                t *= 0.5
            if t <= 1e-10:                       # no decrease possible: at fp64 resolution of the optimum
                break
            th = th - t * step
        out[lam] = th.cpu().numpy()
    return np.stack([out[lam] for lam in lams])
