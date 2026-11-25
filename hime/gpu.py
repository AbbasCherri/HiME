"""GPU fits for unconstrained convex linear experts. The CPU scipy path (LinearExpert.fit_central) stays the
reference; tests/test_gpu.py checks the two agree. Torch is imported lazily (it ships in the `embed` extra)."""
import numpy as np


def available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _t(a, device):
    import torch
    if isinstance(a, torch.Tensor):
        return a.to(device=device, dtype=torch.float64)
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)


def pca_basis(Z, device="cuda"):
    """GPU twin of features.fed_standardise_pca on pooled rows Z (the federated sums equal the pooled ones):
    mean, sd and the full correlation eigenbasis with the same ordering and sign rule. Torch fp64 tensors."""
    import torch
    Z = _t(Z, device)
    n = Z.shape[0]
    mu = Z.sum(0) / n
    C = Z.T @ Z / n - torch.outer(mu, mu)
    sd = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-12))
    _, V = torch.linalg.eigh(C / torch.outer(sd, sd))
    V = V.flip(1)
    V = V * torch.sign(V[V.abs().argmax(0), torch.arange(V.shape[1], device=V.device)])
    return mu, sd, V


def design(Z, mu, sd, V, r=None):
    """[1, ((Z - mu) / sd) @ V[:, :r]] on the device of the basis (features.transform plus intercept)."""
    import torch
    Zs = (_t(Z, mu.device) - mu) / sd
    T = Zs @ (V if r is None else V[:, :r])
    return torch.cat([torch.ones(len(T), 1, dtype=T.dtype, device=T.device), T], 1)


def fit_newton(X, y, logit_pi, lams, device="cuda", gtol=1e-10, maxit=100):
    """Minimise mean softplus(z) - y z, z = X th + logit_pi, plus lam/2 ||th[1:]||^2, for each lam.
    Damped Newton: gradient, objective and iterate in fp64; Hessian built and solved in fp32 (an inexact
    Newton step still converges to the fp64 optimum because the gradient is exact). Warm-started across
    lams in ascending order. Returns an array (len(lams), d) in the order of `lams`."""
    import torch
    Xd = _t(X, device)
    X32 = Xd.float()
    yd = _t(y, device)
    n, d = Xd.shape
    pen = torch.ones(d, dtype=torch.float64, device=device)
    pen[0] = 0.0
    zero = torch.zeros((), dtype=torch.float64, device=device)

    def obj(th, lam):
        z = Xd @ th + logit_pi
        return (torch.logaddexp(zero, z) - yd * z).mean() + 0.5 * lam * (pen * th * th).sum()

    out, th = {}, torch.zeros(d, dtype=torch.float64, device=device)
    for lam in sorted(lams):
        exact = False                            # switch to an fp64 Hessian if an fp32 step fails
        for _ in range(maxit):
            p = torch.sigmoid(Xd @ th + logit_pi)
            g = Xd.T @ (p - yd) / n + lam * pen * th
            if g.abs().max().item() < gtol:
                break
            w = p * (1 - p)
            if exact:
                H = (Xd.T * w) @ Xd / n
                H.diagonal().add_(lam * pen + 1e-12)
                step = torch.linalg.solve(H, g)
            else:
                H = (X32.T * w.float()) @ X32 / n
                H.diagonal().add_((lam * pen).float() + 1e-10)
                step = torch.linalg.solve(H, g.float()).double()
            f0, slope, t = obj(th, lam), (g @ step).item(), 1.0
            while t > 1e-10 and obj(th - t * step, lam) > f0 - 1e-4 * t * slope:
                t *= 0.5
            if t <= 1e-10:
                if exact:
                    break                        # no decrease possible even with exact steps
                exact = True
                continue
            th = th - t * step
        p = torch.sigmoid(Xd @ th + logit_pi)
        gmax = (Xd.T @ (p - yd) / n + lam * pen * th).abs().max().item()
        if gmax > max(gtol, 1e-7):
            raise RuntimeError(f"fit_newton did not converge (lam={lam:g}, max |grad| {gmax:.2e}, d={d}, n={n})")
        out[lam] = th.cpu().numpy()
    return np.stack([out[lam] for lam in lams])
