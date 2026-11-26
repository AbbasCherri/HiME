"""HiME-2 and central references on the real release, development/tuning seeds, validation fold only.
Usage: uv run python scripts/run_real.py --N 100 --seeds 1000 1001 [--T 100] [--encoder dinov2_vits14]"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from hime import data as D
from hime.engine import Config, central, evaluate, run
from hime.pipeline import prepare
from hime.rng import stream
from hime.schedule import kappas
from hime.uav import best_altitude, calibrate_K, kmeans

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, nargs="+", default=[100])
ap.add_argument("--seeds", type=int, nargs="+", default=[1000])
ap.add_argument("--T", type=int, default=100)
ap.add_argument("--r", type=int, default=128, help="PCA dimension for E_I (0 = full)")
ap.add_argument("--encoder", default="dinov2_vits14")
ap.add_argument("--emb-root", default=None)
ap.add_argument("--out", default="runs/real")
ap.add_argument("--L", type=float, default=123.0, help="max path loss (dB); 123 chosen by the coverage sweep")
ap.add_argument("--kmax", type=int, default=30, help="largest UAV count tried")
a = ap.parse_args()

rows = []
for N in a.N:
    for seed in a.seeds:
        assert D.seed_role(seed) in ("development", "tuning"), "test seeds only after freeze"
        t0 = time.time()
        w = D.world(N, seed, eval_role="val", encoder=a.encoder, emb_root=a.emb_root)
        pb = prepare(w, r=a.r or None)
        ts = central(pb)
        ref = evaluate(pb, ts)
        h, rad = best_altitude(L=a.L)
        nd = ~pb["direct"]
        K = calibrate_K(pb["client_xy"][nd], rad, Ks=range(4, a.kmax + 1, 2))
        uav0 = kmeans(pb["client_xy"][nd], K, np.ones(nd.sum()), stream(seed, "kmeans"))
        cov = (np.hypot(pb["client_xy"][:, None, 0] - uav0[None, :, 0], pb["client_xy"][:, None, 1] - uav0[None, :, 1]) <= rad).any(1) | pb["direct"]
        M = max(1, round(0.5 * float(pb["alpha"][cov].sum())))
        P = np.array([e.nbytes for e in pb["experts"].values()], float)
        kap = kappas(P, np.ones(len(P)), K, budget=0.25 * 2 * M * P.sum())   # B_bh = 0.25 x relay-everything
        cfg = Config(T=a.T, M=M, C=math.ceil(1.5 * M / K), kappa=tuple(int(k) for k in kap), L=a.L, seed=seed)
        out = run(pb, cfg, uav0, theta_star=ts)
        ev = evaluate(pb, out["theta"])
        prev = ev["prevalence"]
        rec = dict(N=N, seed=seed, L=a.L, kmax=a.kmax, K=K, M=M, kappa=list(cfg.kappa), Tp=out["Tp"], radius_m=rad, altitude_m=h, coverage0=float(cov.mean()),
                   direct=int(pb["direct"].sum()), nonpositive_im=w["nonpositive_im"], central=ref, hime2=ev,
                   bytes=out["bytes"], degenerate={k: v < 3 * prev for k, v in
                                                   (("hime2", ev["auprc"]), ("central", ref["auprc"]),
                                                    ("central_S", ref["auprc_S"]), ("central_I", ref["auprc_I"]))},
                   seconds=time.time() - t0)
        d = Path(a.out) / str(N) / str(seed); d.mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(d / "final.json", "w"), indent=1, default=float)
        json.dump(out["history"], open(d / "rounds.json", "w"), default=float)
        rows.append(rec)
        print(f"L={a.L:g} kmax={a.kmax} N={N} seed={seed} K={K} M={M} r={rad:.0f}m cov={cov.mean():.2f} | val AUPRC: HiME-2 {ev['auprc']:.3f} "
              f"central {ref['auprc']:.3f} (S {ref['auprc_S']:.3f}, I {ref['auprc_I']:.3f}) prevalence {prev:.3f} "
              f"| {rec['seconds']:.0f}s", flush=True)
json.dump(rows, open(Path(a.out) / "summary.json", "w"), indent=1, default=float)
