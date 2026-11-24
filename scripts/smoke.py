"""Smoke run on the synthetic oracle world (Phase 7 exit: < 5 min on CPU). Synthetic: demonstrates the
pipeline runs; it does not reproduce the paper's results."""
import json
import os
import time
from dataclasses import replace

import numpy as np

from hime.engine import Config, central, evaluate, run
from hime.oracle import make_world
from hime.pipeline import prepare

t0 = time.time()
w = make_world(n=2000, N=50, d_i=16, seed=1000)
pb = prepare(w, r=None)
uav = np.random.default_rng(0).uniform(0, w["extent"], (4, 2))
ts = central(pb)
base = Config(T=100, M=12, seed=1000)
arms = {"hime2": base, "uniform": replace(base, sampler="uniform"), "v1_sampler": replace(base, sampler="v1"),
        "static_uav": replace(base, placement="static"), "no_correction_k2": replace(base, kappa=(2, 2), correction=False),
        "hime2_k2": replace(base, kappa=(2, 2))}
out = {"central": evaluate(pb, ts)}
for k, cfg in arms.items():
    r = run(pb, cfg, uav, theta_star=ts)
    out[k] = {**evaluate(pb, r["theta"]), "bytes": r["bytes"], "Tp": r["Tp"],
              "dist": {m: float(np.linalg.norm(r["theta"][m] - ts[m])) for m in ts}}
out["seconds"] = time.time() - t0
os.makedirs("runs", exist_ok=True)
json.dump(out, open("runs/smoke.json", "w"), indent=1)
for k, v in out.items():
    if isinstance(v, dict):
        print(f"{k:18s} AUPRC {v['auprc']:.3f}  prevalence {v['prevalence']:.3f}")
print(f"{out['seconds']:.1f} s")
assert out["hime2"]["auprc"] > 3 * out["hime2"]["prevalence"], "degenerate run"
assert out["seconds"] < 300
