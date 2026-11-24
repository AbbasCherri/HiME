"""Phase 9.1: V0 feature selection by leave-fold-out validation AUPRC (plan 2.3, D2). Central image expert,
lambda = 1e-4; then lambda in {1e-5..1e-2} for the chosen cell. Validation folds only; no test rows read.
Splits repeat every 5 seeds, so seeds 1000-1004 give the 5 distinct validation folds.
Usage: uv run python scripts/select_v0.py [--encoders dinov2_vits14 dinov2_vitb14] [--workers 3]"""
import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from hime import data as D
from hime.metrics import average_precision
from hime.pipeline import prepare

ap = argparse.ArgumentParser()
ap.add_argument("--encoders", nargs="+", default=None, help="default: every encoder found under --emb-root")
ap.add_argument("--emb-root", default="runs/embeddings")
ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
ap.add_argument("--rs", type=int, nargs="+", default=[64, 128, 256, 0], help="0 = full dimension")
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--out", default="runs/v0")
a = ap.parse_args()
encs = a.encoders or sorted(p.name for p in Path(a.emb_root).iterdir() if (p / "chip_cls_d8.npy").exists())
CELLS = [f"v0:{e}:{p}:{t}:{c}" for e, p, t, c in itertools.product(encs, ("cls", "cls+mean"), ("none", "d8"), ("0", "1"))]


def folds_of(seed):
    b = D.buildings()[["building_id", "fold"]].merge(D.split(seed), on="building_id")
    return int(b.loc[b.split_role == "test", "fold"].iloc[0]), int(b.loc[b.split_role == "val", "fold"].iloc[0])


def fit_cell(job):
    cell, seed, lams = job
    w = D.world(100, seed, eval_role="val", encoder=cell, emb_root=a.emb_root)
    out = []
    for r in a.rs:
        if r and r >= w["xI"].shape[1]:
            continue
        pb = prepare(w, r=r or None)
        e, X, y = pb["experts"]["I"], pb["X"]["I"], pb["y"]
        tr = np.concatenate(pb["client_rows"] + [pb["calib_rows"]])     # central: whole labelled pool
        for lam in lams:
            e.lam = lam
            th = e.fit_central(X[tr], y[tr])
            out.append(dict(cell=cell, r=r or X.shape[1] - 1, lam=lam, seed=seed,
                            auprc=average_precision(y[pb["test_rows"]], X[pb["test_rows"]] @ th)))
    return out


def choose(res, f, key):
    """Leave-fold-out: seeds whose test and validation folds both differ from f; 1-SE rule, smallest dim."""
    ok = [s for s in a.seeds if f not in FOLDS[s]]
    tab = {}
    for rec in res:
        if rec["seed"] in ok:
            tab.setdefault(key(rec), []).append(rec["auprc"])
    stats = {k: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0) for k, v in tab.items()}
    best = max(stats, key=lambda k: stats[k][0])
    near = [k for k in stats if stats[k][0] >= stats[best][0] - stats[best][1]]
    return min(near, key=lambda k: (k[1], -stats[k][0])), stats


if __name__ == "__main__":
    FOLDS = {s: folds_of(s) for s in a.seeds}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = [r for rs in ex.map(fit_cell, [(c, s, [1e-4]) for c in CELLS for s in a.seeds]) for r in rs]
    json.dump(res, open(out / "grid.json", "w"), indent=1)
    picks = {}
    for f in range(5):
        (cell, r), stats = choose(res, f, lambda x: (x["cell"], x["r"]))
        with ProcessPoolExecutor(a.workers) as ex:
            lres = [x for rs in ex.map(fit_cell, [(cell, s, [1e-5, 1e-4, 1e-3, 1e-2]) for s in a.seeds]) for x in rs if x["r"] == r]
        lam = max({x["lam"] for x in lres}, key=lambda l: np.mean([x["auprc"] for x in lres if x["lam"] == l and f not in FOLDS[x["seed"]]]))
        picks[f] = dict(cell=cell, r=r, lam=lam, val_auprc=stats[(cell, r)][0], se=stats[(cell, r)][1])
        print(f"test fold {f}: {cell} r={r} lambda={lam:g} val AUPRC {stats[(cell, r)][0]:.3f} ± {stats[(cell, r)][1]:.3f}", flush=True)
    allm = {}
    for rec in res:
        allm.setdefault((rec["cell"], rec["r"]), []).append(rec["auprc"])
    top = sorted(allm, key=lambda k: -np.mean(allm[k]))[:10]
    print("top cells over all 5 validation folds:")
    for k in top:
        print(f"  {k[0]} r={k[1]}: {np.mean(allm[k]):.3f}")
    json.dump({"folds": FOLDS, "picks": picks}, open(out / "selection.json", "w"), indent=1, default=float)
