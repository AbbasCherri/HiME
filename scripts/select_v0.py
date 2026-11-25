"""Phase 9.1: V0 feature selection by leave-fold-out validation AUPRC (plan 2.3, D2). Central image expert,
lambda = 1e-4; then lambda in {1e-5..1e-2} for the chosen cell. Validation folds only; no test rows read.
Splits repeat every 5 seeds, so seeds 1000-1004 give the 5 distinct validation folds.
Usage: uv run python scripts/select_v0.py [--encoders dinov2_vits14 dinov2_vitb14] [--workers 3]"""
import argparse
import itertools
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from hime import data as D
from hime.gpu import available, fit_newton
from hime.metrics import average_precision
from hime.pipeline import prepare

ap = argparse.ArgumentParser()
ap.add_argument("--encoders", nargs="+", default=None, help="default: every encoder found under --emb-root")
ap.add_argument("--emb-root", default="runs/embeddings")
ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
ap.add_argument("--rs", type=int, nargs="+", default=[64, 128, 256, 0], help="0 = full dimension")
ap.add_argument("--workers", type=int, default=2, help="each worker can need several GB for full-dim cells")
ap.add_argument("--out", default="runs/v0")
ap.add_argument("--fit", choices=["auto", "cuda", "cpu", "scipy"], default="auto",
                help="central fit: torch Newton on cuda/cpu, or the scipy reference; auto = cuda if available")
a = ap.parse_args()
FIT = ("cuda" if available() else "scipy") if a.fit == "auto" else a.fit
NEED = ["ids.npy"] + [f"{s}_{p}_{t}.npy" for s in ("chip", "ctx") for p in ("cls", "mean") for t in ("none", "d8")]


def usable(d):
    return d.is_dir() and all((d / f).exists() for f in NEED)


def encoders():
    root = Path(a.emb_root)
    cands = a.encoders or (sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else [])
    for e in cands:
        if not usable(root / e) and (a.encoders or not e.startswith("limit")):
            print(f"skipping {e}: incomplete grid under {root / e}", flush=True)
    return [e for e in cands if usable(root / e)]


def folds_of(seed):
    b = D.buildings()[["building_id", "fold"]].merge(D.split(seed), on="building_id")
    return int(b.loc[b.split_role == "test", "fold"].iloc[0]), int(b.loc[b.split_role == "val", "fold"].iloc[0])


def fit_cell(job):
    cell, seed, lams, rs = job
    w = D.world(100, seed, eval_role="val", encoder=cell, emb_root=a.emb_root)
    out = []
    for r in rs:
        if r and r >= w["xI"].shape[1]:
            continue
        pb = prepare(w, r=r or None)
        e, X, y = pb["experts"]["I"], pb["X"]["I"], pb["y"]
        tr = np.concatenate(pb["client_rows"] + [pb["calib_rows"]])     # central: whole labelled pool
        Xtr, Xte = X[tr], X[pb["test_rows"]]
        dim = X.shape[1] - 1
        del pb, X
        if FIT == "scipy":
            ths = []
            for lam in lams:
                e.lam = lam
                ths.append(e.fit_central(Xtr, y[tr]))
        else:
            ths = fit_newton(Xtr, y[tr], e.lp, lams, device=FIT)
        for lam, th in zip(lams, ths):
            out.append(dict(cell=cell, r=r or dim, lam=lam, seed=seed,
                            auprc=average_precision(y[w["test"]], Xte @ th)))
        del Xtr, Xte
    return out


def run(jobs):
    """Fresh spawned worker per job, so no worker carries one cell's memory into the next."""
    try:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
            return [x for rs in ex.map(fit_cell, jobs) for x in rs]
    except BrokenProcessPool:
        print(f"select_v0: a worker was killed abruptly (most likely out of memory) with --workers {a.workers}; "
              f"rerun with fewer workers, e.g. --workers 1", file=sys.stderr, flush=True)
        sys.exit(1)


def choose(res, f, key):
    """Leave-fold-out: seeds whose test and validation folds both differ from f; 1-SE rule, smallest dim."""
    ok = [s for s in a.seeds if f not in FOLDS[s]]
    tab = {}
    for rec in res:
        if rec["seed"] in ok:
            tab.setdefault(key(rec), []).append(rec["auprc"])
    if not tab:
        sys.exit(f"select_v0: no results for fold {f} from seeds {ok} (folds {FOLDS})")
    stats = {k: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0) for k, v in tab.items()}
    best = max(stats, key=lambda k: stats[k][0])
    near = [k for k in stats if stats[k][0] >= stats[best][0] - stats[best][1]]
    return min(near, key=lambda k: (k[1], -stats[k][0])), stats


if __name__ == "__main__":
    encs = encoders()
    if not encs:
        print(f"Blocked: no encoder under {a.emb_root} has the full V0 grid ({', '.join(NEED)}); "
              f"run scripts/embed_v0.py first", flush=True)
        sys.exit(1)
    CELLS = [f"v0:{e}:{p}:{t}:{c}" for e, p, t, c in itertools.product(encs, ("cls", "cls+mean"), ("none", "d8"), ("0", "1"))]
    print(f"encoders: {' '.join(encs)}; {len(CELLS)} cells x {len(a.seeds)} seeds, {a.workers} workers, fit {FIT}", flush=True)
    FOLDS = {s: folds_of(s) for s in a.seeds}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = run([(c, s, [1e-4], a.rs) for c in CELLS for s in a.seeds])
    json.dump(res, open(out / "grid.json", "w"), indent=1)
    picks = {}
    for f in range(5):
        (cell, r), stats = choose(res, f, lambda x: (x["cell"], x["r"]))
        rarg = 0 if r == max(x["r"] for x in res if x["cell"] == cell) and 0 in a.rs else r   # full dim is passed as 0
        lres = [x for x in run([(cell, s, [1e-5, 1e-4, 1e-3, 1e-2], [rarg]) for s in a.seeds]) if x["r"] == r]
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
