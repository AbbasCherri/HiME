"""Is the imagery audit's flag real? Three checks on the buildings it compared (validation rows only):
1. near-identical chips: seamlessphoto vs nendophoto2022 cosine above 0.95 / 0.98 (same image) vs the same for 2024;
2. collapsed vs not: a post-event chip of a collapsed building should move away from 2022, so collapsed
   buildings should be flagged less often and have a lower cos(2022) - cos(2024);
3. image expert AUPRC on flagged vs unflagged vs not-compared validation rows (trained on train rows of the
   same seed; seeds 1000-1004 cover the 5 validation folds). Pre-event chips should push AUPRC to prevalence.
Reads the audit's embeddings in runs/imagery_audit. Never reads test-role rows.
Usage: uv run python scripts/audit_check.py [--cell v0:dinov3_vitb16:cls+mean:d8:0] [--r 64] [--lam 0.01]"""
import argparse
import json
from pathlib import Path

import numpy as np

from hime import data as D
from hime.experts import LinearExpert, logit
from hime.features import fed_standardise_pca, suff_stats, transform
from hime.gpu import available, design, fit_newton, pca_basis
from hime.metrics import average_precision

ap = argparse.ArgumentParser()
ap.add_argument("--audit", default="runs/imagery_audit")
ap.add_argument("--cell", default="v0:dinov3_vitb16:cls+mean:d8:0")
ap.add_argument("--emb-root", default="runs/embeddings")
ap.add_argument("--r", type=int, default=64)
ap.add_argument("--lam", type=float, default=0.01)
ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
a = ap.parse_args()
au = Path(a.audit)
rep = json.loads((au / "report.json").read_text())


def layer(name):
    d = au / name / "dinov2_vits14"
    E = np.load(d / "chip_cls_none.npy")
    return np.load(d / "ids.npy"), E / np.linalg.norm(E, axis=1, keepdims=True)


def cos_to(ref, q_ids, q_E):
    ref_ids, ref_E = ref
    pos = np.searchsorted(ref_ids, q_ids)
    ok = (pos < len(ref_ids)) & (ref_ids[np.minimum(pos, len(ref_ids) - 1)] == q_ids)
    c = np.full(len(q_ids), np.nan)
    c[ok] = (ref_E[pos[ok]] * q_E[ok]).sum(1)
    return c


# same cosines and flag rule as scripts/audit_imagery.py
s_ids, s_E = layer("seamlessphoto")
c22 = cos_to(layer("nendophoto2022"), s_ids, s_E)
c24 = np.full(len(s_ids), np.nan)
for L in rep["post_layers"]:
    c = cos_to(layer(L), s_ids, s_E)
    c24 = np.where(np.isnan(c), c24, c)
both = ~np.isnan(c22) & ~np.isnan(c24)
flag = both & (c22 > c24)
assert int(flag.sum()) == rep["flagged_buildings"], "flag rule does not reproduce the audit report"

b = D.buildings().set_index("building_id")
y_all = b["y_collapse"].reindex(s_ids).to_numpy(float)
val_any = np.zeros(len(s_ids), bool)
for s in a.seeds:
    sp = D.split(s).set_index("building_id")["split_role"].reindex(s_ids)
    val_any |= (sp == "val").to_numpy()

out = {}
q = [0.05, 0.25, 0.5, 0.75, 0.95]
out["quantiles"] = {k: np.quantile(v[both], q).round(4).tolist() for k, v in
                    (("cos_2022", c22), ("cos_2024", c24), ("cos_2022_minus_2024", c22 - c24))}
out["near_identical_share"] = {f"{k}>{t}": float((v[both] > t).mean()) for k, v in (("cos_2022", c22), ("cos_2024", c24))
                               for t in (0.95, 0.98)}
m = both & val_any & ~np.isnan(y_all)
out["by_collapse"] = {lab: dict(n=int((m & (y_all == y)).sum()), flagged_share=float(flag[m & (y_all == y)].mean()),
                                mean_cos_diff=float((c22 - c24)[m & (y_all == y)].mean()))
                      for lab, y in (("collapsed", 1.0), ("not_collapsed", 0.0))}

# 3. image expert, train rows -> validation rows, per seed
ids, E = D.embeddings(a.cell, emb_root=a.emb_root)
dev = "cuda" if available() else None
grp = {"flagged": [], "compared_unflagged": [], "not_compared": []}
ftab = dict(zip(s_ids, np.where(flag, "flagged", np.where(both, "compared_unflagged", "not_compared"))))
for s in a.seeds:
    t = b[["y_collapse"]].join(D.split(s).set_index("building_id")["split_role"])
    t = t[t["y_collapse"].notna() & t["split_role"].isin(["train", "val"])]
    pos = np.searchsorted(ids, t.index.to_numpy())
    X = E[pos].astype(float)
    y = t["y_collapse"].to_numpy(float)
    tr, va = (t["split_role"] == "train").to_numpy(), (t["split_role"] == "val").to_numpy()
    lp = logit(y[tr].mean())
    if dev:
        mu, sd, V = pca_basis(X[tr], dev)
        th = fit_newton(design(X[tr], mu, sd, V, a.r), y[tr], lp, [a.lam], device=dev)[0]
        g = design(X[va], mu, sd, V, a.r).cpu().numpy() @ th
    else:
        mu, sd, P = fed_standardise_pca([suff_stats(X[tr])], a.r)
        Xtr = np.c_[np.ones(tr.sum()), transform(X[tr], mu, sd, P)]
        th = LinearExpert(Xtr.shape[1], lp, a.lam).fit_central(Xtr, y[tr])
        g = np.c_[np.ones(va.sum()), transform(X[va], mu, sd, P)] @ th
    lab = np.array([ftab.get(i, "not_compared") for i in t.index[va]])
    for k in grp:
        sel = lab == k
        if sel.sum() and 0 < y[va][sel].sum() < sel.sum():
            grp[k].append((average_precision(y[va][sel], g[sel]), y[va][sel].mean(), int(sel.sum())))
out["image_expert_val_auprc"] = {k: dict(auprc=float(np.mean([x[0] for x in v])), prevalence=float(np.mean([x[1] for x in v])),
                                         lift=float(np.mean([x[0] / x[1] for x in v])), rows_per_seed=int(np.mean([x[2] for x in v])))
                                 for k, v in grp.items() if v}
(au / "check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
for k, v in out["image_expert_val_auprc"].items():
    print(f"audit check {k}: AUPRC {v['auprc']:.3f} prevalence {v['prevalence']:.3f} lift {v['lift']:.1f}x")
print(f"audit check near-identical to 2022 (cos>0.98): {out['near_identical_share']['cos_2022>0.98']:.3f}; "
      f"flagged share collapsed {out['by_collapse']['collapsed']['flagged_share']:.3f} vs not {out['by_collapse']['not_collapsed']['flagged_share']:.3f}")
