"""Phase 1 imagery audit: does each seamlessphoto chip look post-event? Compares DINOv2 embeddings of the
seamlessphoto chip with the dated post-event 2024 layers and with nendophoto2022. A building is flagged if
its chip is closer to 2022 than to its latest 2024 chip; a block is flagged if most of its compared
buildings are. Needs the dated layers' chips (build repo), not only the published seamlessphoto.
Usage: uv run python scripts/audit_imagery.py [--layers-root ~/HiME-dataset-build/data]"""
import argparse
import json
from pathlib import Path

import numpy as np

from hime import data as D
from hime.encoders import embed

ap = argparse.ArgumentParser()
ap.add_argument("--layers-root", default=str(Path.home() / "HiME-dataset-build" / "data"))
ap.add_argument("--out", default="runs/imagery_audit")
a = ap.parse_args()

lr = Path(a.layers_root).expanduser()
chips = lr / "chips"
post = sorted(p.name for p in chips.glob("20240102*")) if chips.exists() else []
if not post or not (chips / "nendophoto2022").exists():
    raise SystemExit(f"Blocked: dated 2024 layers or nendophoto2022 chips not found under {chips}")
root = D.data_root()
b = D.buildings(root)
ids, lat, lon = b["building_id"].to_numpy(), b["lat"].to_numpy(), b["lon"].to_numpy()
out = Path(a.out)


def emb(layer, src):
    have = np.array([(src / "chips" / layer / str(i // 1000) / f"{i}.png").exists() for i in ids])
    sel = ids[have]
    d = out / layer
    if not (d / "dinov2_vits14" / "chip_cls_none.npy").exists():
        embed("dinov2_vits14", src, layer, sel, lat[have], lon[have], d)
    E = np.load(d / "dinov2_vits14" / "chip_cls_none.npy")[:, :]
    return sel, E / np.linalg.norm(E, axis=1, keepdims=True)


def cos_to(ref_ids, ref_E, q_ids, q_E):
    pos = np.searchsorted(ref_ids, q_ids)
    ok = (pos < len(ref_ids)) & (ref_ids[np.minimum(pos, len(ref_ids) - 1)] == q_ids)
    c = np.full(len(q_ids), np.nan)
    c[ok] = (ref_E[pos[ok]] * q_E[ok]).sum(1)
    return c


s_ids, s_E = emb("seamlessphoto", root)
c22 = cos_to(*emb("nendophoto2022", lr), s_ids, s_E)
c24 = np.full(len(s_ids), np.nan)
for layer in post:                                          # latest dated layer wins (sorted by name/date)
    c = cos_to(*emb(layer, lr), s_ids, s_E)
    c24 = np.where(np.isnan(c), c24, c)
both = ~np.isnan(c22) & ~np.isnan(c24)
flag = both & (c22 > c24)
bb = b.set_index("building_id").loc[s_ids]
df = bb.assign(compared=both, flagged=flag)[["block_id", "fold", "compared", "flagged"]]
blk = df[df.compared].groupby("block_id")["flagged"].mean()
flag_blocks = blk[blk > 0.5].index
df["block_flagged"] = df["block_id"].isin(flag_blocks)
rep = dict(post_layers=post, compared=int(both.sum()), flagged_buildings=int(flag.sum()),
           flagged_blocks=int(len(flag_blocks)),
           per_fold=df.groupby("fold")[["compared", "flagged", "block_flagged"]].mean().to_dict())
out.mkdir(parents=True, exist_ok=True)
np.save(out / "flagged_blocks.npy", np.asarray(flag_blocks))
(out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
print(json.dumps(rep, indent=1, default=float))
