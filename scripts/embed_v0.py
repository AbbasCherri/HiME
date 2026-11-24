"""Compute the V0 embedding grid on the GPU (plan 2.3). Reads chips and tiles from the data root
read-only; writes to runs/embeddings/<encoder>/ and records sha256 in runs/embeddings/data_lock.json.
Usage: uv run python scripts/embed_v0.py [--encoders dinov2_vits14 ...] [--limit 2000]"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from hime import data as D
from hime.encoders import ENCODERS, embed

ap = argparse.ArgumentParser()
ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
ap.add_argument("--layer", default="seamlessphoto")
ap.add_argument("--out", default="runs/embeddings")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--limit", type=int, default=0, help="first n buildings only (throughput test)")
a = ap.parse_args()

root = D.data_root()
b = D.buildings(root)
if a.limit:
    b = b.iloc[: a.limit]
    a.out = str(Path(a.out) / f"limit{a.limit}")
lock_f = Path(a.out) / "data_lock.json"
lock = json.loads(lock_f.read_text()) if lock_f.exists() else {}
for enc in a.encoders:
    t0 = time.time()
    try:
        embed(enc, root, a.layer, b["building_id"].to_numpy(), b["lat"].to_numpy(), b["lon"].to_numpy(),
              a.out, batch=a.batch, workers=a.workers)
    except Exception as e:                       # e.g. gated DINOv3 weights without an accepted licence
        print(f"{enc}: FAILED ({type(e).__name__}: {e})", flush=True)
        continue
    dt = time.time() - t0
    for f in sorted((Path(a.out) / enc).glob("*.npy")):
        lock[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    lock_f.parent.mkdir(parents=True, exist_ok=True)
    lock_f.write_text(json.dumps(lock, indent=1, sort_keys=True))
    print(f"{enc}: {len(b)} buildings x 16 views in {dt / 60:.1f} min ({16 * len(b) / dt:.0f} images/s)", flush=True)
