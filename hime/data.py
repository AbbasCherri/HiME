"""Real-data layer (Phase 1). Reads the release read-only; nothing is ever written under the data root.

Layout (Hugging Face `AbbasABC/HiME-dataset`, mirrored at ~/HiME-dataset):
  derived/buildings.parquet, derived/folds.parquet, derived/splits/seed_<s>.parquet,
  derived/clients/N<N>_seed_<s>.parquet and N<N>_seed_<s>_meta.parquet,
  embeddings/<encoder>/<layer>.npy + <layer>.ids.npy, chips/<layer>/<id//1000>/<id>.png
"""
import os
import subprocess
from pathlib import Path

import numpy as np

SEED_ROLES = {"development": range(1000, 1005), "tuning": range(2000, 2005),
              "confirmation": range(3000, 3020), "replication": range(4000, 4020)}
TEST_ROLES = {"confirmation", "replication"}
# MMI_shape is undefined in every source the release checked (notodata source.py), so it is not used.
LOG_IM = ["PGA", "PGV", "SA_0_3", "SA_1_0", "SA_3_0"]
SEISMIC = LOG_IM + ["MMI_original"]
IM_FLOOR = 1e-6                                   # non-positive intensities are floored before the log


class DataBlocked(RuntimeError):
    """Raised when the dataset is absent; tests map this to the gate status `Blocked`."""


def data_root() -> Path:
    root = Path(os.environ.get("HIME_DATA", Path.home() / "HiME-dataset")).expanduser()
    for r in (root, root / "data"):
        if (r / "derived" / "buildings.parquet").exists():
            return r
    raise DataBlocked(f"derived/buildings.parquet not found under {root} (set HIME_DATA)")


def seed_role(seed: int) -> str:
    for role, r in SEED_ROLES.items():
        if seed in r:
            return role
    raise ValueError(f"seed {seed} has no registered role")


class TestVault:
    """Hands out test rows only for test-role seeds, on a clean tree tagged freeze-v1."""
    __test__ = False

    def __init__(self, test_index, tag="freeze-v1"):
        self._idx, self.tag = test_index, tag

    def frozen(self) -> bool:
        tags = subprocess.run(["git", "tag", "--points-at", "HEAD"], capture_output=True, text=True).stdout.split()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return self.tag in tags and not dirty

    def rows(self, seed: int):
        if seed_role(seed) not in TEST_ROLES:
            raise PermissionError(f"seed {seed} is {seed_role(seed)}: no test access")
        if not self.frozen():
            raise PermissionError("tree not frozen: test rows refused")
        return self._idx


def _pq(path):
    import pandas as pd
    return pd.read_parquet(path)


def buildings(root=None):
    return _pq(Path(root or data_root()) / "derived" / "buildings.parquet").sort_values("building_id", ignore_index=True)


def split(seed, root=None):
    return _pq(Path(root or data_root()) / "derived" / "splits" / f"seed_{seed}.parquet").rename(columns={"role": "split_role"})


def clients(N, seed, root=None):
    d = Path(root or data_root()) / "derived" / "clients"
    return (_pq(d / f"N{N}_seed_{seed}.parquet").rename(columns={"role": "client_role"}),
            _pq(d / f"N{N}_seed_{seed}_meta.parquet"))


def embeddings(encoder="dinov2_vits14", layer="seamlessphoto", root=None, emb_root=None):
    """Memory-mapped (ids, vectors). `emb_root` points at locally built V0 embeddings (never the data root)."""
    if encoder.startswith("v0:"):                 # "v0:<encoder>:<cls|cls+mean>:<none|d8>:<0|1>" (Phase 9 grid)
        from .encoders import features
        _, enc, pool, tta, ctx = encoder.split(":")
        return features(emb_root or "runs/embeddings", enc, pool, tta, ctx == "1")
    base = Path(emb_root) if emb_root else Path(root or data_root()) / "embeddings"
    d = base / encoder
    return np.load(d / f"{layer}.ids.npy"), np.load(d / f"{layer}.npy", mmap_mode="r")


def seismic_features(df):
    X = np.column_stack([np.log(np.maximum(df[c].to_numpy(float), IM_FLOOR)) for c in LOG_IM] + [df["MMI_original"].to_numpy(float)])
    return X, int(sum((df[c].to_numpy(float) <= 0).sum() for c in LOG_IM))


def world(N, seed, eval_role="val", encoder="dinov2_vits14", emb_root=None, root=None, direct_frac=0.2,
          exclude_blocks=None):
    """Build the dict that `pipeline.prepare` consumes. Only client rows (train, calib) and the evaluation
    rows are materialised. eval_role='val' for development/tuning; 'test' goes through the TestVault."""
    from .rng import stream
    root = root or data_root()
    b = buildings(root)
    sp = split(seed, root)
    cl, meta = clients(N, seed, root)
    b = b.merge(sp[["building_id", "split_role"]], on="building_id").merge(
        cl[["building_id", "client_id", "client_role"]], on="building_id", how="left")
    if exclude_blocks is not None:
        b = b[~b["block_id"].isin(exclude_blocks)]
    lab = b["y_collapse"].notna()
    member = lab & b["client_role"].isin(["train", "calib"]) & (b["client_id"] >= 0)
    ev = lab & (b["split_role"] == eval_role)
    if eval_role == "test":
        TestVault(np.flatnonzero(ev)).rows(seed)
    b = b[member | ev].reset_index(drop=True)
    member, ev = member[member | ev].to_numpy(), ev[member | ev].to_numpy()
    ids, E = embeddings(encoder, root=root, emb_root=emb_root)
    pos = np.searchsorted(ids, b["building_id"].to_numpy())
    assert (ids[pos] == b["building_id"].to_numpy()).all(), "embedding ids do not cover the rows"
    o = np.argsort(pos)                           # read rows in file order; one float64 copy of the rows only
    xI = np.empty((len(pos), E.shape[1]))
    xI[o] = E[pos[o]]
    xS, nonpos = seismic_features(b)
    cid = np.sort(meta["client_id"].unique())
    remap = {c: k for k, c in enumerate(cid)}
    client = np.where(member, b["client_id"].map(remap).fillna(-1).to_numpy(), -1).astype(int)
    meta = meta.set_index("client_id").loc[cid]
    mmi = b.set_index("building_id")["MMI_original"]
    med = meta["medoid_building_id"].map(lambda i: mmi.get(i, np.nan)).to_numpy()
    if np.isnan(med).any():                       # medoid not among loaded rows: fall back to full table
        full = buildings(root).set_index("building_id")["MMI_original"]
        med = meta["medoid_building_id"].map(full).to_numpy()
    direct = med <= np.quantile(med, direct_frac) if direct_frac > 0 else np.zeros(len(cid), bool)
    alpha = stream(seed, "availability_draw", N).beta(4, 2, size=len(cid))
    return dict(xS=xS, xI=xI, y=b["y_collapse"].to_numpy(float), client=client,
                calib=member & (b["client_role"] == "calib").to_numpy(), test=ev,
                client_xy=meta[["x_m", "y_m"]].to_numpy(float), direct=direct, alpha=alpha, N=len(cid),
                building_id=b["building_id"].to_numpy(), block_id=b["block_id"].to_numpy(),
                fold=b["fold"].to_numpy(), nonpositive_im=nonpos)
