"""Phase 1 real-data contracts. Skip as `Blocked` when the release is absent (HIME_DATA or ~/HiME-dataset)."""
import numpy as np
import pytest

from hime import data as D


def test_P1_seed_roles_and_vault_guard():
    assert D.seed_role(1000) == "development" and D.seed_role(3005) == "confirmation"
    with pytest.raises(PermissionError):
        D.TestVault([1, 2]).rows(1000)


@pytest.fixture(scope="module")
def root():
    try:
        return D.data_root()
    except D.DataBlocked as e:
        pytest.skip(f"Blocked: {e}")


@pytest.fixture(scope="module")
def B(root):
    return D.buildings(root)


def test_P1_counts_labels_and_pre(B):
    assert len(B) == 140_208 and B["building_id"].is_unique
    y = B["y_collapse"]
    assert (y == 0).sum() == 112_275 and (y == 1).sum() == 3_461 and y.isna().sum() == 24_472
    assert (B["has_pre"] == 0).all()                                  # no pre-event chips: pattern is always I
    print("has_post share:", B["has_post"].mean(), "post_status:", B["post_status"].value_counts().to_dict())


def test_P1_seismic_columns(B):
    X, nonpos = D.seismic_features(B)
    assert np.isfinite(X).all() and "MMI_shape" not in D.SEISMIC
    print("non-positive intensity values floored:", nonpos)


def test_P1_embeddings_aligned(root, B):
    ids, E = D.embeddings(root=root)
    assert E.shape == (140_208, 384) and (np.diff(ids) > 0).all()
    assert np.array_equal(ids, B["building_id"].to_numpy())
    rng = np.random.default_rng(0); k = rng.choice(len(ids), 200, replace=False)
    assert np.isfinite(np.asarray(E[np.sort(k)])).all()


def test_P1_folds_blocks_and_fold0(root, B):
    f = D._pq(root / "derived" / "folds.parquet")
    assert (f.groupby("block_id")["fold"].nunique() == 1).all()       # no block in two folds
    assert set(B["fold"].unique()) == set(range(5))
    f0 = B[B["fold"] == 0]
    print("fold 0:", len(f0), "collapse share among labelled:", f0["y_collapse"].mean())
    assert len(f0) == 14_344


@pytest.mark.parametrize("seed", [1000, 1001, 2003, 3000, 4019])
def test_P1_split_roles_and_buffer(root, B, seed):
    from scipy.spatial import cKDTree
    s = D.split(seed, root).merge(B[["building_id", "x_m", "y_m", "fold"]], on="building_id")
    assert len(s) == len(B) and set(s["split_role"]) <= {"test", "val", "train_pool", "buffer_dropped"}
    ft, fv = (s.loc[s["split_role"] == r, "fold"].unique() for r in ("test", "val"))
    assert len(ft) == 1 and len(fv) == 1 and ft[0] != fv[0]
    held = s["split_role"].isin(["test", "val"]).to_numpy()
    d, _ = cKDTree(s.loc[held, ["x_m", "y_m"]].to_numpy()).query(s.loc[~held, ["x_m", "y_m"]].to_numpy())
    pool = (s.loc[~held, "split_role"] == "train_pool").to_numpy()
    assert d[pool].min() >= 500.0 and d[~pool].max() < 500.0           # recomputed 500 m buffer
    assert np.allclose(d, s.loc[~held, "nearest_heldout_m"].to_numpy(), atol=1e-6)


def test_P1_splits_repeat_mod5_clients_do_not(root):
    a, b, c = (D.split(s, root).sort_values("building_id") for s in (1000, 1005, 1001))
    assert (a["split_role"].to_numpy() == b["split_role"].to_numpy()).all()
    assert (a["split_role"].to_numpy() != c["split_role"].to_numpy()).any()
    x, _ = D.clients(100, 1000, root); y, _ = D.clients(100, 1005, root)
    assert not x.sort_values("building_id")["client_id"].equals(y.sort_values("building_id")["client_id"])


@pytest.mark.parametrize("N,seed", [(50, 1000), (100, 1000), (200, 2001)])
def test_P1_clients_subset_calib_and_meta(root, B, N, seed):
    cl, meta = D.clients(N, seed, root)
    sp = D.split(seed, root)
    pool = set(sp.loc[sp["split_role"] == "train_pool", "building_id"])
    assert set(cl["building_id"]) <= pool
    y = cl.merge(B[["building_id", "y_collapse"]], on="building_id")
    kept = y[y["client_id"] >= 0]
    lab = kept[kept["client_role"].isin(["train", "calib"])]
    share = lab.groupby("client_id")["client_role"].apply(lambda r: (r == "calib").mean())
    print(f"N={N} seed={seed}: clients {len(meta)}, calib share min {share.min():.3f} max {share.max():.3f}")
    assert share.between(0.1, 0.3).all()
    g = lab.groupby("client_id")
    rec = g.apply(lambda d: ((d["client_role"] == "calib") & (d["y_collapse"] == 1)).sum())
    assert (rec.reindex(meta["client_id"]).to_numpy() == meta["calib_collapsed_I"].to_numpy()).all()


def test_P1_single_class_calibration_clients(root):
    _, meta = D.clients(100, 1000, root)
    n = int((meta["calib_collapsed_I"] == 0).sum())
    print("N=100 seed=1000 clients without a collapsed calibration row:", n, "of", len(meta))
    assert n == 63


def test_P1_world_adapter(root):
    w = D.world(100, 1000, eval_role="val", root=root)
    assert w["xS"].shape[1] == 6 and w["xI"].shape[1] == 384
    assert not np.isnan(w["y"]).any() and w["test"].any() and (w["client"][w["test"]] == -1).all()
    assert w["calib"].sum() > 0 and w["N"] == len(w["client_xy"])
    with pytest.raises(PermissionError):
        D.world(100, 3000, eval_role="test", root=root)
