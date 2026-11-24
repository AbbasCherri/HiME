"""Phase 1 real-data contracts. Report `Blocked` (skip) until the dataset is mounted (HIME_DATA)."""
import pytest

from hime.data import DataBlocked, TestVault, data_root, seed_role


def test_P1_seed_roles_and_vault_guard():
    assert seed_role(1000) == "development" and seed_role(3005) == "confirmation"
    with pytest.raises(PermissionError):
        TestVault([1, 2]).rows(1000)


def test_P1_dataset_contracts():
    try:
        root = data_root()
    except DataBlocked as e:
        pytest.skip(f"Blocked: {e}")
    pytest.fail(f"Dataset found at {root}: write the Phase 1 contract tests against DATASET.md "
                "(counts 140,208 / 112,275 / 3,461 / 24,472; has_pre=0; buffer; folds; client files; imagery audit)")
