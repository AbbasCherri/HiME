"""Real-data guards. The loader itself is written once DATASET.md and the schema are seen (Phase 1)."""
import os
import subprocess
from pathlib import Path

SEED_ROLES = {"development": range(1000, 1005), "tuning": range(2000, 2005),
              "confirmation": range(3000, 3020), "replication": range(4000, 4020)}
TEST_ROLES = {"confirmation", "replication"}


class DataBlocked(RuntimeError):
    """Raised when the dataset is absent; tests map this to the gate status `Blocked`."""


def data_root() -> Path:
    root = Path(os.environ.get("HIME_DATA", "data"))
    if not (root / "DATASET.md").exists():
        raise DataBlocked(f"dataset not found at {root} (set HIME_DATA)")
    return root


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
