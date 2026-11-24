"""Run provenance: commit, patch hash and untracked-file hash."""
import hashlib
import json
import subprocess


def _git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout


def provenance() -> dict:
    untracked = _git("ls-files", "--others", "--exclude-standard").split()
    h = hashlib.sha256()
    for f in sorted(untracked):
        with open(f, "rb") as fh:
            h.update(f.encode() + fh.read())
    return {"commit": _git("rev-parse", "HEAD").strip(),
            "patch_sha256": hashlib.sha256(_git("diff", "HEAD").encode()).hexdigest(),
            "untracked_sha256": h.hexdigest(), "untracked": untracked}


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
