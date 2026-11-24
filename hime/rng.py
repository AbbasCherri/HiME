"""Keyed RNG streams: each purpose gets an independent stream, free of PYTHONHASHSEED."""
import hashlib

import numpy as np


def stream(seed: int, purpose: str, *keys) -> np.random.Generator:
    h = hashlib.sha256("/".join([purpose, *map(str, keys)]).encode()).digest()
    words = [int(seed)] + [int(x) for x in np.frombuffer(h[:16], dtype=np.uint32)]
    return np.random.default_rng(np.random.SeedSequence(words))
