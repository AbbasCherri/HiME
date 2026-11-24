import os
import subprocess
import sys

from hime.provenance import config_hash
from hime.rng import stream


def test_P0_streams_independent_and_reproducible():
    a, b = stream(1, "availability").random(5), stream(1, "sampling").random(5)
    assert (a != b).all()
    assert (stream(1, "availability").random(5) == a).all()


def test_P0_stream_free_of_hash_seed():
    code = "from hime.rng import stream;print(stream(3,'x',1).integers(1<<30))"
    outs = {subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": h},
                           capture_output=True, text=True).stdout for h in ("0", "123")}
    assert len(outs) == 1


def test_P0_config_hash_order_free():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
