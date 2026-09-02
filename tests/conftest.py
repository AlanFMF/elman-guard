import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def synthetic_flows():
    from elmanguard.synthetic import SyntheticSpec, generate

    return generate(SyntheticSpec(n_benign_hosts=25, n_attack_hosts=10, seed=1))
