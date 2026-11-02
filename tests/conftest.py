import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Tests must never touch the network or the real data folder.
# (Set before anything imports fieldwise.config, which reads env at import.)
import tempfile  # noqa: E402

os.environ["FIELDWISE_DOTENV"] = "0"      # never read the developer's real .env (or API key) in tests
os.environ["FIELDWISE_DATA_DIR"] = tempfile.mkdtemp(prefix="fieldwise-test-")
os.environ.setdefault("FIELDWISE_FORCE_OFFLINE", "1")
os.environ.setdefault("MODEL_API_KEY", "")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_account_guard():
    """The Spark/SAM account guard is per process; give every test a clean one."""
    from fieldwise.providers.spark import reset_guard
    reset_guard()
    yield
    reset_guard()
