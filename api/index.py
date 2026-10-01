import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climbing_elo.api.app import app  # noqa: E402, F401
