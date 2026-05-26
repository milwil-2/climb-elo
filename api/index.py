import sys
from pathlib import Path

# Make the climbing_elo package importable from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climbing_elo.api.app import create_app  # noqa: E402

app = create_app()
