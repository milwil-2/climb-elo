import sys
import traceback
from pathlib import Path

# Make the climbing_elo package importable from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import PlainTextResponse  # noqa: E402

_startup_error: str | None = None

try:
    from climbing_elo.api.app import create_app  # noqa: E402
    app = create_app()
except Exception:
    _startup_error = traceback.format_exc()
    app = FastAPI()

    @app.get("/{path:path}")
    async def _error(path: str = "") -> PlainTextResponse:
        return PlainTextResponse(
            f"Startup failed:\n\n{_startup_error}", status_code=500
        )
