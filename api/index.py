import sys
import traceback
from pathlib import Path

# Make the climbing_elo package importable from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import PlainTextResponse  # noqa: E402

# Declare at module top-level so Vercel's static analyzer can find it.
# Overridden below if create_app() succeeds.
_startup_error: str | None = None
app = FastAPI()

try:
    from climbing_elo.api.app import create_app  # noqa: E402
    app = create_app()
except Exception:
    _startup_error = traceback.format_exc()

    @app.get("/{path:path}")
    async def _error(path: str = "") -> PlainTextResponse:
        return PlainTextResponse(
            f"Startup failed:\n\n{_startup_error}", status_code=500
        )
