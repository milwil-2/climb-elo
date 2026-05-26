from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from climbing_elo.api.limiter import limiter
from climbing_elo.api.routes import router as html_router
from climbing_elo.api.v1_routes import router as v1_router
from climbing_elo.api.sse import router as sse_router
from climbing_elo.database import init_db

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app() -> FastAPI:
    init_db()

    application = FastAPI(
        title="Climbing ELO",
        version="0.1.0",
        description=(
            "ELO rating system for World Climbing competitions. "
            "HTML dashboard at `/` — REST API under `/api/v1/`."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Wire rate limiter into the app.
    # SlowAPIMiddleware applies default_limits to all routes that are NOT
    # decorated with @limiter.limit() (those handle their own stricter limits).
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    application.add_middleware(SlowAPIMiddleware)

    # Allow all origins — this is a public API.
    # POST is allowed for /api/v1/projections (idempotent: same input → same output,
    # bounded compute, no DB writes). Credentials are NOT allowed (default), so
    # the wildcard origin is safe.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Serve static files (styles.css, etc.) from src/climbing_elo/static/
    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    application.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    application.include_router(html_router)
    application.include_router(v1_router)
    application.include_router(sse_router)
    return application


app = create_app()
