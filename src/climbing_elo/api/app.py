from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from climbing_elo.api.routes import router as html_router
from climbing_elo.api.v1_routes import router as v1_router
from climbing_elo.api.sse import router as sse_router
from climbing_elo.database import init_db

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


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

    # Allow all origins — this is a read-only public API
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    application.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    application.include_router(html_router)
    application.include_router(v1_router)
    application.include_router(sse_router)
    return application


app = create_app()
