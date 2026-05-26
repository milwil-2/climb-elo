from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from climbing_elo.api.routes import router
from climbing_elo.database import init_db

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def create_app() -> FastAPI:
    init_db()

    application = FastAPI(title="Climbing ELO", version="0.1.0")
    application.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    application.include_router(router)
    return application


app = create_app()
