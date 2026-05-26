from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from climbing_elo.models import Base

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2].parent / "data" / "climbing_elo.db"


def get_engine(db_path: Path | str = DEFAULT_DB_PATH):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def get_session_factory(db_path: Path | str = DEFAULT_DB_PATH) -> sessionmaker[Session]:
    engine = get_engine(db_path)
    return sessionmaker(bind=engine)


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> sessionmaker[Session]:
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
