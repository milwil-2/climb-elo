import os

# Issue #82: the SQLite fallback in ``climbing_elo.database.get_engine`` is
# gone. The test suite needs *something* for any test-module import that
# eagerly instantiates the FastAPI app at module load (`api/app.py` calls
# ``init_db()`` from ``create_app``). Point DATABASE_URL at a throwaway
# in-memory SQLite when running under pytest so the import succeeds; the
# individual tests still monkey-patch ``get_engine`` to use their own
# seeded DBs where needed.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from climbing_elo.models import (  # noqa: E402
    Athlete,
    Base,
    Discipline,
    Event,
    EventTier,
    Gender,
    Rating,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session
    session.close()


@pytest.fixture
def sample_event(db_session: Session) -> Event:
    from datetime import date

    event = Event(
        name="Test World Cup",
        tier=EventTier.WORLD_CUP,
        season=2024,
        start_date=date(2024, 6, 1),
        discipline=Discipline.LEAD,
    )
    db_session.add(event)
    db_session.flush()
    return event


@pytest.fixture
def eight_athletes(db_session: Session) -> list[Athlete]:
    athletes = []
    for i, name in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"]):
        a = Athlete(name=name, gender=Gender.M)
        db_session.add(a)
        athletes.append(a)
    db_session.flush()
    return athletes


@pytest.fixture
def eight_athletes_with_ratings(
    db_session: Session, eight_athletes: list[Athlete]
) -> list[Athlete]:
    mus = [1750, 1700, 1680, 1650, 1620, 1600, 1570, 1540]
    for athlete, mu in zip(eight_athletes, mus):
        db_session.add(
            Rating(
                athlete_id=athlete.id,
                discipline=Discipline.LEAD,
                mu=mu,
                sigma=100.0,
                n_events=10,
                provisional=False,
            )
        )
    db_session.flush()
    return eight_athletes
