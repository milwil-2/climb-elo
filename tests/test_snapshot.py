"""
Tests for scripts/snapshot_db.py and scripts/restore_snapshot.py.

These tests exercise the snapshot/restore logic directly (without the GitHub
Release layer) so they run fully offline with no external dependencies.
"""

import gzip
import hashlib
import importlib.util
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers — load the scripts as modules without requiring them to be
# installed packages.
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


snapshot_mod = _load_script("snapshot_db")
restore_mod = _load_script("restore_snapshot")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_db(tmp_path: Path) -> Path:
    """Create a minimal SQLite database with the three key tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE events  (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE ratings (id INTEGER PRIMARY KEY, mu REAL);
        CREATE TABLE results (id INTEGER PRIMARY KEY, rank INTEGER);
        INSERT INTO events   VALUES (1, 'Test Event 1');
        INSERT INTO events   VALUES (2, 'Test Event 2');
        INSERT INTO ratings  VALUES (1, 1500.0);
        INSERT INTO results  VALUES (1, 1);
        INSERT INTO results  VALUES (2, 2);
        INSERT INTO results  VALUES (3, 3);
        """
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# snapshot_db tests
# ---------------------------------------------------------------------------


class TestSnapshotDb:
    def test_snapshot_creates_gz_file(self, fake_db: Path, tmp_path: Path):
        out_dir = tmp_path / "snapshots"
        snapshot_mod._backup_db(fake_db, tmp_path / "backup.db")
        # Use the public function chain: backup → gzip
        backup_path = tmp_path / "backup.db"
        gz_path = out_dir / "climbing_elo_2025-01-01.db.gz"
        out_dir.mkdir()
        snapshot_mod._gzip_file(backup_path, gz_path)
        assert gz_path.exists()
        assert gz_path.stat().st_size > 0

    def test_sha256_sidecar_matches(self, fake_db: Path, tmp_path: Path):
        """The sha256 computed by the module matches a manual computation."""
        out_dir = tmp_path / "snapshots"
        out_dir.mkdir()
        backup_path = tmp_path / "backup.db"
        gz_path = out_dir / "climbing_elo_2025-01-01.db.gz"

        snapshot_mod._backup_db(fake_db, backup_path)
        snapshot_mod._gzip_file(backup_path, gz_path)

        digest = snapshot_mod._sha256(gz_path)

        # Manual sha256
        h = hashlib.sha256()
        with open(gz_path, "rb") as f:
            h.update(f.read())
        assert digest == h.hexdigest()

    def test_gz_content_matches_original(self, fake_db: Path, tmp_path: Path):
        """Decompressing the gz yields a DB with identical row counts."""
        out_dir = tmp_path / "snapshots"
        out_dir.mkdir()
        backup_path = tmp_path / "backup.db"
        gz_path = out_dir / "climbing_elo_2025-01-01.db.gz"
        restored_path = tmp_path / "restored.db"

        snapshot_mod._backup_db(fake_db, backup_path)
        snapshot_mod._gzip_file(backup_path, gz_path)

        # Decompress manually
        import shutil

        with gzip.open(gz_path, "rb") as f_in, open(restored_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Compare row counts
        orig_counts = snapshot_mod._row_counts(fake_db)
        rest_counts = snapshot_mod._row_counts(restored_path)
        assert orig_counts == rest_counts

    def test_row_counts_returns_correct_values(self, fake_db: Path):
        counts = snapshot_mod._row_counts(fake_db)
        assert counts["events"] == 2
        assert counts["ratings"] == 1
        assert counts["results"] == 3

    def test_row_counts_missing_table(self, tmp_path: Path):
        """A DB without the expected tables returns zeros, not an exception."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()
        counts = snapshot_mod._row_counts(db_path)
        assert counts["events"] == 0
        assert counts["ratings"] == 0
        assert counts["results"] == 0

    def test_full_main_creates_expected_files(
        self, fake_db: Path, tmp_path: Path, monkeypatch
    ):
        """End-to-end: main() produces .db.gz and .sha256 files."""
        out_dir = tmp_path / "snapshots"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "snapshot_db.py",
                "--db-path",
                str(fake_db),
                "--out-dir",
                str(out_dir),
                "--date",
                "2025-06-01",
            ],
        )
        snapshot_mod.main()

        gz_path = out_dir / "climbing_elo_2025-06-01.db.gz"
        sha_path = out_dir / "climbing_elo_2025-06-01.db.gz.sha256"
        assert gz_path.exists(), "gz file not created"
        assert sha_path.exists(), "sha256 sidecar not created"

        # Verify the sidecar digest is correct
        actual = snapshot_mod._sha256(gz_path)
        sidecar_line = sha_path.read_text().strip()
        sidecar_digest = sidecar_line.split()[0]
        assert actual == sidecar_digest

    def test_main_exits_if_db_missing(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "snapshot_db.py",
                "--db-path",
                str(tmp_path / "nonexistent.db"),
                "--out-dir",
                str(tmp_path / "out"),
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            snapshot_mod.main()
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# restore_snapshot tests (offline — GitHub calls are mocked)
# ---------------------------------------------------------------------------


class TestRestoreSnapshot:
    def _make_release(self, tmp_path: Path, fake_db: Path, date_str: str = "2025-06-01"):
        """Build a fake gz + sha256 pair in tmp_path (simulating a release asset)."""
        backup_path = tmp_path / "backup.db"
        snapshot_mod._backup_db(fake_db, backup_path)

        gz_name = f"climbing_elo_{date_str}.db.gz"
        gz_path = tmp_path / gz_name
        snapshot_mod._gzip_file(backup_path, gz_path)

        sha_name = f"{gz_name}.sha256"
        sha_path = tmp_path / sha_name
        digest = snapshot_mod._sha256(gz_path)
        sha_path.write_text(f"{digest}  {gz_name}\n")

        return gz_path, sha_path, digest

    def test_verify_sha256_passes(self, fake_db: Path, tmp_path: Path):
        gz_path, sha_path, _ = self._make_release(tmp_path, fake_db)
        # Should not raise
        restore_mod._verify_sha256(gz_path, sha_path)

    def test_verify_sha256_fails_on_tamper(self, fake_db: Path, tmp_path: Path):
        gz_path, sha_path, _ = self._make_release(tmp_path, fake_db)
        # Corrupt the gz file
        with open(gz_path, "ab") as f:
            f.write(b"corruption")
        with pytest.raises(SystemExit):
            restore_mod._verify_sha256(gz_path, sha_path)

    def test_decompress_roundtrip(self, fake_db: Path, tmp_path: Path):
        gz_path, _, _ = self._make_release(tmp_path, fake_db)
        out_path = tmp_path / "restored.db"
        restore_mod._decompress(gz_path, out_path)
        assert out_path.exists()
        # Verify the restored DB has the expected tables
        counts = snapshot_mod._row_counts(out_path)
        assert counts["events"] == 2
        assert counts["ratings"] == 1
        assert counts["results"] == 3

    def test_latest_snapshot_date(self):
        assets = [
            "climbing_elo_2025-01-01.db.gz",
            "climbing_elo_2025-06-15.db.gz",
            "climbing_elo_2025-03-10.db.gz",
            "climbing_elo_2025-06-15.db.gz.sha256",  # sidecar — should be ignored
        ]
        latest = restore_mod._latest_snapshot_date(assets)
        assert latest == "2025-06-15"

    def test_latest_snapshot_date_no_assets(self):
        with pytest.raises(SystemExit):
            restore_mod._latest_snapshot_date([])

    def test_full_restore_roundtrip(
        self, fake_db: Path, tmp_path: Path, monkeypatch
    ):
        """
        End-to-end restore: mock gh CLI to serve local files, verify the
        restored DB is identical to the original.
        """
        date_str = "2025-06-01"
        gz_name = f"climbing_elo_{date_str}.db.gz"
        sha_name = f"{gz_name}.sha256"
        gz_path, sha_path, digest = self._make_release(tmp_path, fake_db, date_str)

        dest_db = tmp_path / "out" / "climbing_elo.db"
        dest_db.parent.mkdir()

        # Patch _list_release_assets to return our fake asset names
        monkeypatch.setattr(
            restore_mod,
            "_list_release_assets",
            lambda: [gz_name, sha_name],
        )

        # Patch _gh to copy the local test files into the download directory
        def fake_gh(*args, capture=False, **kwargs):
            # Find the --dir argument
            arg_list = list(args)
            if "--dir" in arg_list:
                dir_idx = arg_list.index("--dir") + 1
                dl_dir = Path(arg_list[dir_idx])
                import shutil

                shutil.copy(gz_path, dl_dir / gz_name)
                shutil.copy(sha_path, dl_dir / sha_name)
            mock = MagicMock()
            mock.stdout = ""
            return mock

        monkeypatch.setattr(restore_mod, "_gh", fake_gh)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "restore_snapshot.py",
                "--date",
                date_str,
                "--db-path",
                str(dest_db),
            ],
        )
        restore_mod.main()

        assert dest_db.exists()
        counts = snapshot_mod._row_counts(dest_db)
        assert counts["events"] == 2
        assert counts["results"] == 3

    def test_restore_backs_up_existing_db(
        self, fake_db: Path, tmp_path: Path, monkeypatch
    ):
        """Existing DB is moved to .bak before restore."""
        date_str = "2025-06-01"
        gz_name = f"climbing_elo_{date_str}.db.gz"
        sha_name = f"{gz_name}.sha256"
        gz_path, sha_path, _ = self._make_release(tmp_path, fake_db, date_str)

        dest_db = tmp_path / "out" / "climbing_elo.db"
        dest_db.parent.mkdir()
        # Pre-create an existing DB
        dest_db.write_bytes(b"old database content")

        monkeypatch.setattr(
            restore_mod,
            "_list_release_assets",
            lambda: [gz_name, sha_name],
        )

        def fake_gh(*args, capture=False, **kwargs):
            arg_list = list(args)
            if "--dir" in arg_list:
                dir_idx = arg_list.index("--dir") + 1
                dl_dir = Path(arg_list[dir_idx])
                import shutil

                shutil.copy(gz_path, dl_dir / gz_name)
                shutil.copy(sha_path, dl_dir / sha_name)
            mock = MagicMock()
            mock.stdout = ""
            return mock

        monkeypatch.setattr(restore_mod, "_gh", fake_gh)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "restore_snapshot.py",
                "--date",
                date_str,
                "--db-path",
                str(dest_db),
            ],
        )
        restore_mod.main()

        bak_path = dest_db.with_suffix(".db.bak")
        assert bak_path.exists(), ".bak file should exist"
        assert bak_path.read_bytes() == b"old database content"
