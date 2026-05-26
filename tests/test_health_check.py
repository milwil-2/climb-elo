"""Tests for the health-check CLI (scripts/health_check_cli.py).

All tests use mocks — no real network calls are made.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: load the CLI module from its file path without relying on it being
# on sys.path as a package.
# ---------------------------------------------------------------------------

_CLI_PATH = Path(__file__).parent.parent / "scripts" / "health_check_cli.py"


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("health_check_cli", _CLI_PATH)
    assert spec is not None, f"Could not load {_CLI_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture()
def cli() -> ModuleType:
    return _load_cli()


# ---------------------------------------------------------------------------
# Tests: health_check() → exit code
# ---------------------------------------------------------------------------


class TestExitCodes:
    """CLI exits 0 on healthy, 1 on unhealthy."""

    def test_healthy_exits_zero(self, cli: ModuleType, capsys: pytest.CaptureFixture):
        with patch("climbing_elo.scraper.ifsc_api.health_check", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                cli.main.__module__  # ensure cli is loaded
                # Reload to pick up fresh state
                _cli = _load_cli()
                with (
                    patch.object(sys, "argv", ["health_check_cli.py"]),
                    patch(
                        "climbing_elo.scraper.ifsc_api.health_check", return_value=True
                    ),
                ):
                    _cli.main()
        assert exc_info.value.code == 0

    def test_unhealthy_exits_one(self, capsys: pytest.CaptureFixture):
        _cli = _load_cli()
        with (
            patch.object(sys, "argv", ["health_check_cli.py"]),
            patch("climbing_elo.scraper.ifsc_api.health_check", return_value=False),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _cli.main()
        assert exc_info.value.code == 1

    def test_healthy_quiet_no_output(self, capsys: pytest.CaptureFixture):
        _cli = _load_cli()
        with (
            patch.object(sys, "argv", ["health_check_cli.py", "--quiet"]),
            patch("climbing_elo.scraper.ifsc_api.health_check", return_value=True),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _cli.main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_unhealthy_quiet_no_output(self, capsys: pytest.CaptureFixture):
        _cli = _load_cli()
        with (
            patch.object(sys, "argv", ["health_check_cli.py", "--quiet"]),
            patch("climbing_elo.scraper.ifsc_api.health_check", return_value=False),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _cli.main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_healthy_prints_status(self, capsys: pytest.CaptureFixture):
        _cli = _load_cli()
        with (
            patch.object(sys, "argv", ["health_check_cli.py"]),
            patch("climbing_elo.scraper.ifsc_api.health_check", return_value=True),
        ):
            with pytest.raises(SystemExit):
                _cli.main()
        captured = capsys.readouterr()
        assert "HEALTHY" in captured.out

    def test_unhealthy_prints_status(self, capsys: pytest.CaptureFixture):
        _cli = _load_cli()
        with (
            patch.object(sys, "argv", ["health_check_cli.py"]),
            patch("climbing_elo.scraper.ifsc_api.health_check", return_value=False),
        ):
            with pytest.raises(SystemExit):
                _cli.main()
        captured = capsys.readouterr()
        assert "UNHEALTHY" in captured.out


# ---------------------------------------------------------------------------
# Tests: Discord webhook alerting
# ---------------------------------------------------------------------------


class TestDiscordWebhook:
    """Webhook is called only on failure and only once per cooldown window."""

    FAKE_WEBHOOK = "https://discord.com/api/webhooks/fake/token"

    def _run_cli(self, healthy: bool, extra_argv: list[str] | None = None) -> int:
        """Run main() with health_check mocked; return exit code."""
        _cli = _load_cli()
        argv = ["health_check_cli.py", "--webhook", self.FAKE_WEBHOOK]
        if extra_argv:
            argv.extend(extra_argv)
        with (
            patch.object(sys, "argv", argv),
            patch("climbing_elo.scraper.ifsc_api.health_check", return_value=healthy),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _cli.main()
        return exc_info.value.code

    def test_webhook_not_called_on_success(self):
        _cli = _load_cli()
        # Ensure no cooldown is in effect
        _cli.SENTINEL_FILE.unlink(missing_ok=True)

        with (
            patch("httpx.Client") as mock_httpx,
            patch.object(
                sys, "argv", ["health_check_cli.py", "--webhook", self.FAKE_WEBHOOK]
            ),
            patch("climbing_elo.scraper.ifsc_api.health_check", return_value=True),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _cli.main()

        assert exc_info.value.code == 0
        mock_httpx.assert_not_called()

    def test_webhook_called_on_failure(self):
        _cli = _load_cli()
        _cli.SENTINEL_FILE.unlink(missing_ok=True)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(return_value=None)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(return_value=mock_response)

        with (
            patch("httpx.Client", return_value=mock_client),
            patch.object(
                sys, "argv", ["health_check_cli.py", "--webhook", self.FAKE_WEBHOOK]
            ),
            patch("climbing_elo.scraper.ifsc_api.health_check", return_value=False),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _cli.main()

        assert exc_info.value.code == 1
        mock_client.post.assert_called_once()
        # Verify it was POSTed to the webhook URL
        call_args = mock_client.post.call_args
        assert call_args[0][0] == self.FAKE_WEBHOOK

    def test_webhook_payload_is_valid_json(self):
        _cli = _load_cli()
        _cli.SENTINEL_FILE.unlink(missing_ok=True)

        posted_payload: dict = {}

        def capture_post(url, content, headers):
            posted_payload.update(json.loads(content))
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock(return_value=None)
            return mock_resp

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(side_effect=capture_post)

        with (
            patch("httpx.Client", return_value=mock_client),
            patch.object(
                sys, "argv", ["health_check_cli.py", "--webhook", self.FAKE_WEBHOOK]
            ),
            patch("climbing_elo.scraper.ifsc_api.health_check", return_value=False),
        ):
            with pytest.raises(SystemExit):
                _cli.main()

        assert "embeds" in posted_payload
        assert len(posted_payload["embeds"]) == 1
        embed = posted_payload["embeds"][0]
        assert "FAILED" in embed["title"]

    def test_webhook_not_called_within_cooldown(self):
        """If the sentinel file records a recent alert, no second alert is sent."""
        _cli = _load_cli()
        # Simulate a very recent alert (2 seconds ago)
        _cli.SENTINEL_FILE.write_text(str(time.time() - 2))

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            patch("httpx.Client", return_value=mock_client),
            patch.object(
                sys, "argv", ["health_check_cli.py", "--webhook", self.FAKE_WEBHOOK]
            ),
            patch("climbing_elo.scraper.ifsc_api.health_check", return_value=False),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _cli.main()

        assert exc_info.value.code == 1
        # post() must NOT have been called — cooldown is active
        mock_client.post.assert_not_called()

        # Cleanup
        _cli.SENTINEL_FILE.unlink(missing_ok=True)

    def test_webhook_called_after_cooldown_expires(self):
        """After the cooldown elapses, a new alert is sent."""
        _cli = _load_cli()
        # Simulate an alert that happened 2 hours ago (past the 1-hour cooldown)
        _cli.SENTINEL_FILE.write_text(str(time.time() - 7201))

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(return_value=None)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(return_value=mock_response)

        with (
            patch("httpx.Client", return_value=mock_client),
            patch.object(
                sys, "argv", ["health_check_cli.py", "--webhook", self.FAKE_WEBHOOK]
            ),
            patch("climbing_elo.scraper.ifsc_api.health_check", return_value=False),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _cli.main()

        assert exc_info.value.code == 1
        mock_client.post.assert_called_once()

        # Cleanup
        _cli.SENTINEL_FILE.unlink(missing_ok=True)

    def test_no_webhook_flag_no_post_on_failure(self):
        """Without --webhook, failure exits 1 but no HTTP call is made."""
        _cli = _load_cli()
        _cli.SENTINEL_FILE.unlink(missing_ok=True)

        with (
            patch("httpx.Client") as mock_httpx,
            patch.object(sys, "argv", ["health_check_cli.py"]),
            patch("climbing_elo.scraper.ifsc_api.health_check", return_value=False),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _cli.main()

        assert exc_info.value.code == 1
        mock_httpx.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _should_send_alert / _record_alert_sent helpers
# ---------------------------------------------------------------------------


class TestAlertRateLimiting:
    def test_should_send_when_no_sentinel(self, tmp_path: Path):
        _cli = _load_cli()
        sentinel = tmp_path / "test_sentinel"
        with patch.object(_cli, "SENTINEL_FILE", sentinel):
            assert _cli._should_send_alert() is True

    def test_should_not_send_within_cooldown(self, tmp_path: Path):
        _cli = _load_cli()
        sentinel = tmp_path / "test_sentinel"
        sentinel.write_text(str(time.time() - 10))  # 10 seconds ago
        with patch.object(_cli, "SENTINEL_FILE", sentinel):
            assert _cli._should_send_alert() is False

    def test_should_send_after_cooldown(self, tmp_path: Path):
        _cli = _load_cli()
        sentinel = tmp_path / "test_sentinel"
        sentinel.write_text(str(time.time() - 7201))  # 2 hours ago
        with patch.object(_cli, "SENTINEL_FILE", sentinel):
            assert _cli._should_send_alert() is True

    def test_record_alert_updates_sentinel(self, tmp_path: Path):
        _cli = _load_cli()
        sentinel = tmp_path / "test_sentinel"
        before = time.time()
        with patch.object(_cli, "SENTINEL_FILE", sentinel):
            _cli._record_alert_sent()
        after = time.time()
        recorded = float(sentinel.read_text().strip())
        assert before <= recorded <= after
