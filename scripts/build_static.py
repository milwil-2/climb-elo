"""Publish the app's static assets to Vercel's CDN during the build."""

from pathlib import Path
import shutil


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "src" / "climbing_elo" / "static"
    target = root / "public"
    shutil.rmtree(target / "static", ignore_errors=True)
    shutil.copytree(source, target / "static")
    shutil.copyfile(source / "favicon.svg", target / "favicon.ico")


if __name__ == "__main__":
    main()
