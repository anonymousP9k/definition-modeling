"""Utility functions for the vizvector package."""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


def parse_path(path: str | Path) -> Path:
    """
    Ensure that the path is absolute and is in a pathlib.Path format.

    :param path: The path to parse.
    :return: The parsed path.
    """
    path = Path(path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path
