"""Utility module for ruscorpora."""
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


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


def write_results_to_file(results: list[dict], output_file: str | Path) -> None:
    """
    Write the results to a JSON lines file.

    :param results: List of dictionaries, each representing a found instance.
    :param output_file: Path to the output file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in results:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
