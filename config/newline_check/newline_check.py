"""Checks newline at the end of the files."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def filter_paths_to_check(paths: list[Path]) -> list[Path]:
    """
    Get all paths available to check.

    :param paths: List of paths
    :return: List of paths to check
    """
    paths_to_exclude = [
        "venv",
        ".git",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        "build",
        "__init__.py",
        "htmlcov"
    ]
    extensions_to_check = [
        ".py",
        ".json",
        ".jsonl",
        ".md",
        ".sh",
        ".yaml",
        ".toml",
        ".gitignore",
        ".yml"
    ]
    paths_to_check = []
    for path in paths:
        for path_to_exclude in paths_to_exclude:
            if path_to_exclude in path.parts:
                break
        else:
            if path.suffix in extensions_to_check:
                paths_to_check.append(path)
    return paths_to_check


def check_paths(paths: list[Path]) -> bool:
    """
    Check for a newline at the end of the files.

    :param paths: A list of paths to check
    :return: True if all files have a newline at the end, False otherwise
    """
    files_without_newline = []

    for path in paths:
        print(f'Analyzing {path}')
        with open(path, encoding='utf-8') as file:
            lines = file.readlines()
        if lines[-1][-1] != '\n':
            files_without_newline.append(path)

    if files_without_newline:
        print('The following files do not have a newline at the end:')
        for path in files_without_newline:
            print(path)
        return False
    print('All files conform to the template.')
    return True


def main() -> None:
    """Entrypoint for module."""
    paths = filter_paths_to_check(list(PROJECT_ROOT.rglob("*")))
    sys.exit(not check_paths(paths))


if __name__ == '__main__':
    main()
