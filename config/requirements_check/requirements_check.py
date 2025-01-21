"""Checks dependencies files."""
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


def get_paths() -> list[Path]:
    """Return list of paths to non-python files."""
    list_with_paths = []
    for file in ROOT_DIR.iterdir():
        if file.name in ['requirements.txt', 'requirements_ci.txt', 'requirements_train.txt']:
            list_with_paths.append(file)
    return list_with_paths


def get_requirements(path: str | Path) -> list:
    """
    Return a list of dependencies.

    :param path: Path to requirements file
    :return: List of dependencies
    """
    with open(path, 'r', encoding='utf-8') as requirements_file:
        lines = requirements_file.readlines()
    return [line.strip() for line in lines if line.strip()]


def compile_pattern() -> re.Pattern:
    """
    Return the compiled pattern.

    :return: Compiled pattern
    """
    return re.compile(r'\w+(-\w+|\[\w+\])*==\d+(\.\d+)+')


def check_dependencies(lines: list, compiled_pattern: re.Pattern) -> bool:
    """Check that dependencies conform to the template."""
    if sorted(lines) != lines:
        print('Dependencies do not conform to the template.')
        return False
    for line in lines:
        if not re.search(compiled_pattern, line):
            print('Dependencies do not conform to the template.')
            return False
    print('Dependencies: OK.')
    return True


def main() -> None:
    """Entrypoint for module."""
    paths = get_paths()
    compiled_pattern = compile_pattern()

    for path in paths:
        print(f"Checking {path} file...")
        lines = get_requirements(path)
        if not check_dependencies(lines, compiled_pattern):
            sys.exit(1)


if __name__ == '__main__':
    main()
