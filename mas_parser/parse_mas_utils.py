"""Module for utility functions used in the MAS parser."""
import json
from pathlib import Path

from pydantic.dataclasses import dataclass

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class DatasetEntry:
    """A class representing a dataset entry."""

    id: int
    title: str
    definitions: dict[str, dict[str, list[str]]]


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


@dataclass
class ParseMASConfig:
    """A class for representing the config for parsing the MAS dataset."""

    remove_tags: bool
    tags_to_remove: list[str]
    other_tags: list[str]
    ignore_entries: list[str]

    def __post_init__(self):
        self.tags_to_remove = sorted(self.tags_to_remove, key=len, reverse=True)


def load_parse_config(config_path: str | Path) -> ParseMASConfig:
    """
    Load the config with the settings of how to parse the dataset.

    :param config_path: The path to the config file.
    :return: The parse MAS config.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    return ParseMASConfig(**config)


@dataclass
class MasCleaningConfig:
    """A class for representing the cleaning config."""

    max_definition_character_length: int
    remove_entries_without_examples: bool
    throw_out_definition_markers: list[str]
    replace: dict[str, str]


def load_cleaning_config(config_path: str | Path) -> MasCleaningConfig:
    """
    Load the cleaning config from the given path.

    :param config_path: The path to the config.
    :return: The cleaning config.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    return MasCleaningConfig(**config)


def load_dataset(dataset_path: str | Path) -> list[dict[str, str | int | dict[str, list[str]]]]:
    """
    Load the dataset of jsonl format from the given path.

    :param dataset_path: The path to the dataset.
    :return: The dataset.
    """
    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        dataset = [json.loads(line) for line in dataset_file if line.strip()]

    return dataset


def dump_dataset(dataset: list[dict[str, str | int | dict[str, list[str]]]],
                 output_path: str | Path) -> None:
    """
    Dump the dataset to the given path.

    :param dataset: The dataset to dump.
    :param output_path: The path to dump the dataset to.
    """
    with open(output_path, "w", encoding="utf-8") as output_file:
        for entry in dataset:
            output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
