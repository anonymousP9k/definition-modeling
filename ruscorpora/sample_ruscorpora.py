"""Module to sample ruscorpora."""
import json
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from ruscorpora_utils import parse_path, write_results_to_file


def load_data(file_path: str | Path) -> list[dict]:
    """
    Load JSON Lines file into a list of dictionaries.

    :param file_path: Path to the JSON Lines file.
    :return: List of dictionaries representing the JSON Lines.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data


def sample_x_rows_per_word(data: list[dict], sample_size: int, seed: int) -> list[dict]:
    """
    Sample X rows for each unique word in the dataset.

    :param data: List of dictionaries representing the JSON Lines.
    :param sample_size: Number of entries to sample per word.
    :param seed: Seed for the random number generator.
    :return: List of sampled dictionaries.
    """
    if seed > 0:
        random.seed(seed)

    word_date_groups = defaultdict(list)
    for item in data:
        word_date_groups[(item['word'], item['date'])].append(item)

    sorted_word_date_keys = sorted(word_date_groups.keys(), key=lambda x: (x[0], x[1]))

    sampled_data = []
    for key in sorted_word_date_keys:
        items = word_date_groups[key]
        if len(items) <= sample_size:
            sampled_data.extend(items)
        else:
            if seed > 0:
                sampled_data.extend(random.sample(items, sample_size))
            else:
                sampled_data.extend(items[:sample_size])

    return sampled_data


def main():
    """Execute the sampling based on CLI arguments."""
    parser = ArgumentParser(
        description="Sample X rows for each unique word from a JSON Lines file.")
    parser.add_argument("file_path", help="Path to the JSON Lines file.")
    parser.add_argument("output_file_path",
                        help="Path for the output JSON Lines file.")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Number of entries to sample per word and date combination.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the random number generator. Set to -1 to turn it off.")
    args = parser.parse_args()

    path_to_data = parse_path(args.file_path)
    output_file_path = parse_path(args.output_file_path)

    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_data(path_to_data)
    sampled_data = sample_x_rows_per_word(data, args.sample_size, args.seed)
    write_results_to_file(sampled_data, output_file=output_file_path)


if __name__ == "__main__":
    main()
