"""Calculate mean score values."""
import argparse
from pathlib import Path

import pandas as pd

from rushifteval_utils import parse_path


def calculate_means(filenames: list[str | Path]) -> dict[str, list[float]]:
    """
    Calculate the mean values of the 'mean' column for each word in the given files.

    :param filenames: A list of filenames to process.
    :return: A dictionary where keys are words and values are lists of mean values.
    """
    data: dict[str, list[float]] = {}

    for filename in filenames:
        file_path = parse_path(filename)
        df = pd.read_csv(file_path, sep='\t', usecols=['word', 'mean'])
        df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
        word_means = df.groupby('word')['mean'].mean()

        for word, mean in word_means.items():
            if word not in data:
                data[str(word)] = []
            data[str(word)].append(float(mean))

    return data


def write_to_tsv(data: dict[str, list[float]], output_path: str | Path) -> None:
    """
    Write the data to a TSV file.

    :param data: A dictionary containing words and their corresponding mean values.
    :param output_path: The name of the output TSV file.
    """
    parsed_output_path = parse_path(output_path)

    if not parsed_output_path.parent.exists():
        parsed_output_path.parent.mkdir(parents=True, exist_ok=True)

    df_output = pd.DataFrame.from_dict(data, orient='index')
    df_output.to_csv(parsed_output_path, sep='\t', header=False, float_format='%.16f')


def main() -> None:
    """Parse command line arguments and generate the TSV file."""
    parser = argparse.ArgumentParser(description='Create a TSV file from multiple input files '
                                                 'containing word means.')
    parser.add_argument('input_files', nargs=3, help='Input files to process')
    parser.add_argument('output_file', help='Output TSV file')

    args = parser.parse_args()

    data = calculate_means(args.input_files)
    write_to_tsv(data, parse_path(args.output_file))


if __name__ == "__main__":
    main()
