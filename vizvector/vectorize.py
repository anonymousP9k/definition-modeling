"""Module for vectorization of the annotated dataset."""
import argparse
import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_json_dataset(input_file_path: str | Path,
                      limit: int | None = None) -> list[dict[str, str]]:
    """
    Load the dataset for inference.

    :param input_file_path: The path to the input file.
    :param limit: How many rows to load. Will load all if not passed.
    :return: The dataset.
    """
    data = []

    with open(input_file_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            json_object = json.loads(line)
            data.append(json_object)

            if limit and len(data) == limit:
                break

    return data


def vectorize_text(data: list[dict[str, str]], text_column: str,
                   model_name: str = 'cointegrated/rubert-tiny2') -> list[dict[str, Any]]:
    """
    Vectorize the text in the specified column of a dataset using sentence transformers.

    :param data: The dataset.
    :param text_column: The column name which contains the text to vectorize.
    :param model_name: The model to use for vectorization or path to it.
    :raises ValueError: If the data contains empty values.
    :return: The dataset with added vector column.
    """
    model = SentenceTransformer(model_name)
    for record in tqdm(data):
        text = record.get(text_column, "")
        if not text:
            raise ValueError(f"{text_column} column contains empty values.")
        vector = model.encode(text).tolist()
        record['vector'] = vector
    return data


def save_json_dataset(data: list[dict[str, Any]], output_file_path: str | Path) -> None:
    """
    Save the dataset to a JSON file.

    :param data: The dataset.
    :param output_file_path: The path to the output file.
    """
    if not Path(output_file_path).parent.exists():
        Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for record in data:
            json_line = json.dumps(record, ensure_ascii=False)
            output_file.write(f"{json_line}\n")


def main():
    """Vectorize the dataset."""
    parser = argparse.ArgumentParser(
        description="Vectorize text in a JSON file and save it with a new 'vector' column.")
    parser.add_argument("input_file",
                        help="The input JSON file path")
    parser.add_argument("output_file",
                        help="The output JSON file path")
    parser.add_argument("--text_column",
                        default="generated_text",
                        help="The column name with the text to vectorize "
                             "(default: 'generated_text')")
    parser.add_argument("--model-name",
                        default="cointegrated/rubert-tiny2",
                        help="The model name or path to it (default: 'cointegrated/rubert-tiny2')")
    parser.add_argument("--limit",
                        type=int,
                        help="Limit the number of records to process")

    args = parser.parse_args()

    input_file_path = Path(args.input_file)
    output_file_path = Path(args.output_file)
    text_column = args.text_column
    model_name = args.model_name
    limit = args.limit

    data = load_json_dataset(input_file_path, limit)
    data_with_vectors = vectorize_text(data, text_column, model_name)
    save_json_dataset(data_with_vectors, output_file_path)


if __name__ == "__main__":
    main()
