"""A module for functions that process data for the model."""
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def prepare_row(row: dict) -> dict:
    """
    Convert a row from the dataset to a readable format.

    :param row: The row to prepare.
    :return: The prepared row.
    """
    input_text = f"<LM>Контекст: \"{row['context']}\" Определение слова \"{row['word']}\": "
    target_text = row['definition']
    return {"word": row['word'], "input_text": input_text, "target_text": target_text}


def prepare_dataset(filepath: str | Path, output_directory: str | Path,
                    random_state: int = 42, train_size: float = 0.8) -> None:
    """
    Load and create a dataset from a jsonl file.

    :param filepath: The path to the jsonl file.
    :param output_directory: The directory to save the splits to.
    :param random_state: The random state for the train/test/validation split.
    :param train_size: The relative size of the train split.
    """
    if not Path(output_directory).exists():
        Path(output_directory).mkdir(parents=True, exist_ok=True)

    rows = load_dump(filepath)

    df = pd.DataFrame(rows)

    new_df = pd.DataFrame([prepare_row(row.to_dict()) for _, row in df.iterrows()])

    train_df, temp_df = train_test_split(new_df, test_size=1-train_size, random_state=random_state)
    test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)

    train_df.to_json(Path(output_directory) / "train.jsonl",
                     orient="records",
                     lines=True,
                     force_ascii=False)
    test_df.to_json(Path(output_directory) / "test.jsonl",
                    orient="records",
                    lines=True,
                    force_ascii=False)
    val_df.to_json(Path(output_directory) / "val.jsonl",
                   orient="records",
                   lines=True,
                   force_ascii=False)


def load_dump(filepath: str | Path) -> list[dict]:
    """
    Load a dataset from a jsonl file.

    :param filepath: The path to the jsonl file.
    :return: The dataset.
    """
    rows = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            json_line = json.loads(line)
            for definition in json_line["definitions"]:
                examples = json_line["definitions"][definition]["examples"]
                for example in examples:
                    rows.append({"word": json_line["title"],
                                 "definition": definition,
                                 "context": example})

    return rows


def load_dataset_split(directory: str | Path, debug_mode: bool = False) -> DatasetDict:
    """
    Load the train, test and validation splits of the dataset.

    :param directory: The directory containing the splits.
    :param debug_mode: If True, only load a small subset of the data.
    :return: The splits.
    """
    directory_path = Path(directory)

    train_df = pd.read_json(directory_path / "train.jsonl", lines=True)
    test_df = pd.read_json(directory_path / "test.jsonl", lines=True)
    val_df = pd.read_json(directory_path / "val.jsonl", lines=True)

    if debug_mode:
        train_df = train_df[:100]
        test_df = test_df[:20]
        val_df = val_df[:20]

    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df),
        "val": Dataset.from_pandas(val_df)
    })
