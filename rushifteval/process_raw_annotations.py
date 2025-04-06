"""Convert raw annotations to a dataset for inference."""
import argparse
import csv
from pathlib import Path
from typing import Any

from rushifteval_utils import AnnotatedWordPair, parse_path, save_as_json


def load_tsv_dataset(file_path: str | Path) -> list[AnnotatedWordPair]:
    """
    Load a TSV dataset and parses each row into an AnnotatedWord dataclass.

    :param file_path: The path to the TSV file.
    :return: A list of AnnotatedWord instances.
    """
    dataset = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            annotated_word = AnnotatedWordPair(
                word=row['word'],
                sent1=row['sent1'],
                sent2=row['sent2'],
                mean=float(row['mean'])
            )
            dataset.append(annotated_word)
    return dataset


def create_input_text(sentence: str, word: str) -> str:
    """
    Create the input text for the model inference.

    :param sentence: The sentence in which the word is used.
    :param word: The target word.
    :return: Formatted input text for the model.
    """
    sentence = sentence.replace("<b><i>", "").replace("</i></b>", "")
    return f"<LM> Контекст: \"{sentence}\" Определение слова \"{word}\": "


def prepare_dataset_for_inference(dataset: list[AnnotatedWordPair]) -> list[dict[str, Any]]:
    """
    Prepare the dataset for inference by creating an 'input_text' field and splitting examples.

    :param dataset: The original dataset loaded from the TSV file.
    :return: A list of dictionaries with the prepared data for inference.
    """
    prepared_data = []
    for idx, annotated_word in enumerate(dataset):
        for sentence_id, sentence in enumerate([annotated_word.sent1, annotated_word.sent2], 1):
            prepared_data.append({
                "id": idx,
                "sentence_id": sentence_id,
                "word": annotated_word.word,
                "sentence": sentence,
                "input_text": create_input_text(sentence, annotated_word.word)
            })
    return prepared_data


def main() -> None:
    """Process raw annotations."""
    parser = argparse.ArgumentParser(
        description="Prepare shifteval dataset for inference.")

    parser.add_argument("dataset_path",
                        type=str,
                        help="The path to the dataset. It can be either a .tsv file or"
                             "a directory containing multiple .tsv files.")
    parser.add_argument("output_path",
                        type=str,
                        help="The output path. Either a .jsonl file or a directory.")

    args = parser.parse_args()

    dataset_path = parse_path(args.dataset_path)
    output_path = parse_path(args.output_path)

    if not dataset_path.exists():
        raise ValueError(f"This dataset ({dataset_path}) does not exist.")

    if dataset_path.is_dir():
        for path in dataset_path.glob("*.tsv"):
            if path.name.startswith("._"):
                continue
            dataset = load_tsv_dataset(path)
            dataset_for_inference = prepare_dataset_for_inference(dataset)
            save_as_json(dataset_for_inference, parse_path(f"{output_path}/{path.stem}.jsonl"))
    else:
        dataset = load_tsv_dataset(dataset_path)
        dataset_for_inference = prepare_dataset_for_inference(dataset)
        save_as_json(dataset_for_inference, output_path)


if __name__ == "__main__":
    main()
