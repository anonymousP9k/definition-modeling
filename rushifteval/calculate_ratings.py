"""Calculate scores for the dataset."""
import argparse
import csv
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression

from rushifteval_utils import (AnnotatedWordPair, compute_distance, load_annotated_data,
                               load_jsonl_vectors, parse_path)


def load_model(regression_model_path: str | Path) -> LinearRegression:
    """
    Load a regression model from the specified path.

    :param regression_model_path: The path to the regression model file.
    :return: The loaded regression model.
    """
    regression_model_path = parse_path(regression_model_path)
    return joblib.load(regression_model_path)


def save_annotated_data_to_tsv(annotated_data: list[AnnotatedWordPair],
                               output_file_path: str | Path) -> None:
    """
    Save the annotated data to a TSV file.

    :param annotated_data: A list of AnnotatedWord instances containing the annotated data.
    :param output_file_path: The path to save the TSV output file.
    """
    if not Path(output_file_path).parent.exists():
        Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file_path, 'w', newline='', encoding="utf-8") as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        writer.writerow(['word', 'sent1', 'sent2', 'mean'])
        for annotated_word in annotated_data:
            writer.writerow([
                annotated_word.word,
                annotated_word.sent1,
                annotated_word.sent2,
                annotated_word.mean
            ])


def process_and_annotate(tsv_file_path: str | Path,  # pylint: disable=too-many-arguments
                         jsonl_vectors_path: str | Path,
                         regression_model_path: str | Path, metric: str,
                         normalize_flag: bool, output_file_path: str | Path) -> None:
    """
    Process the TSV file and annotate it with a mean value based on a regression model.

    :param tsv_file_path: The path to the input TSV file containing annotations.
    :param jsonl_vectors_path: The path to the JSONL file containing vectors.
    :param regression_model_path: The path to the regression model file.
    :param metric: The metric to use for computing distances between vectors.
    :param normalize_flag: A flag indicating whether to normalize vectors.
    :param output_file_path: The path to save the output TSV file.
    :raises ValueError: If the vector is empty.
    """
    jsonl_vectors = load_jsonl_vectors(jsonl_vectors_path)
    model = load_model(regression_model_path)

    annotated_data = load_annotated_data(tsv_file_path, jsonl_vectors)

    for word in annotated_data:
        if word.vect1 is None or word.vect2 is None:
            raise ValueError("Vector is None!")
        distance = compute_distance(word.vect1, word.vect2, metric, normalize_flag)
        word.mean = model.predict([[distance]])[0]

    save_annotated_data_to_tsv(annotated_data, output_file_path)


def main() -> None:
    """Parse arguments and call process_and_annotate function with the appropriate parameters."""
    parser = argparse.ArgumentParser(
        description="Process and annotate words with a mean value based on a regression model.")
    parser.add_argument("--tsv", required=True,
                        help="Path to input TSV file containing annotations.")
    parser.add_argument("--jsonl", required=True,
                        help="Path to JSONL file containing vectors.")
    parser.add_argument("--model", required=True,
                        help="Path to the regression model file.")
    parser.add_argument("--output", required=True,
                        help="Path to save the output TSV file.")
    parser.add_argument("--metric", default="manhattan",
                        choices=["cosine", "manhattan", "euclidean", "hamming", "minkowski",
                                 "dot_product", "l2-squared"],
                        help="The metric to use for computing distances between vectors.")
    parser.add_argument("--normalize", action="store_true",
                        help="Whether to normalize vectors before distance computation.")

    args = parser.parse_args()

    tsv_input_file_path = parse_path(args.tsv)
    jsonl_vectors_file_path = parse_path(args.jsonl)
    regression_model_file_path = parse_path(args.model)
    output_tsv_file_path = parse_path(args.output)

    process_and_annotate(
        tsv_file_path=tsv_input_file_path,
        jsonl_vectors_path=jsonl_vectors_file_path,
        regression_model_path=regression_model_file_path,
        metric=args.metric,
        normalize_flag=args.normalize,
        output_file_path=output_tsv_file_path
    )


if __name__ == "__main__":
    main()
