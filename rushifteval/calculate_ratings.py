"""Calculate scores for the dataset."""
import argparse
import csv
from pathlib import Path

from rushifteval_utils import (AnnotatedWordPair, compute_distance, load_vectorized_data,
                               normalize_distance, parse_path)


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


def process_and_annotate(jsonl_vectors_path: str | Path,
                         metric: str,
                         normalize_flag: bool, output_file_path: str | Path) -> None:
    """
    Process the TSV file and annotate it with a mean value based on a regression model.

    :param jsonl_vectors_path: The path to the JSONL file containing vectors.
    :param metric: The metric to use for computing distances between vectors.
    :param normalize_flag: A flag indicating whether to normalize vectors.
    :param output_file_path: The path to save the output TSV file.
    :raises ValueError: If the vector is empty.
    """
    annotated_data = load_vectorized_data(jsonl_vectors_path)

    for word in annotated_data:
        if word.vect1 is None or word.vect2 is None:
            raise ValueError("Vector is None!")
        word.mean = compute_distance(word.vect1, word.vect2, metric, normalize_flag)

    min_value = min(word.mean for word in annotated_data)
    max_value = max(word.mean for word in annotated_data)

    for word in annotated_data:
        word.mean = normalize_distance(word.mean, metric, min_value, max_value)

    save_annotated_data_to_tsv(annotated_data, output_file_path)


def main() -> None:
    """Parse arguments and call process_and_annotate function with the appropriate parameters."""
    parser = argparse.ArgumentParser(
        description="Process and annotate words with a mean value based on a regression model.")
    parser.add_argument("--jsonl", required=True,
                        help="Path to JSONL file containing vectors.")
    parser.add_argument("--output", required=True,
                        help="Path to save the output TSV file.")
    parser.add_argument("--metric", default="manhattan",
                        choices=["cosine", "manhattan", "euclidean", "hamming", "minkowski",
                                 "dot_product", "l2-squared"],
                        help="The metric to use for computing distances between vectors.")
    parser.add_argument("--normalize", action="store_true",
                        help="Whether to normalize vectors before distance computation.")

    args = parser.parse_args()

    jsonl_vectors_file_path = parse_path(args.jsonl)
    output_tsv_file_path = parse_path(args.output)

    process_and_annotate(
        jsonl_vectors_path=jsonl_vectors_file_path,
        metric=args.metric,
        normalize_flag=args.normalize,
        output_file_path=output_tsv_file_path
    )


if __name__ == "__main__":
    main()
