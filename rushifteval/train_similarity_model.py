"""Training a linear regression model to convert distance to scores."""
import argparse

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from rushifteval_utils import (AnnotatedWordPair, compute_distance, load_annotated_data,
                               load_jsonl_vectors, parse_path)


def train_model(annotated_words: list[AnnotatedWordPair], metric: str,
                normalize_flag: bool) -> LinearRegression:
    """
    Train a logistic regression model to predict word similarity ratings.

    :param annotated_words: A list of AnnotatedWord instances with vector data.
    :param metric: The metric to use for computing distances between vectors.
    :param normalize_flag: Whether to normalize vectors before distance computation.
    :raises ValueError: If the vector is None.
    :return: A trained LinearRegression model.
    """
    distances = []
    for word in annotated_words:
        if word.vect1 is None or word.vect2 is None:
            raise ValueError("Vector is None!")
        distances.append(compute_distance(word.vect1, word.vect2, metric, normalize_flag))
    data = np.array(distances).reshape(-1, 1)
    targets = np.array([word.mean for word in annotated_words])
    model = LinearRegression()
    model.fit(data, targets)
    return model


def main() -> None:
    """Parse arguments and train a similarity model."""
    parser = argparse.ArgumentParser(
        description='Train a similarity rating model based on vector differences.'
    )
    parser.add_argument('--tsv_file', required=True, type=str,
                        help='Path to the TSV file containing annotations.')
    parser.add_argument('--jsonl_file', required=True, type=str,
                        help='Path to the JSONL file containing sentence vectors.')
    parser.add_argument('--metric', default='manhattan',
                        choices=["cosine", "manhattan", "euclidean", "hamming", "minkowski",
                                 "dot_product", "l2-squared"],
                        help='Distance metric to use.')
    parser.add_argument('--normalize', action='store_true',
                        help='Apply normalization to vectors before computing distance.')
    args = parser.parse_args()

    tsv_file_path = parse_path(args.tsv_file)
    jsonl_file_path = parse_path(args.jsonl_file)

    jsonl_vectors = load_jsonl_vectors(jsonl_file_path)
    annotated_words = load_annotated_data(tsv_file_path, jsonl_vectors)
    model = train_model(annotated_words, args.metric, args.normalize)

    if args.normalize:
        model_path = f'rushifteval/tmp/models/similarity_model_{args.metric}_normalize.joblib'
    else:
        model_path = f'rushifteval/tmp/models/similarity_model_{args.metric}.joblib'

    full_model_path = parse_path(model_path)

    if not full_model_path.parent.exists():
        full_model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, full_model_path)

    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()
