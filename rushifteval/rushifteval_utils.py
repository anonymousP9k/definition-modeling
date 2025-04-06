"""Utility module for rushifteval."""
import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic.dataclasses import dataclass
from scipy.spatial.distance import hamming, minkowski
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances
from sklearn.preprocessing import normalize

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class AnnotatedWordPair:
    """
    A dataclass representing a row in the RuShiftEval-like dataset.

    :param word: The target word.
    :param sent1: The first sentence.
    :param sent2: The second sentence.
    :param mean: The mean annotation value.
    :param definition1: The definition for the usage in the first sentence.
    :param definition2: The definition for the usage in the second sentence.
    :param vect1: Embedding for the usage in the first sentence.
    :param vect2: Embedding for the usage in the second sentence.
    """

    word: str
    sent1: str
    sent2: str
    mean: Optional[float] = None
    definition1: Optional[str] = None
    definition2: Optional[str] = None
    vect1: Optional[list[float]] = None
    vect2: Optional[list[float]] = None


def load_jsonl_vectors(
        jsonl_file_path: str | Path) -> dict[tuple[int, int], dict[str, list[float] | str]]:
    """
    Load vectors from a JSONL file.

    :param jsonl_file_path: Path to the JSONL file containing sentence vectors.
    :return: A dictionary with keys as (id, sentence_id) and values as dictionaries with 'vector'.
    """
    jsonl_data = {}
    with open(jsonl_file_path, 'r', encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            key = (data['id'], data['sentence_id'])
            jsonl_data[key] = {
                "vector": data.get('vector', []),
                "definition": data.get('generated_text', '')
            }
    return jsonl_data


def normalize_distance(distance: float, metric: str, min_value: float, max_value: float) -> float:
    """
    Normalize the distance based on the metric by negating and scaling it between 1 and 4.

    :param distance: The computed distance value.
    :param metric: The metric used to compute the distance.
    :param min_value: The minimum distance value for the metric.
    :param max_value: The maximum distance value for the metric.
    :return: The normalized distance between 1 and 4.
    :raises ValueError: If the metric is unsupported.
    """
    if metric != "dot_product":
        distance = -distance
        min_value, max_value = -max_value, -min_value

    if min_value == max_value:
        raise ValueError("Min and max values are the same, cannot normalize.")

    return 1 + 3 * (distance - min_value) / (max_value - min_value)


def compute_distance(vect1: list[float],  # pylint: disable=too-many-return-statements
                     vect2: list[float],
                     metric: str,
                     normalize_flag: bool) -> float:
    """
    Compute distance between two vectors using the specified metric.

    :param vect1: The first vector.
    :param vect2: The second vector.
    :param metric: The metric to use for distance computation.
    :param normalize_flag: Whether to normalize the vectors before computing distance.
    :raises ValueError: If given an empty value or an unsupported metric.
    :return: The computed distance between the two vectors.
    """
    if vect1 is None or vect2 is None:
        raise ValueError("Vector is None!")

    if normalize_flag:
        vect1 = normalize([vect1])[0]
        vect2 = normalize([vect2])[0]

    match metric:
        case 'cosine':
            return cosine_distances([vect1], [vect2])[0][0]
        case 'manhattan':
            return manhattan_distances([vect1], [vect2])[0][0]
        case 'euclidean':
            return float(np.linalg.norm(np.array(vect1) - np.array(vect2)))
        case 'hamming':
            return hamming(vect1, vect2)
        case 'minkowski':
            p_value = 3
            return minkowski(vect1, vect2, p=p_value)
        case 'dot_product':
            return np.dot(vect1, vect2)
        case 'l2-squared':
            return float(np.sum((np.array(vect1) - np.array(vect2)) ** 2))
        case _:
            raise ValueError(f"Invalid metric: {metric}")


def load_annotated_data(tsv_file_path: str | Path,
                        jsonl_vectors: dict[tuple[int, int], dict[str, list[float] | str]]) \
        -> list[AnnotatedWordPair]:
    """
    Load annotated data from a TSV file.

    :param tsv_file_path: Path to the TSV file containing annotations.
    :param jsonl_vectors: Dictionary with loaded vectors from the JSONL file.
    :raises ValueError: If no corresponding jsonl data is found for a row from the TSV file.
    :return: A list of AnnotatedWord instances.
    """
    annotated_data = []
    with open(tsv_file_path, 'r', encoding="utf-8") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        for idx, row in enumerate(reader):
            row_id = idx
            jsonl_1_data = jsonl_vectors.get((row_id, 1))
            jsonl_2_data = jsonl_vectors.get((row_id, 2))

            if not jsonl_1_data or not jsonl_2_data:
                raise ValueError(f"No jsonl data with row_id {row_id}.")

            definition1 = jsonl_1_data['definition'] if isinstance(
                jsonl_1_data['definition'], str) else None
            definition2 = jsonl_2_data['definition'] if isinstance(
                jsonl_2_data['definition'], str) else None
            vect1 = jsonl_1_data['vector'] if isinstance(
                jsonl_1_data['vector'], list) else None
            vect2 = jsonl_2_data['vector'] if isinstance(
                jsonl_2_data['vector'], list) else None

            annotated_data.append(AnnotatedWordPair(
                word=row['word'], sent1=row['sent1'], sent2=row['sent2'], mean=float(row['mean']),
                vect1=vect1, vect2=vect2,
                definition1=definition1, definition2=definition2)
            )
    return annotated_data


def load_vectorized_data(jsonl_file_path: str | Path) -> list[AnnotatedWordPair]:
    """
    Load vectorized data from a JSONL file and represent it as a list of AnnotatedWord instances.

    :param jsonl_file_path: The path to the JSONL file containing vectors.
    :return: A list of AnnotatedWord instances.
    """
    annotated_data = []
    with open(jsonl_file_path, 'r', encoding="utf-8") as jsonl_file:
        while True: # Read two sequential lines at a time
            line1 = jsonl_file.readline()
            if not line1.strip():
                break
            line2 = jsonl_file.readline()
            if not line2.strip():
                break

            data1 = json.loads(line1)
            data2 = json.loads(line2)

            annotated_data.append(AnnotatedWordPair(
                word=data1['word'], sent1=data1['sentence'], sent2=data2['sentence'],
                vect1=data1['vector'], vect2=data2['vector'])
            )

    return annotated_data


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


def save_as_json(prepared_data: list[dict[str, Any]], output_path: str | Path) -> None:
    """
    Save the prepared dataset as a JSON file.

    :param prepared_data: The dataset prepared for inference.
    :param output_path: The path where the JSON file will be saved.
    """
    parsed_output_path = parse_path(output_path)

    if not parsed_output_path.parent.exists():
        parsed_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(parsed_output_path, 'w', encoding='utf-8') as output_file:
        for record in prepared_data:
            json_record = json.dumps(record, ensure_ascii=False)
            output_file.write(json_record + '\n')
