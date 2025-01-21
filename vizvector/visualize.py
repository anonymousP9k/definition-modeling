"""Module to visualize shifts in meaning in the inferred and vectorized dataset."""
import argparse
import json
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pydantic.dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from transliterate import detect_language, translit

from vizvector_utils import parse_path

BASE_FONT_SIZE = 10
DECREMENT_STEP = 0.1


@dataclass
class WordUsage:
    """
    A dataclass representing a row in the JSONL dataset.

    :param word: The target word.
    :param sent: The sentence.
    :param definition: The generated definition for the word usage.
    :param vect: Embedding for the usage.
    """

    word: str
    sent: str
    date: Optional[str] = None
    definition: Optional[str] = None
    vect: Optional[list[float]] = None


def load_data(input_paths: list[str], target_word: str) -> list[WordUsage]:
    """
    Load word usages from the JSONL files for a given target word.

    :param input_paths: The paths to the input JSONL files.
    :param target_word: The target word to filter and cluster.
    :return: The loaded usages for the target word.
    """
    all_data = []

    for input_path in input_paths:
        data = []
        with open(input_path, 'r', encoding='utf-8') as file:
            for line in file:
                item = json.loads(line)
                if item['word'] == target_word:
                    definition = item['generated_text']
                    if not definition.endswith("."):
                        definition += "."

                    word_usage = WordUsage(
                        word=item['word'],
                        sent=item['sentence'],
                        date=item['date'] if 'date' in item else input_path,
                        definition=definition,
                        vect=item.get('vector', [])
                    )
                    data.append(word_usage)
        all_data.extend(data)

    return all_data


def count_vectors_per_cluster(unique_dates: list[str], clusters: np.ndarray,
                              date_mapping: list[int], relative: bool) -> dict:
    """
    Count the number of vectors assigned to each cluster including noise.

    :param unique_dates: A list with time periods.
    :param clusters: A 1D numpy array of cluster labels corresponding to each row in `data`.
    :param date_mapping: Mapping of usages to unique_dates.
    :param relative: Show the values in percentages relative to all usages in the epoch.
    :return: A dictionary with cluster labels as keys and counts as values.
    """
    date_clusters: dict[str, list[int]] = {date: [] for date in unique_dates}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id != -1:
            date_clusters[unique_dates[date_mapping[idx]]].append(cluster_id)

    date_cluster_counts = {}
    for date, cluster_list in date_clusters.items():
        unique, counts = np.unique(cluster_list, return_counts=True)
        counts_dict = dict(zip(unique, counts))

        if relative:
            total_counts = sum(counts)
            for cluster_id in counts_dict:
                counts_dict[cluster_id] = counts_dict[cluster_id] / total_counts * 100

        date_cluster_counts[date] = counts_dict

    return date_cluster_counts


def count_outliers(clusters: np.ndarray) -> tuple[int, float]:
    """
    Count the number of outliers in the clusters data.

    :param clusters: A 1D numpy array of cluster labels corresponding to each row in `data`.
    :return: A tuple containing the absolute and relative count.
    """
    count = 0

    for cluster in np.unique(clusters):
        if cluster == -1:
            cluster_points_idx = np.where(clusters == cluster)[0]
            count = len(cluster_points_idx)

    return count, count / len(clusters)


def find_representative_index(data: np.ndarray, clusters: np.ndarray,
                              method: str = 'centroid') -> dict:
    """
    Find the representative index for each cluster based on the specified method.

    :param data: A 2D numpy array where each row represents a vector.
    :param clusters: A 1D numpy array of cluster labels corresponding to each row in `data`.
    :param method: The method to use for finding the representative index ('centroid' or 'medoid').
    :return: A dictionary with cluster labels as keys and the index of the vector as values.
    :raises ValueError: If the method name is incorrect.
    """
    if method not in ['centroid', 'medoid']:
        raise ValueError("Method must be 'centroid' or 'medoid'")

    representative_idx = {}
    for cluster in np.unique(clusters):
        if cluster == -1:
            continue
        cluster_points_idx = np.where(clusters == cluster)[0]
        cluster_points = data[cluster_points_idx]

        if method == 'centroid':
            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            min_idx = np.argmin(distances)
            representative_idx[cluster] = cluster_points_idx[min_idx]
        elif method == 'medoid':
            pairwise_dist = pairwise_distances(cluster_points, metric="cosine")
            medoid_idx = np.argmin(np.sum(pairwise_dist, axis=0))
            representative_idx[cluster] = cluster_points_idx[medoid_idx]

    return representative_idx


def cluster_data_and_get_definitions(all_data: list[WordUsage],  # pylint: disable=too-many-arguments, too-many-locals
                                     eps: float = 0.5, relative: bool = True,
                                     min_samples: int = 5,
                                     do_normalize: bool = False,
                                     metric: str = "cosine") -> tuple[dict, dict]:
    """
    Cluster the data and get the definitions for the clusters.

    :param all_data: The data to cluster.
    :param eps: Maximum distance for considering neighborhood.
    :param relative: Show the values in percentages relative to all usages in the epoch.
    :param min_samples: Minimum samples for a core point.
    :param do_normalize: Normalize the vectors.
    :param metric: The metric to use for clustering.
    :return: The date-cluster counts and the definitions.
    """
    unique_dates = sorted(list(set(usage.date for usage in all_data
                                   if usage.vect is not None and usage.date is not None)))
    date_mapping = [unique_dates.index(usage.date) for usage in all_data
                    if usage.vect is not None and usage.date is not None]
    aggregated_vectors = np.array([usage.vect for usage in all_data if usage.vect is not None])

    if do_normalize:
        aggregated_vectors = normalize(aggregated_vectors, axis=1, norm='l2')

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    clusters = dbscan.fit_predict(aggregated_vectors)

    representative_indices = find_representative_index(aggregated_vectors, clusters,
                                                       method='centroid')

    definitions = {cluster: all_data[idx].definition
                   for cluster, idx in representative_indices.items()}

    date_cluster_counts = count_vectors_per_cluster(unique_dates, clusters, date_mapping,
                                                    relative=relative)

    outliers_count, outliers_percentage = count_outliers(clusters)

    print(f"Outliers count: {outliers_count}, outliers percentage: {outliers_percentage}")

    return date_cluster_counts, definitions


def get_definitions_subplot_text(definitions: dict) -> str:
    """
    Get the text for the subplot with definitions.

    :param definitions: The definitions.
    :return: The text for the subplot.
    """
    definitions_text = ""

    if len(definitions) > 15:
        print("Too many meanings to display. Showing in console.")
        print_definitions(definitions)
        return "Слишком много значений для отображения. Смотрите в консоли."

    for idx, (_, meaning) in enumerate(definitions.items()):
        wrapped_text = textwrap.fill(f"{idx + 1}: {meaning}", width=70)
        definitions_text += wrapped_text + "\n"
    return definitions_text


def print_definitions(definitions: dict) -> None:
    """
    Print the definitions.

    :param definitions: The definitions.
    """
    for idx, definition in definitions.items():
        print(f"{idx + 1}: {definition}")


def visualize_changes_with_legend(date_cluster_counts: dict,  # pylint: disable=too-many-locals, too-many-arguments
                                  definitions: dict,
                                  word: str,
                                  eps: float,
                                  min_samples: int,
                                  relative: bool = True,
                                  output_path: Path | str | None = None,
                                  minimal: bool = False) -> None:
    """
    Visualize semantic changes.

    :param date_cluster_counts: The counts of clusters for each date.
    :param definitions: The definitions for the clusters.
    :param word: The target word.
    :param eps: Maximum distance for considering neighborhood.
    :param min_samples: Minimum samples for a core point.
    :param relative: Show the values in percentages relative to all usages in the epoch.
    :param output_path: The path to save the plot.
    :param minimal: Do not show the definitions subplot and parameters.
    """
    dates = list(date_cluster_counts.keys())
    max_meaning_id = max(max(counts.keys()) for counts in date_cluster_counts.values()) + 1

    total_width = 0.8
    bar_width = total_width / len(dates)

    index = np.arange(max_meaning_id) * (total_width + 0.2)

    colors = plt.cm.Blues(  # type: ignore  # pylint: disable=no-member
        np.linspace(0.2, 1, len(dates)))

    if not minimal:
        fig, axs = plt.subplots(3, 1, figsize=(6.5, 7),
                                gridspec_kw={'height_ratios': [1, 8, 7], 'hspace': 0.05})
    else:
        fig, axs = plt.subplots(2, 1, figsize=(6.5, 4),
                                gridspec_kw={'height_ratios': [0.5, 8], 'hspace': 0.05})
        plt.subplots_adjust(left=0.125, right=0.9, top=0.95, bottom=0.15)
    ax_legend = axs[0]
    ax = axs[1]

    ax_legend.set_axis_off()
    legend_elements = [plt.Rectangle((0, 0), 1, 1,
                                     fc=colors[i], label=f'{date}') for i, date in
                       enumerate(dates)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=len(dates), frameon=False)

    for i, (date, counts) in enumerate(date_cluster_counts.items()):
        date_counts = [counts.get(meaning_id, 0) for meaning_id in range(max_meaning_id)]
        ax.bar(index + i * bar_width, date_counts, bar_width, color=colors[i])

    ax.set_xlabel('Значения')
    ax.set_ylabel('Процент от всех использований' if relative else 'Частота')
    ax.set_xticks(index + total_width / 2 - bar_width / 2)
    ax.set_xticklabels([f'{i + 1}' for i in range(max_meaning_id)])

    definitions_text = get_definitions_subplot_text(definitions)

    adjusted_font_size = max(BASE_FONT_SIZE * (1 - max(len(definitions) - 8, 0) * DECREMENT_STEP),
                             8)

    if not minimal:
        ax_definitions = axs[2]
        ax_definitions.text(0.0, 0.3, definitions_text, verticalalignment='center',
                            horizontalalignment='left', fontsize=adjusted_font_size, wrap=True)
        ax_definitions.set_axis_off()

        param_description = f"Параметры: eps={eps}, min_samples={min_samples}"
        fig.text(0.02, 0.02, param_description, fontsize=9, verticalalignment='bottom',
                 horizontalalignment='left')
    else:
        print(f"\nСлово: {word}")
        print_definitions(definitions)
        print(f"\nПараметры: eps={eps}, min_samples={min_samples}")

    if output_path:
        parsed_path = parse_path(output_path)
        if detect_language(parsed_path.name) == 'ru':
            transliterated_file_name = translit(parsed_path.name, "ru", True)
            parsed_path = parsed_path.parent / transliterated_file_name
        if not parsed_path.parent.exists():
            parsed_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(parsed_path, dpi=300)
    else:
        plt.show()


def main() -> None:
    """Visualize semantic changes."""
    parser = argparse.ArgumentParser(
        description="Visualize semantic change across multiple epochs.")
    parser.add_argument("input_paths", type=str, nargs='+',
                        help="Paths to the input JSONL files.")
    parser.add_argument("word", type=str,
                        help="Target word to filter and cluster.")
    parser.add_argument("--eps", type=float, default=0.5,
                        help="The maximum distance between two samples for one "
                             "to be considered as in the neighborhood of the other.")
    parser.add_argument("--relative", type=bool, default=True,
                        help="Show the values in percentages relative "
                             "to all usages in the epoch.")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="The number of samples in a neighborhood for a "
                             "point to be considered as a core point")
    parser.add_argument("--metric", type=str, default="cosine",
                        help="The metric to use for clustering.")
    parser.add_argument("--do-normalize", action='store_true',
                        help="Normalize the vectors.")
    parser.add_argument("--output-path", type=str, default=None,
                        help="The path to save the plot. "
                             "If not specified, the plot will be shown.")
    parser.add_argument("--minimal", action='store_true',
                        help="Do not show the definitions subplot and parameters.")
    args = parser.parse_args()

    all_data = load_data(args.input_paths, args.word)
    epoch_cluster_counts, definitions = cluster_data_and_get_definitions(all_data, eps=args.eps,
                                                                         relative=args.relative,
                                                                         min_samples=
                                                                         args.min_samples,
                                                                         do_normalize=
                                                                         args.do_normalize,
                                                                         metric=args.metric)
    visualize_changes_with_legend(epoch_cluster_counts, definitions, word=args.word,
                                  relative=args.relative, eps=args.eps,
                                  min_samples=args.min_samples,
                                  output_path=args.output_path,
                                  minimal=args.minimal)


if __name__ == "__main__":
    main()
