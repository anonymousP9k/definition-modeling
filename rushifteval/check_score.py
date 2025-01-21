"""Module that checks how good is the correlation."""
import argparse
from pathlib import Path

import pandas as pd
from pydantic.dataclasses import dataclass

from rushifteval_utils import parse_path


@dataclass
class Scores:
    """A class to hold test score results with correlation values."""

    mean_correlation: float
    correlations: dict

    def __str__(self) -> str:
        """Return a string representation of the Scores object."""
        correlations_str = '\n'.join(f'{k}: {v}' for k, v in self.correlations.items())
        return (f"Mean correlation: {self.mean_correlation}\n"
                f"Correlations for all epoch pairs:\n{correlations_str}")


def calculate_test_score(predictions_path: str | Path, gold_scores_path: str | Path) -> Scores:
    """
    Calculate the Spearman correlation between predicted scores and gold standard scores.

    :param predictions_path: The file path to the predictions TSV file.
    :param gold_scores_path: The file path to the gold standard TSV file.
    :returns: A Scores object containing correlations.
    """
    predictions_path = parse_path(predictions_path)
    gold_scores_path = parse_path(gold_scores_path)

    preds_df = pd.read_csv(predictions_path, sep='\t', header=None)
    gold_df = pd.read_csv(gold_scores_path, sep='\t', header=None)

    preds_df = preds_df.drop(columns=[0])
    gold_df = gold_df.drop(columns=[0])

    correlations = preds_df.corrwith(gold_df, method='spearman')
    correlations_dict = dict(
        zip(['pre-Soviet:Soviet', 'Soviet:post-Soviet', 'pre-Soviet:post-Soviet'], correlations))
    mean_correlation = correlations.mean()

    return Scores(mean_correlation=mean_correlation, correlations=correlations_dict)


def main() -> None:
    """Handle command line arguments and invoke the score calculation."""
    parser = argparse.ArgumentParser(description='Calculate test scores correlation')
    parser.add_argument('--predictions-filepath', type=str,
                        default="rushifteval/tmp/result/result_testset.tsv",
                        help='File path to the predictions tsv file.')
    parser.add_argument('--gold-scores-filepath', type=str,
                        default="rushifteval/data/gold/annotated_testset.tsv",
                        help='File path to the gold standards tsv file.')
    args = parser.parse_args()

    scores = calculate_test_score(
        predictions_path=args.predictions_filepath,
        gold_scores_path=args.gold_scores_filepath
    )

    print(scores)


if __name__ == '__main__':
    main()
