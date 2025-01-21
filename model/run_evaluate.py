"""A module for evaluation of the model."""
import argparse
import json
from pathlib import Path

from src.metrics import get_bert_score, get_bleu_score, get_rouge_score
from src.utils import get_current_torch_device, parse_path


def load_target_pred_dataset(dataset_path: str | Path, target_field: str,
                             pred_field: str) -> tuple[list[str], list[str]]:
    """
    Load the target and predicted texts from a dataset.

    :param dataset_path: The path to the JSON Lines dataset.
    :param target_field: The name of the field containing the target texts.
    :param pred_field: The name of the field containing the predicted texts.
    :return: The target texts and the predicted texts.
    """
    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        dataset = [json.loads(line) for line in dataset_file if line.strip()]

    target_texts = [sample[target_field] for sample in dataset]
    pred_texts = [sample[pred_field] for sample in dataset]

    return target_texts, pred_texts


def evaluate_model_with_validation_dataset(dataset_path: str | Path, target_field: str,
                                           pred_field: str) -> dict[str, float]:
    """
    Evaluate the model on a dataset.

    :param dataset_path: The path to the JSON Lines dataset.
    :param target_field: The name of the field containing the target texts.
    :param pred_field: The name of the field containing the predicted texts.
    :return: The scores.
    """
    target_texts, pred_texts = load_target_pred_dataset(dataset_path, target_field, pred_field)

    bleu_score = get_bleu_score(target_texts, pred_texts)
    rouge_score = get_rouge_score(target_texts, pred_texts)
    bert_score = get_bert_score(target_texts, pred_texts, get_current_torch_device())

    return {
        "bleu": bleu_score,
        "rougeL": rouge_score,
        "bert-f1": bert_score
    }


def main() -> None:
    """Evaluate the model."""
    parser = argparse.ArgumentParser(
        description="Evaluate the model on a dataset.")

    parser.add_argument("dataset_path",
                        type=str,
                        help="The path to the dataset.")
    parser.add_argument("--target-field",
                        type=str,
                        default="target_text",
                        help="The name of the field containing the target texts.")
    parser.add_argument("--pred-field",
                        type=str,
                        default="generated_text",
                        help="The name of the field containing the predicted texts.")

    args = parser.parse_args()

    parsed_dataset_path = parse_path(args.dataset_path)

    scores = evaluate_model_with_validation_dataset(parsed_dataset_path,
                                                    args.target_field,
                                                    args.pred_field)

    print(scores)


if __name__ == "__main__":
    main()
