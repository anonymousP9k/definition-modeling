"""A script that provides metrics, such as ROUGE or BLEU."""
from evaluate import load
from razdel import tokenize as regex_tokenize


def get_bleu_score(predictions: list[str], labels: list[str]) -> float:
    """
    Compute the BLEU score for a list of predictions and labels.

    :param predictions: The list of predictions.
    :param labels: The list of labels.
    :return: The BLEU score.
    """
    blue_metric = load("sacrebleu")

    blue_results = blue_metric.compute(predictions=predictions,
                                       references=labels,
                                       tokenize="intl")
    return blue_results["score"]


def get_rouge_score(predictions: list[str], labels: list[str]) -> float:
    """
    Compute the ROUGE score for a list of predictions and labels.

    :param predictions: The list of predictions.
    :param labels: The list of labels.
    :return: The score for ROUGE-L.
    """
    rouge_metric = load("rouge")

    rouge_results = rouge_metric.compute(predictions=predictions,
                                         references=labels,
                                         use_stemmer=False,
                                         tokenizer=lambda x: [
                                             token.text.lower() for token in regex_tokenize(x)
                                         ])

    return rouge_results["rougeL"]


def get_bert_score(predictions: list[str], labels: list[str],
                   device: str = "cpu") -> float:
    """
    Compute the BERT score for a list of predictions and labels.

    :param predictions: The list of predictions.
    :param labels: The list of labels.
    :param device: The PyTorch device to use (e.g. "cuda", "mps" or "cpu")
    :return: The score F1.
    """
    bert_metric = load("bertscore")

    bert_results = bert_metric.compute(predictions=predictions,
                                       references=labels,
                                       lang="ru",
                                       device=device)

    return sum(bert_results["f1"]) / len(bert_results["f1"])
