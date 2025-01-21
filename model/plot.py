"""A script for plotting functions."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils import parse_path


def plot_graphs_based_on_log_history(log_history: list[dict], output_dir: str | Path,
                                     metrics: list[str]) -> None:
    """
    Plot the graphs based on the log_history.

    :param log_history: The list of all logs from the Trainer.
    :param output_dir: The directory in which the plots will be created.
    :param metrics: The metrics which to create apart from training and test loss.
    """
    parsed_output_directory = parse_path(output_dir)

    plot_training_and_test_loss(log_history, parsed_output_directory / "loss-plot-epoch.png",
                                plot_epochs=True)
    plot_training_and_test_loss(log_history, parsed_output_directory / "loss-plot-step.png",
                                plot_epochs=False)

    for metric in metrics:
        plot_metric(metric, log_history,
                    parsed_output_directory / f"{metric}-plot-epoch.png", plot_epochs=True)
        plot_metric(metric, log_history,
                    parsed_output_directory / f"{metric}-plot-step.png", plot_epochs=False)


def plot_metric(metric: str, log_history: list[dict], output_path: str | Path,
                plot_epochs: bool = True) -> None:
    """
    Plot the metric using information from the log history.

    :param metric: The metric to plot (e.g. "rouge-1").
    :param log_history: The log history from the trainer.
    :param output_path: The path to save the plot to.
    :param plot_epochs: Whether to plot epochs or steps on the x-axis.
    """
    metric_values = []
    steps = []
    epochs = []

    for entry in log_history:
        if metric.strip() in entry:
            metric_values.append(entry[metric])
            steps.append(entry['step'])
            epochs.append(entry['epoch'])

    plt.figure(figsize=(10, 5))

    if plot_epochs:
        plt.plot(epochs, metric_values, label=metric, marker='o', linestyle='-', color='b')
        plt.title(f"{metric} Over Epochs")
        plt.xlabel('Epochs')
    else:
        plt.plot(steps, metric_values, label=metric, marker='o', linestyle='-', color='b')
        plt.title(f"{metric} Over Steps")
        plt.xlabel('Steps')

    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def plot_training_and_test_loss(log_history: list[dict], output_path: str | Path,
                                plot_epochs: bool = True) -> None:
    """
    Plot the training and test loss using information from the log history.

    :param log_history: The log history from the trainer.
    :param output_path: The path to save the plot to.
    :param plot_epochs: Whether to plot epochs or steps on the x-axis.
    :raises ValueError: If the train losses and test losses have different lengths.
    """
    train_losses = []
    test_losses = []
    steps = []
    epochs = []

    for entry in log_history:
        if 'loss' in entry:
            train_losses.append(entry['loss'])
            steps.append(entry['step'])
            epochs.append(entry['epoch'])
        if 'eval_loss' in entry:
            test_losses.append(entry['eval_loss'])

    if len(train_losses) != len(test_losses):
        print(f"Train losses: {train_losses}, test losses: {test_losses}, "
              f"steps: {steps}, epochs: {epochs}")
        raise ValueError("Train losses and test losses have different lengths")

    plt.figure(figsize=(10, 5))

    if plot_epochs:
        plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-',
                 color='b')
        plt.plot(epochs, test_losses, label='Test Loss', marker='o', linestyle='-', color='r')
        plt.title('Training and Test Loss Over Epochs')
        plt.xlabel('Epochs')
    else:
        plt.plot(steps, train_losses, label='Training Loss', marker='o', linestyle='-',
                    color='b')
        plt.plot(steps, test_losses, label='Test Loss', marker='o', linestyle='-', color='r')
        plt.title('Training and Test Loss Over Steps')
        plt.xlabel('Steps')

    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(parse_path(output_path))


def load_log_history_from_checkpoint(checkpoint_dir: str | Path) -> list[dict]:
    """
    Load the log history from the checkpoint directory.

    :param checkpoint_dir: The directory of the checkpoint.
    :return: The log history.
    """
    log_history = []
    with open(parse_path(checkpoint_dir) / "trainer_state.json", "r", encoding="utf-8") as file:
        trainer_state = json.load(file)
        for log in trainer_state['log_history']:
            log_history.append(log)

    return log_history


def main() -> None:
    """Plot the graphs."""
    parser = argparse.ArgumentParser(description="Plot the graphs based on the log history.")
    parser.add_argument("checkpoint_dir",
                        type=str,
                        help="The path to the checkpoint relative to the root directory.")

    parser.add_argument("--metrics",
                        "-m",
                        type=str,
                        nargs="+",
                        default=["eval_rougeL", "eval_blue", "eval_bert-f1"],
                        help="The metrics to plot. i.e. eval_rougeL, eval_bleu.")
    args = parser.parse_args()

    checkpoint_path = parse_path(args.checkpoint_dir)

    checkpoint_log_history = load_log_history_from_checkpoint(checkpoint_path)

    graphs_directory = checkpoint_path / "graphs"

    if not graphs_directory.exists():
        graphs_directory.mkdir(parents=True, exist_ok=True)

    plot_graphs_based_on_log_history(checkpoint_log_history, checkpoint_path / "graphs",
                                     args.metrics)

if __name__ == "__main__":
    main()
