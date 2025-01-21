"""A script for supporting functions."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .constants import ROOT_DIR


@dataclass
class TrainConfigDTO:  # pylint: disable=too-many-instance-attributes
    """A data class for representing the train config."""

    model_checkpoint: str
    dataset_split_directory: str
    learning_rate: float
    batch_size: int
    use_lora: Optional[bool] = False
    r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lr_scheduler_type: Optional[str] = None
    warmup_steps: Optional[int] = None
    gradient_checkpointing: Optional[bool] = None
    gradient_accumulation_steps: Optional[int] = None
    weight_decay: Optional[float] = None
    optimizer: Optional[str] = None
    max_steps: Optional[int] = None
    num_train_epochs: Optional[int] = None
    save_total_limit: Optional[int] = None
    predict_with_generate: Optional[bool] = None
    generation_max_length: Optional[bool] = None
    save_steps: Optional[int] = None
    save_strategy: Optional[str] = None
    logging_steps: Optional[int] = None
    logging_strategy: Optional[str] = None
    eval_steps: Optional[int] = None
    evaluation_strategy: Optional[str] = None
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    load_best_model_at_end: Optional[bool] = None
    metric_for_best_model: Optional[str] = None
    push_to_hub: Optional[bool] = None
    debug: Optional[bool] = None

    def __post_init__(self):
        if self.max_steps is None and self.num_train_epochs is None:
            raise ValueError("TrainConfigDTO must contain either 'max_steps' or "
                             "'num_train_epochs'")

        if self.max_steps is not None and self.num_train_epochs is not None:
            raise ValueError("TrainConfigDTO must not contain both 'max_steps' and "
                             "'num_train_epochs'")

        if self.save_steps is not None and self.save_strategy != "steps":
            raise ValueError("TrainConfigDTO must not contain 'save_steps' and "
                             "'save_strategy' other than 'steps'")

        if self.logging_steps is not None and self.logging_strategy != "steps":
            raise ValueError("TrainConfigDTO must not contain 'logging_steps' and "
                             "'logging_strategy' other than 'steps'")

        if self.eval_steps is not None and self.evaluation_strategy != "steps":
            raise ValueError("TrainConfigDTO must not contain 'eval_steps' and "
                             "'evaluation_strategy' other than 'steps'")

        if (not self.predict_with_generate and
                (self.load_best_model_at_end or self.metric_for_best_model)):
            raise ValueError("predict_with_generate is disabled, but you still set the metrics")


def load_train_config(config_path: str | Path) -> TrainConfigDTO:
    """
    Load a config from a JSON file.

    :param config_path: The path to the config.
    :return: The config.
    """
    with open(config_path, "r", encoding="utf-8") as json_file:
        config = json.load(json_file)

    return TrainConfigDTO(**config)


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


def get_current_torch_device() -> str:
    """
    Get the current torch device.

    :return: The current torch device.
    """
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"
