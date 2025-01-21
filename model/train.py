"""A script for training a T5 model."""
from pathlib import Path
from typing import Any

import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,  # pylint: disable=import-error
                          BatchEncoding, DataCollatorForSeq2Seq, PreTrainedTokenizer,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, TrainerControl,
                          TrainerState, TrainingArguments)

from plot import plot_graphs_based_on_log_history
from src.constants import ROOT_DIR
from src.dataset_processing import load_dataset_split
from src.metrics import get_bert_score, get_bleu_score, get_rouge_score
from src.utils import get_current_torch_device, load_train_config, parse_path


def preprocess_function(examples: dict, tokenizer: PreTrainedTokenizer) -> BatchEncoding:
    """
    Preprocess the examples for the model using the tokenizer.

    :param examples: The readable examples to preprocess.
    :param tokenizer: A tokenizer that was used to train the model.
    :return: The tokenized examples.
    """
    max_input_length = 1024
    max_target_length = 128

    model_inputs = tokenizer(examples["input_text"], max_length=max_input_length, truncation=True)

    targets = [text + tokenizer.eos_token for text in examples["target_text"]]

    labels = tokenizer(text_target=targets,
                       max_length=max_target_length,
                       truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred: tuple, tokenizer: PreTrainedTokenizer) -> dict[str, float]:
    """
    Compute the metrics for the evaluation.

    :param eval_pred: The tuple of predictions and labels.
    :param tokenizer: A tokenizer that was used to encode the dataset.
    :return: The metrics.
    """
    predictions, labels = eval_pred

    # Replace -100 as we can't decode them.
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    results = {
        "rougeL": get_rouge_score(decoded_predictions, decoded_labels),
        "blue": get_bleu_score(decoded_predictions, decoded_labels),
        "bert-f1": get_bert_score(decoded_predictions, decoded_labels, get_current_torch_device())
    }

    return results


def patch_lora(model: Any) -> None:
    """
    Patch the LoRa model to support gradient_checkpointing.

    :param model: The transformers model to be patched.
    """
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):  # pylint: disable=redefined-builtin
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


class LossLoggingCallback(TrainerCallback):  # pylint: disable=too-few-public-methods
    """A callback that draws plots for metrics."""

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        """Event called after an evaluation phase."""
        graphs_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        metrics = {"eval_rougeL", "eval_blue", "eval_bert-f1"}
        log_keys = set()

        for log_entry in state.log_history:
            log_keys.update(log_entry.keys())

        metrics = metrics.intersection(log_keys)

        plot_graphs_based_on_log_history(state.log_history,
                                         graphs_dir, list(metrics))


def main() -> None:
    """Train the model."""
    train_config = load_train_config(ROOT_DIR / "model" / "train_config.json")

    loaded_tokenizer = AutoTokenizer.from_pretrained(train_config.model_checkpoint)
    print(f"Using {type(loaded_tokenizer)}")

    dataset_dict = load_dataset_split(parse_path(train_config.dataset_split_directory),
                                      debug_mode=train_config.debug or False)

    tokenized_datasets = dataset_dict.map(
        lambda examples: preprocess_function(examples, loaded_tokenizer), batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(train_config.model_checkpoint)

    if train_config.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=train_config.r or 16,
            lora_alpha=train_config.lora_alpha or 64,
            lora_dropout=train_config.lora_dropout or 0.1
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        patch_lora(model)

    model.to(get_current_torch_device())

    model_name = train_config.model_checkpoint.rsplit('/', maxsplit=1)[-1]
    arguments = Seq2SeqTrainingArguments(
        ROOT_DIR / "models" / f"{model_name}-definition-modeling",
        learning_rate=train_config.learning_rate,
        lr_scheduler_type=train_config.lr_scheduler_type or "linear",
        warmup_steps=train_config.warmup_steps or 3000,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        gradient_checkpointing=train_config.gradient_checkpointing or False,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps or 1,
        weight_decay=train_config.weight_decay or 0.01,
        optim=train_config.optimizer or "adamw_torch",
        save_total_limit=train_config.save_total_limit or 3,
        evaluation_strategy=train_config.evaluation_strategy or "epoch",
        save_strategy=train_config.save_strategy or "epoch",
        logging_strategy=train_config.logging_strategy or "epoch",
        save_steps=train_config.save_steps or 500,
        logging_steps=train_config.logging_steps or 500,
        eval_steps=train_config.eval_steps or 500,
        num_train_epochs=train_config.num_train_epochs or 3,
        max_steps=train_config.max_steps or -1,
        predict_with_generate=train_config.predict_with_generate or False,
        generation_max_length=train_config.generation_max_length or 128,
        fp16=train_config.fp16 or False,
        bf16=train_config.bf16 or False,
        load_best_model_at_end=train_config.load_best_model_at_end or False,
        metric_for_best_model=train_config.metric_for_best_model or None,
        push_to_hub=train_config.push_to_hub or False
    )

    data_collator = DataCollatorForSeq2Seq(loaded_tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        arguments,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
        tokenizer=loaded_tokenizer,
        compute_metrics=(lambda eval_pred: compute_metrics(eval_pred, loaded_tokenizer))
        if train_config.predict_with_generate else None,
        callbacks=[LossLoggingCallback()]
    )

    trainer.train()


if __name__ == "__main__":
    main()
