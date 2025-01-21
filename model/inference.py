"""A module for the inference of the model."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pymorphy3
import torch  # pylint: disable=import-error
from peft import PeftModel
from torch import dtype
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration  # pylint: disable=import-error

from src.utils import get_current_torch_device, parse_path


def generate_forms(word: str) -> list[str]:
    """
    Generate all forms of a word.

    :param word: The word.
    :return: The forms.
    """
    morph = pymorphy3.MorphAnalyzer()
    parsed_word = morph.parse(word)[0]
    forms = set()

    for form in parsed_word.lexeme:
        forms.add(form.word)

    return list(forms)


def get_bad_words_ids(word: str, tokenizer: AutoTokenizer) -> list[int]:
    """
    Get the token IDs of the bad words in the vocabulary.

    :param word: The word.
    :param tokenizer: The tokenizer.
    :return: The token IDs of the bad words.
    """
    forms = generate_forms(word)
    bad_words_ids = []

    for form in forms:
        bad_words_ids.append(tokenizer.encode(form, add_special_tokens=False))
        bad_words_ids.append(tokenizer.encode(f" {form}", add_special_tokens=False))

    return bad_words_ids


def load_model(model_checkpoint: str | Path, torch_dtype: dtype = torch.float32,
               lora_checkpoint: str | Path = "") \
        -> tuple[T5ForConditionalGeneration, AutoTokenizer]:
    """
    Load a model from a checkpoint.

    :param model_checkpoint: The path to the checkpoint.
    :param torch_dtype: The precision in which to load the model.
    :param lora_checkpoint: The path to the LoRa checkpoint.
    :return: The model and the tokenizer.
    """
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if lora_checkpoint:
        model = PeftModel.from_pretrained(model, lora_checkpoint)
    model.to(get_current_torch_device())

    return model, tokenizer


def run_inference(  # pylint: disable=too-many-arguments
        model: T5ForConditionalGeneration, tokenizer: AutoTokenizer,
        input_texts: list[str], max_length: int = 128,
        avoid_target_word: bool = False, target_word: str = "") -> list[str]:
    """
    Run batched inference.

    :param model: The model.
    :param tokenizer: The tokenizer.
    :param input_texts: The input texts.
    :param max_length: The maximum length of the output.
    :param avoid_target_word: Whether to avoid the target word in the output.
    :param target_word: The target word to avoid.
    :return: The generated texts.
    """
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(model.device)

    output_sequences = model.generate(
        input_ids=inputs.input_ids,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        max_length=max_length,
        num_beams=3,
        bad_words_ids=get_bad_words_ids(target_word, tokenizer) if avoid_target_word else None
    ).cpu()

    outputs = np.where(output_sequences != -100, output_sequences, tokenizer.pad_token_id)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def run_inference_over_dataset(  # pylint: disable=too-many-arguments, too-many-locals
        model: T5ForConditionalGeneration,
        tokenizer: AutoTokenizer,
        data: list[dict[str, str]],
        input_field: str,
        output_file_path: str | Path,
        max_length: int = 100,
        batch_size: int = 1,
        avoid_target_word: bool = True) -> None:
    """
    Run batched inference over a dataset.

    :param model: The model.
    :param tokenizer: The tokenizer.
    :param data: The dataset.
    :param input_field: The name of the field in the dataset that contains the input text.
    :param output_file_path: The path to the output file.
    :param max_length: The maximum length of the output.
    :param batch_size: The size of the batch for batched inference.
    :param avoid_target_word: Whether to avoid the target word in the output.
    :raises ValueError: If avoid_target_word is True and some entries don't have a target word.
    """
    total_entries = len(data)

    for i in tqdm(range(0, total_entries, batch_size)):
        batch = data[i:i + batch_size]
        input_texts = [entry[input_field] for entry in batch]

        if avoid_target_word:
            target_words = [entry["word"] for entry in batch if "word" in entry]

            if len(target_words) != len(batch):
                raise ValueError("If avoid_target_word is True,"
                                 "all entries must have a target word.")

            target_words = list(set(target_words))

            for target_word in target_words:
                target_word_batch = [entry for entry in batch if entry["word"] == target_word]
                target_word_input_texts = [entry[input_field] for entry in target_word_batch]

                output_texts = run_inference(model, tokenizer, target_word_input_texts,
                                             max_length=max_length, avoid_target_word=True,
                                             target_word=target_word)

                for j, output_text in enumerate(output_texts):
                    save_output(output_text, output_file_path, target_word_batch[j])
        else:
            output_texts = run_inference(model, tokenizer, input_texts, max_length=max_length)

            for j, output_text in enumerate(output_texts):
                save_output(output_text, output_file_path, batch[j])


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate the arguments.

    Either both input_file and output_file are specified, or none.
    The model_checkpoint must always be specified.
    :param args: The arguments.
    :raises ValueError: If the arguments are invalid.
    """
    if args.input_file and not args.output_file:
        raise ValueError("If input_file is specified, output_file must be specified too.")
    if not args.input_file and args.output_file:
        raise ValueError("If output_file is specified, input_file must be specified too.")
    if not args.model_checkpoint:
        raise ValueError("model_checkpoint must be specified.")


def save_output(output_text: str, output_save_filepath: str | Path, initial_json: dict) -> None:
    """
    Append generated text to the output file.

    :param output_text: The generated text.
    :param output_save_filepath: The output file of JSON Lines format.
    :param initial_json: The initial JSON object.
    """
    output_save_filepath = parse_path(output_save_filepath)

    with open(output_save_filepath, "a", encoding="utf-8") as output_save_file:
        initial_json["generated_text"] = output_text
        output_save_file.write(json.dumps(initial_json, ensure_ascii=False) + "\n")


def load_dataset_for_inference(input_file_path: str | Path, input_field: str,
                               limit: int | None = None) -> list[dict[str, str]]:
    """
    Load the dataset for inference.

    :param input_file_path: The path to the input file.
    :param input_field: The name of the field in the dataset that contains the input text.
    :param limit: How many rows to load. Will load all if not passed.
    :return: The dataset.
    :raises ValueError: If the input field is not found in the JSON object.
    """
    data = []

    with open(input_file_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            json_object = json.loads(line)
            if input_field not in json_object:
                raise ValueError(f"Field {input_field} not found in JSON object.")
            data.append(json_object)

            if limit and len(data) == limit:
                break

    return data


def validate_output_file(output_file_path: str | Path) -> None:
    """
    Validate the output file. If it exists, ask the user whether to remove it.

    :param output_file_path: The path to the output file.
    """
    output_save_filepath = parse_path(output_file_path)

    if output_save_filepath.exists():
        remove_output = input("Output file already exists. Remove? (y/n) ")
        if remove_output.lower() == "y":
            output_save_filepath.unlink()
        else:
            print("Exiting...")
            sys.exit(0)

    if not output_save_filepath.parent.exists():
        output_save_filepath.parent.mkdir(parents=True)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Process prompts from the command line or a JSON lines file.")

    parser.add_argument("model_checkpoint",
                        type=str,
                        help="The path to the model checkpoint.")
    parser.add_argument("-l", "--lora-checkpoint",
                        type=str,
                        help="The path to the LoRa checkpoint.")
    parser.add_argument("-i", "--input-file",
                        type=str,
                        help="The input file of JSON Lines format.")
    parser.add_argument("-o", "--output-file",
                        type=str,
                        help="The output file of JSON Lines format.")
    parser.add_argument("-b", "--batch-size",
                        type=int,
                        default=1,
                        help="The inference batch size.")
    parser.add_argument("--input-field",
                        type=str,
                        default="input_text",
                        help="The name of the field in the dataset that contains the input text.")
    parser.add_argument("--precision",
                        type=str,
                        default="float32",
                        help="The precision in which to load the model.")
    parser.add_argument("--limit",
                        type=int,
                        help="How many rows to load.")
    parser.add_argument("--avoid-target-word",
                        type=bool,
                        default=True,
                        help="Avoid the target word in the output for dataset inference.")

    args = parser.parse_args()

    validate_args(args)

    torch_dtype = torch.float32

    if args.precision == "float16":
        torch_dtype = torch.float16
    elif args.precision == "bfloat16":
        torch_dtype = torch.bfloat16

    model, tokenizer = load_model(args.model_checkpoint,
                                  torch_dtype=torch_dtype,
                                  lora_checkpoint=args.lora_checkpoint)

    if args.input_file is not None:
        input_file_path = parse_path(args.input_file)
        output_file_path = parse_path(args.output_file)

        validate_output_file(output_file_path)

        data = load_dataset_for_inference(input_file_path, args.input_field, args.limit)

        run_inference_over_dataset(model, tokenizer, data, args.input_field,
                                   output_file_path, batch_size=args.batch_size,
                                   avoid_target_word=args.avoid_target_word)
    else:
        while True:
            prompt = input("Enter a prompt: ")

            if not prompt:
                print("Exiting...")
                sys.exit(0)

            output_texts = run_inference(model, tokenizer, [prompt])
            print(output_texts[0], end="\n\n")


if __name__ == "__main__":
    main()
