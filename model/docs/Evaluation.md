# Evaluation

This is a module for evaluation of the model based on metrics such as BERT-F1, ROUGE-L, BLEU.

## How to run

1. Dataset

    You have to have a `.jsonl` dataset that has a target and a prediction field.
    Usually that would be the dataset created after running the `inference` module.

    You can also download existing predictions:

    > FRED-T5-1.7B-MAS-FN predictions
   > [Download](https://github.com/anonymousP9k/definition-modeling/issues/3)

    Unpack and put them under `model/data/output`.

2. Requirements

    Make sure you've installed the requirements and opened the Terminal or Command prompt
    in the root directory of this repository.
    If not, follow the instructions in [Model training](Model%20training.md).

3. Run

    For example:
    ```bash
    python model/run_evaluate.py model/data/output/output_test.jsonl
    ```

By default, the fields for input text and the generated one are called
`input_text` and `generated_text`.
You can set them manually using arguments `--target-field` and `--pred-field`.

The resulting output will be printed to the console.
