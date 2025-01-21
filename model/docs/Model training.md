# Model training

This is a module for training the FRED-T5-1.7B model.

## How to run

1. Open the Terminal or Command Prompt in the root directory of this repository.

    ```bash
    cd path/to/work-definition-modeling
    ```

2. Activate the virtual environment.

    For Windows:

    ```bash
    python3 -m venv venv
    .\venv\Scripts\activate
    ```

    For Linux and macOS:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Make sure you've installed the requirements.

    You can do it by running

    ```bash
    python -m pip install -r requirements.txt
    python -m pip install -r requirements_train.txt
    ```

    You also have to install the `torch` library that will be optimal for your machine.
    For example, if you have a CUDA enabled GPU on Windows, you would probably want to run this:

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

    Check the [official PyTorch website](https://pytorch.org/get-started/locally/)
    for the exact command for your machine.

4. Make sure you have the dataset.

    It is a `.jsonl` file that has `id`, `title` and `definition` fields.
    The file is produced by running `clean_mas_dataset.py` from the `mas_parser` module.

    If you do not have the file, follow the instructions
    in [mas_parser](../../mas_parser/README.md)

    You can also download existing dataset:

    > MAS definitions
   > [Download](https://github.com/anonymousP9k/definition-modeling/issues/1)

    Unpack and put it under `mas_parser/data`.

5. Split the dataset into train, validation and test sets.

    In order to do that, run the following command:

    ```bash
    python model/split_dataset.py path/to/definitions.jsonl -o output/path
    ```

    For example:

    ```bash
    python model/split_dataset.py mas_parser/data/mas_cleaned_definitions.jsonl \
   -o model/data/splits
    ```

    This will create three files in the `data` folder:
    `train.jsonl`, `val.jsonl` and `test.jsonl`.

    > NOTE: Run the example command above to reproduce results.

    You can also download existing splits:

    > MAS splits
   > [Download](https://github.com/anonymousP9k/definition-modeling/issues/2)

    Unpack and put the files under `model/data/splits`.

6. Configure the training parameters in `train_config.json`.

    They mostly follow the parameters in the Seq2Seq
    [Trainer](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments)
    class of the `transformers` library.

    The exception are `debug`.
    If set to `True`, the training will run on a small subset of the data.
    Parameter `dataset_split_directory` is the directory where the dataset split is stored.
    
    You can also enable training with LoRa.
    Set `use_lora` to `True` and set other parameters which follow the
    [LoraConfig](https://huggingface.co/docs/peft/package_reference/lora) class of the
    `peft` library.

    > NOTE: The final training was done with LoRa. Enable it if you want to replicate results.
    > NOTE: Keep default values in the config file to reproduce the results.

7. Run `train.py` with the following command to start training.

    ```bash
    python model/train.py
    ```

    The checkpoints will be saved in the `models` folder.

## Plots

The plots are generated automatically based on log history from trainer state
when each checkpoint and are saved under `path/to/checkpoint/graphs` directory.

> NOTE: By enabling `predict_with_generate` the plots for metrics like `bleu`, `rouge`
> and `bertscore` will be generated during the training. It may be time-consuming!

But you can also regenerate them manually by running `plot.py` with the following command.

```bash
python model/plot.py path/to/checkpoint
```

You can also specify the metrics you want to plot by passing them as arguments.

```bash
python model/plot.py path/to/checkpoint --metrics eval_rougeL
```
