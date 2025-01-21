# Rushifteval module

This is a module for checking the quality of using the fine-tuned T5 model for
detecting semantic change.

## How to run

### 1. Download `data`

Download the `data` archive, unpack it and put the contents under `rushifteval` directory:
[Download](https://github.com/anonymousP9k/definition-modeling/issues/7)

You'll have a `rushifteval/data` folder with `gold`, `rushifteval`, and `rusemshift` folders
inside.

The meaning of the folders:

- `gold` contains the gold correlation coefficients against which we will check the quality later.
- `rushifteval` contains the dataset for which we will calculate the coefficient.
- `rusemshift` is a training dataset, e.g. we will use it train models converting distances
between vectors to the dataset format.

### 2. Process annotations and get definitions

> NOTE: Inference takes time.
> You can skip this step by downloading the archive and
> unpacking it under `rushifteval/data` directory:
> [Download](https://github.com/anonymousP9k/definition-modeling/issues/5)
>
> You should get `rushifteval/data/preds/FRED-T5-1.7B-MAS-FN_preds` directory
> with 4 `.jsonl` files.

#### 2.1 Process annotations

Since we need to get definitions for each sentence pair, we need to run inference on the dataset.
But before that we must convert the `.tsv` files to `.jsonl` files that `inference.py` can use.

`process_raw_annotations.py` can be run like this:

```bash
python rushifteval/process_raw_annotations.py path_to_the_raw_dataset path_to_the_new_dataset
```

The `path_to_the_raw_dataset` can be either a `.tsv` file or a directory containing such files.

Run `process_raw_annotations.sh` to run commands needed for `rushifteval` and `rusemeval`:

```bash
bash rushifteval/bash/process_raw_annotations.sh
```

> NOTE: Run the bash-command above to reproduce results.

#### 2.2 Get definitions

Now we have the `.jsonl` files that `inference.py` can use.

> NOTE: You can read about inference in the [Inference](../../model/docs/Inference.md) module.

Run the `get_definitions.sh` script to get the definitions.

You have to supply the path to the model like this:

```bash
bash rushifteval/bash/get_definitions.sh path_to_the_model
```

If you have a LoRa adapter, supply the base model name and the path to the adapter, e.g.:

```bash
bash rushifteval/bash/get_definitions.sh ai-forever/FRED-T5-1.7B \
models/FRED-T5-1.7B-MAS-FN
```

> NOTE: Run the command above (the one with LoRa) to reproduce results.

The optional argument is `batch_size`. Use it like this to make inference faster.

```bash
bash rushifteval/bash/get_definitions.sh ai-forever/FRED-T5-1.7B \
models/FRED-T5-1.7B-MAS-FN --batch_size 16
```

A folder will appear under `rushifteval/data/preds` with the definitions.

#### PS: You can download existing LoRa adapters

> FRED-T5-1.7B-MAS-FN (LoRa)
> [Download](https://github.com/anonymousP9k/definition-modeling/issues/4)

### 3. Fine-tune the vectorizer model

> NOTE: You can skip this step by downloading the fine-tuned vectorizer model:
> [Download](https://github.com/anonymousP9k/definition-modeling/issues/6)
>
> Put it under `models` folder.

To get the highest results, you need to fine-tune the vectorizer model on the `rusemshift` dataset.

The best base model for this task is `paraphrase-multilingual-mpnet-base-v2`.

Run the `fine_tune_vectorizer.sh` script to fine-tune the model:

```bash
bash rushifteval/bash/fine_tune_vectorizer.sh path_to_the_annotations path_to_the_predictions
```

For example:

```bash
bash rushifteval/bash/fine_tune_vectorizer.sh \
rushifteval/data/rusemshift/rusemshift_all_raw_annotations.tsv \
rushifteval/data/preds/FRED-T5-1.7B-MAS-FN_preds/preds_rusemshift_all.jsonl
```

> NOTE: Run the example command above to reproduce results.

### 4. Vectorize the definitions

Establish the name of the folder inside `rushifteval/data/preds` that contains the definitions
and which you want to vectorize.

Pass it together with the sentence-transformer model to the `rushifteval/bash/vectorize.sh` file.

For example:

```bash
bash rushifteval/bash/vectorize.sh FRED-T5-1.7B-MAS-FN_preds \
models/paraphrase-multilingual-mpnet-base-dm
```

> NOTE: Run the command above to reproduce results.

### 5. Calculate the scores

Run `rushifteval/bash/get_scores.sh`:

For example:

```bash
bash rushifteval/bash/get_scores.sh FRED-T5-1.7B-MAS-FN_preds_paraphrase-multilingual-mpnet-base-dm \
cosine
```

where the first argument is the name of the folder with vectors, the second
argument is the metric and the third optional argument is whether to
normalize the vectors.

> NOTE: Run the command above to reproduce results.

The script will report the results in the following format:

```text
Mean correlation: 0.8002271148267935
Correlations for all epoch pairs:
pre-Soviet:Soviet: 0.7843978826860809
Soviet:post-Soviet: 0.8139468755069555
pre-Soviet:post-Soviet: 0.8023365862873442
```
