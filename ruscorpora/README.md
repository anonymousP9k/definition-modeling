# Ruscorpora

This is a module for qualitative testing of the project.

## How to run

### 1. Create samples for the ruscorpora module

> NOTE: You can skip this step by downloading the sample and putting it under `ruscorpora/data`:
> [Download](https://github.com/anonymousP9k/definition-modeling/issues/8)

#### 1.1 Request diachronic ruscorpora corpus

Request the corpus from Ruscorpora.
You can read more about it [here](https://ruscorpora.ru/page/corpora-datasets/).

The resulting corpus are three files:

- `rnc_post-soviet.txt.gz`
- `rnc_pre-soviet.txt.gz`
- `rnc_soviet.txt.gz`

Put them under `ruscorpora/data`.

#### 1.2 Process the corpus

Run the `process_ruscorpora_dump.py` script to process the corpus:

```bash
python ruscorpora/process_ruscorpora_dump.py
```

The settings for the script are in `process_ruscorpora_config.json`.

> NOTE: Run the command above to reproduce results.

With default settings, the script will create a `results.jsonl` file under `ruscorpora/tmp`
with word usages from the corpus.

#### 1.3 Sample the corpus

Run the `sample_ruscorpora.py` script to sample the corpus:

```bash
python ruscorpora/sample_ruscorpora.py ruscorpora/tmp/results.jsonl \
ruscorpora/data/ruscorpora_sample_300_seed_42.jsonl --sample_size 300 --seed 42
```

It will randomly sample 300 word usages for each word-epoch pair and save them to a file.

### 2. Get predictions for the sample

> NOTE: You can skip this step by downloading the predictions and putting them
> under `ruscorpora/data`:
> [Download](https://github.com/anonymousP9k/definition-modeling/issues/9)

We would use `inference.py` script from the `model` module to get predictions for the sample.

Run the `inference.py` script to get the predictions:

```bash
python model/inference.py ai-forever/FRED-T5-1.7B -l models/checkpoint-41000
--input-file ruscorpora/data/ruscorpora_sample_300_seed_42.jsonl --output-file \
ruscorpora/data/preds/ruscorpora_sample_300_seed_42_preds.jsonl
```

Add `--batch_size` argument with an integer value to make inference faster
if it is available for you.

### 3. Vectorize the predictions

> NOTE: You can skip this step by downloading the vectors and putting them
> under `ruscorpora/data/vectors`:
> [Download](https://drive.google.com/file/d/1E4bzgejBOJGmM_JNChl7QQkvpULUndtl/view?usp=sharing)

You are supposed to have a fine-tuned vectorizer model to vectorize the predictions by now.
If not, download the fine-tuned vectorizer model and put it under `models`:
[Download](https://github.com/anonymousP9k/definition-modeling/issues/6)

Run the `vectorize.py` from the `vizvector` module to vectorize the predictions:

```bash
python vizvector/vectorize.py ruscorpora/data/preds/ruscorpora_sample_300_seed_42_preds.jsonl \
ruscorpora/data/vectors/ruscorpora_sample_300_seed_42_vectors.jsonl \
--model-name models/paraphrase-multilingual-mpnet-base-dm
```

### 4. Visualize the predictions

You can visualize the predictions using `visualize.py` script from the `vizvector` module:

```bash
python vizvector/visualize.py ruscorpora/data/vectors/ruscorpora_sample_300_seed_42_vectors.jsonl \
пакет --eps 0.49 --min-samples 15
```

To get visualizations for all words, run the bash script:

```bash
bash ruscorpora/bash/visualize_all.sh
```

The visualizations will be located under `ruscorpora/tmp/visualizations`.
