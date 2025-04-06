#!/bin/bash

predictions_folder="$1"
vectorizer="$2"

vectorizer_name=$(basename "$vectorizer")

vectors_folder="${predictions_folder}_${vectorizer_name}"

python vizvector/vectorize.py "rushifteval/data/preds/${predictions_folder}/preds_rusemshift_all.jsonl" \
"rushifteval/tmp/vectors/${vectors_folder}/vectors_rusemshift_all.jsonl" --model-name "${vectorizer}"

python vizvector/vectorize.py "rushifteval/data/preds/${predictions_folder}/preds_rushifteval1_test.jsonl" \
"rushifteval/tmp/vectors/${vectors_folder}/vectors_rushifteval1_test.jsonl" --model-name "${vectorizer}"

python vizvector/vectorize.py "rushifteval/data/preds/${predictions_folder}/preds_rushifteval2_test.jsonl" \
"rushifteval/tmp/vectors/${vectors_folder}/vectors_rushifteval2_test.jsonl" --model-name "${vectorizer}"

python vizvector/vectorize.py "rushifteval/data/preds/${predictions_folder}/preds_rushifteval3_test.jsonl" \
"rushifteval/tmp/vectors/${vectors_folder}/vectors_rushifteval3_test.jsonl" --model-name "${vectorizer}"
