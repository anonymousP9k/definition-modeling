#!/bin/bash

rusemshift_annotations="$1"
rusemshift_preds="$2"

python3 rushifteval/fine_tune_vectorizer.py --tsv "$rusemshift_annotations" \
--jsonl "$rusemshift_preds"
