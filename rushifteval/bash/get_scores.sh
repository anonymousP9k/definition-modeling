#!/bin/bash

vectors_folder="$1"
metric="$2"

normalize=""
if [ $# -eq 3 ] && [ "$3" == "--normalize" ]; then
    normalize="--normalize"
fi

for i in {1..3}
do
    python rushifteval/calculate_ratings.py \
    --jsonl "rushifteval/tmp/vectors/${vectors_folder}/vectors_rushifteval${i}_test.jsonl" \
    --output "rushifteval/tmp/ratings/rushifteval${i}_test.tsv" \
    --metric "$metric" $normalize
done

python rushifteval/calculate_score.py \
rushifteval/tmp/ratings/rushifteval1_test.tsv \
rushifteval/tmp/ratings/rushifteval2_test.tsv \
rushifteval/tmp/ratings/rushifteval3_test.tsv \
rushifteval/tmp/result/result_testset.tsv

python rushifteval/check_score.py
