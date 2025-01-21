#!/bin/bash

vectors_folder="$1"
metric="$2"

normalize=""
if [ $# -eq 3 ] && [ "$3" == "--normalize" ]; then
    normalize="--normalize"
fi

python3 rushifteval/train_similarity_model.py \
--tsv rushifteval/data/rusemshift/rusemshift_all_raw_annotations.tsv \
--jsonl "rushifteval/tmp/vectors/${vectors_folder}/vectors_rusemshift_all.jsonl" \
--metric "$metric" $normalize

model_postfix="${metric}"

if [ "$normalize" == "--normalize" ]; then
    model_postfix="${metric}_normalize"
fi

for i in {1..3}
do
    python3 rushifteval/calculate_ratings.py \
    --tsv "rushifteval/data/rushifteval/rushifteval${i}_test_raw_annotations.tsv" \
    --jsonl "rushifteval/tmp/vectors/${vectors_folder}/vectors_rushifteval${i}_test.jsonl" \
    --model "rushifteval/tmp/models/similarity_model_${model_postfix}.joblib" \
    --output "rushifteval/tmp/ratings/rushifteval${i}_test.tsv" \
    --metric "$metric" $normalize
done

python3 rushifteval/calculate_score.py \
rushifteval/tmp/ratings/rushifteval1_test.tsv \
rushifteval/tmp/ratings/rushifteval2_test.tsv \
rushifteval/tmp/ratings/rushifteval3_test.tsv \
rushifteval/tmp/result/result_testset.tsv

python3 rushifteval/check_score.py
