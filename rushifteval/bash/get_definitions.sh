#!/bin/bash

lora_file_path=""
lora_file_path_arg=""
batch_size=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --batch_size)
            batch_size="--batch_size $2"
            shift
            shift
            ;;
        *)
            if [ -z "$base_model_name_or_path" ]; then
                base_model_name_or_path="$1"
            elif [ -z "$lora_file_path_arg" ]; then
                lora_file_path="$1"
                lora_file_path_arg="-l $1"  # NOTE: Includes -l !!!
            fi
            shift
            ;;
    esac
done

if [ -z "$lora_file_path" ]; then
    model_name=$(basename "$base_model_name_or_path")
else
    model_name=$(basename "$lora_file_path")
fi

python model/inference.py "$base_model_name_or_path" $lora_file_path_arg $batch_size \
    --input-file rushifteval/tmp/for_inference/rusemshift/rusemshift_all_raw_annotations.jsonl \
    --output-file "rushifteval/data/preds/${model_name}_preds/preds_rusemshift_all.jsonl" \

python model/inference.py "$base_model_name_or_path" $lora_file_path_arg $batch_size \
    --input-file rushifteval/tmp/for_inference/rushifteval/rushifteval1_test_raw_annotations.jsonl \
    --output-file "rushifteval/data/preds/${model_name}_preds/preds_rushifteval1_test.jsonl"

python model/inference.py "$base_model_name_or_path" $lora_file_path_arg $batch_size \
    --input-file rushifteval/tmp/for_inference/rushifteval/rushifteval2_test_raw_annotations.jsonl \
    --output-file "rushifteval/data/preds/${model_name}_preds/preds_rushifteval2_test.jsonl"

python model/inference.py "$base_model_name_or_path" $lora_file_path_arg $batch_size \
    --input-file rushifteval/tmp/for_inference/rushifteval/rushifteval3_test_raw_annotations.jsonl \
    --output-file "rushifteval/data/preds/${model_name}_preds/preds_rushifteval3_test.jsonl"
