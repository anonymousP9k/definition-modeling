#!/bin/bash

words=(
    "знатный:0.2 10"
    "кануть:0.18 10"
    "классный:0.22 5"
    "мама:0.08 5"
    "машина:0.15 5"
    "молодец:0.18 10"
    "пакет:0.12 8"  # "пакет:0.12 15"
    "передовой:0.18 12"  # 0.18 20
    "пионер:0.18 15"
    "пожалуй:0.15 10"
    "пока:0.23 5"
    "привет:0.1 20"  # 0.15 10
    "пружина:0.25 50"
    "публика:0.11 10"
    "свалка:0.12 25"
    "сволочь:0.1 10"
    "стиль:0.15 25"
    "тётка:0.1 15"
    "тройка:0.28 12"  # 0.28 12
    "червяк:0.16 10"
)

for word_value in "${words[@]}"; do
    IFS=':' read -ra word_parts <<< "$word_value"
    word="${word_parts[0]}"
    IFS=' ' read -ra values <<< "${word_parts[1]}"
    eps="${values[0]}"
    min_samples="${values[1]}"
    python3 vizvector/visualize.py ruscorpora/data/vectors/ruscorpora_sample_300_seed_42_vectors.jsonl \
        "$word" --eps "$eps" --min-samples "$min_samples" \
        --output-path "ruscorpora/tmp/visualizations/${word}"
    python3 vizvector/visualize.py ruscorpora/data/vectors/ruscorpora_sample_300_seed_42_vectors.jsonl \
        "$word" --eps "$eps" --min-samples "$min_samples" \
        --output-path "ruscorpora/tmp/visualizations/${word}_minimal" --minimal
done
