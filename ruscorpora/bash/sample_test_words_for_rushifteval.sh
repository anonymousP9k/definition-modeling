#!/bin/bash

python ruscorpora/process_ruscorpora_dump.py ruscorpora/process_ruscorpora_test_words_config.json
python ruscorpora/sample_ruscorpora.py ruscorpora/tmp/rushifteval_test_words.jsonl ruscorpora/tmp/ruscorpora_sampled_100_test_words.jsonl
python ruscorpora/recreate_rushifteval_pairs.py ruscorpora/tmp/ruscorpora_sampled_100_test_words.jsonl
