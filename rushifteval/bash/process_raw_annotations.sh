#!/bin/bash

python3 rushifteval/process_raw_annotations.py rushifteval/data/rushifteval rushifteval/tmp/for_inference/rushifteval
python3 rushifteval/process_raw_annotations.py rushifteval/data/rusemshift rushifteval/tmp/for_inference/rusemshift
