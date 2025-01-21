#!/bin/bash
set -ex

which python

python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv venv

source venv/bin/activate

which python

INCLUDE_TRAIN=false

while getopts ":t" opt; do
  case $opt in
    t)
      INCLUDE_TRAIN=true
      ;;
    *)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

if [[ "$INCLUDE_TRAIN" == true ]]; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements_train.txt
fi

python -m pip install -r requirements.txt
python -m pip install -r requirements_ci.txt
