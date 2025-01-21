#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running pytest...'

configure_script

directories=$(get_project_directories)
INITIAL_PYTHONPATH=$PYTHONPATH
FAILED=false

if [[ -z "${directories}" ]]; then
  echo "No tests to run currently."
  exit 0
fi

for directory in $directories; do
  export PYTHONPATH="${INITIAL_PYTHONPATH}:$(pwd)/${directory}"
  python3 -m pytest "$directory"

  if [[ $? -ne 0 ]]; then
    FAILED=true
  fi
done

export PYTHONPATH=$INITIAL_PYTHONPATH

if [[ "$FAILED" = true ]]; then
  echo "Pytest failed."
  exit 1
fi

echo "Pytest passed."
