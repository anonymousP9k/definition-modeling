#!/bin/bash
source config/common.sh

set -ex

echo -e '\n'
echo 'Running docstring style check...'

configure_script

pydocstyle config
darglint --docstring-style sphinx --strictness long config

directories=$(get_project_directories)

for directory in $directories; do
  pydocstyle "${directory}"
  check_if_failed

  darglint --docstring-style sphinx --strictness long "${directory}"
  check_if_failed
done

echo "Docstring style check passed."
