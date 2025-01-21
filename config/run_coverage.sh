#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running coverage check...'

configure_script

directories=$(get_project_directories)

if [[ -z "${directories}" ]]; then
  echo "No files to run coverage on currently."
  exit 0
fi

coverage run -m pytest ${directories}
coverage report

check_if_failed
