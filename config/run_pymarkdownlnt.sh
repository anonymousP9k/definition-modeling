#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running pymarkdownlnt check...'

configure_script

python3 -m pymarkdown scan *.md

directories=$(get_project_directories)

for directory in $directories; do
  python3 -m pymarkdown scan "${directory}"

  check_if_failed
done

echo "Pymarkdownlnt check passed."
