#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running spellcheck...'

configure_script

python3 -m pyspelling -c config/spellcheck/.spellcheck.yaml -v

check_if_failed

echo "Spellcheck passed."
