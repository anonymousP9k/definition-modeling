source config/common.sh
set -ex

echo -e '\n'

echo "Check requirements files"

configure_script

python3 config/requirements_check/requirements_check.py

check_if_failed

echo "Requirements check passed."
