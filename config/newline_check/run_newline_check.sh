set -ex
source config/common.sh

echo -e '\n'

echo "Check newline at the end of the file"

configure_script

python3 config/newline_check/newline_check.py

check_if_failed

echo "Newline check passed."
