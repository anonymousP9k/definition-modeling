#!/bin/bash

bash config/pylint/run_pylint.sh
bash config/run_mypy.sh
bash config/run_tests.sh
bash config/spellcheck/run_spellcheck.sh
bash config/run_flake8.sh
bash config/run_pymarkdownlnt.sh
bash config/requirements_check/run_requirements_check.sh
bash config/run_docstrings_check.sh
bash config/newline_check/run_newline_check.sh
bash config/run_coverage.sh
