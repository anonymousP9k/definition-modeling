#!/bin/bash

PROJECT_DIR=$(pwd)
EXCLUDE_DIRS=("venv")

is_excluded() {
    dir="$1"
    for excluded in "${EXCLUDE_DIRS[@]}"; do
        if [ "$dir" == "$excluded" ]; then
            return 0
        fi
    done
    return 1
}

run_isort_on_file() {
    file="$1"
    if [ -f "$file" ] && [[ "$file" == *.py ]]; then
        isort "$file"
        echo "Ran isort on: $file"
    fi
}

run_isort_on_directory() {
    directory="$1"
    for file in "$directory"/*; do
        if [ -d "$file" ]; then
            dir_name=$(basename "$file")
            if ! is_excluded "$dir_name"; then
                run_isort_on_directory "$file"
            fi
        else
            run_isort_on_file "$file"
        fi
    done
}

run_isort_on_project() {
    project_dir="$1"
    run_isort_on_directory "$project_dir"
}

run_isort_on_project "$PROJECT_DIR"
