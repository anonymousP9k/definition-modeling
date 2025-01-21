configure_script() {
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source venv/bin/activate
    export PYTHONPATH=$(pwd):$PYTHONPATH
    which python
    python -m pip list
  elif [[ "$OSTYPE" == "msys" ]]; then
    source venv/Scripts/activate
    export PYTHONPATH=$(pwd)
  fi
}

check_if_failed() {
  if [[ $? -ne 0 ]]; then
    echo "Check failed."
    exit 1
  else
    echo "Check passed."
  fi
}

get_project_directories() {
  local directories=('wiktionary_parser' 'model' 'vizvector' 'rushifteval' 'ruscorpora')
  echo ${directories[@]}
}
