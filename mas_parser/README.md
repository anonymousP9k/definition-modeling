# MAS parser

This parser is designed to extract definitions and examples of usage from
the [MAS](http://feb-web.ru/feb/mas/mas-abc/default.asp) dictionary
and to prepare the data for being used in the definition modeling task.

Example of the output for a page:

```json lines
{
  "id": 13226034,
  "title": "метиловый",
  "definitions": {
    "содержащий метил": {
      "examples": ["Метиловый эфир."]
    }
  }
}
```

## How to run

### 1. Project set up

Open the Terminal or Command Prompt in the root directory of this repository.

```bash
cd path/to/work-definition-modeling
```

Activate the virtual environment.

For Windows:

```bash
python3 -m venv venv
.\venv\Scripts\activate
```

For Linux and macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Make sure you've installed the requirements.
You can do it by running

```bash
python -m pip install -r requirements.txt
```

Also, you need to install the `ru_core_news_sm` model for the spaCy library.

```bash
python -m spacy download ru_core_news_sm
```

## Download the data from the website

Run `download_mas.py` to download the data from the website.

```bash
python download_mas.py
```

It will create a `.jsonl` file with `html` pages of the dictionary.

You can also specify the location of the output file and the need to continue the download
if the file already exists from the previous unfinished download.

```bash
python download_mas.py --output-path mas_parser/data/mas_articles.jsonl \
--continue-from-the-last-url
```

### 2. Parse the data

Run `parse_mas.py` to parse the data.

```bash
python parse_mas.py --input-path mas_parser/data/mas_articles.jsonl \
--output-path mas_parser/data/mas_definitions.jsonl
```

You can set parameters for the parser in `parser_config.json` file.

These parameters are:

- `remove_tags`: whether to remove tags like `спец.` or `устар.` from the definitions.
- `tags_to_remove`: a list of tags to remove.
- `other_tags`: a list of other tags that can be found in the definitions, used for
detecting definitions.
- `ignore_entries`: a list of entries to ignore.

### 3. Prepare the data for the definition modeling task

Run `clean_mas_dataset.py` to prepare the data for the definition modeling task.

```bash
python clean_mas_dataset.py --input-path mas_parser/data/mas_definitions.jsonl \
--output-path mas_parser/data/cleaned_mas_definitions.jsonl
```

You can set parameters for the cleaner in `mas_cleaning_config.json` file.

These parameters are:

- `max_definition_character_length`: the maximum length of the definition.
- `remove_entries_without_examples`: whether to remove entries without examples.
- `throw_out_definition_markers`: will ignore definitions that contain these markers.
- `replace`: a dictionary with the replacement rules for the definitions.
