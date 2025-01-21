# pylint: disable=too-many-return-statements,too-many-locals,too-many-statements,too-many-branches
"""Module to parse MAS articles."""
import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import pymorphy3
import spacy
from bs4 import BeautifulSoup, Tag
from pymorphy3 import MorphAnalyzer
from razdel import tokenize
from spacy import Language
from tqdm import tqdm

from parse_mas_utils import DatasetEntry, load_parse_config, parse_path, ParseMASConfig, ROOT_DIR


def check_if_text_is_example(text: str, title: str, mas_config: ParseMASConfig,
                             nlp: Language, morph: MorphAnalyzer) -> bool:
    """
    Check if a text is an example based on the title.

    :param text: The text to check.
    :param title: The title of the article.
    :param mas_config: The configuration object.
    :param nlp: A spaCy Language object.
    :param morph: A pymorphy3 MorphAnalyzer object.
    :return: True if the text is an example, False otherwise.
    """
    for tag in [*mas_config.tags_to_remove, *mas_config.other_tags]:
        if re.search(rf"\b{re.escape(tag)}", text, flags=re.IGNORECASE):
            return False

    if len(text) > 40:
        return True

    # Fixing the bug with "дву/двух..." titles
    if title.lower().startswith("дву"):
        if ";" in title:
            words = title.split(";")
            for word in words:
                if word.strip()[:-2] in text.lower():
                    return True
        else:
            if title[:-2] in text.lower():
                return True

    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc]).lower()
    lemmatized_title = " ".join([token.lemma_ for token in nlp(title)]).lower()

    # Alternative lemmatization with pymorphy2
    alternative_lemmatized_text = " ".join(
        [morph.parse(token.text)[0].normal_form for token in tokenize(text)])

    if ";" not in title:
        title_pattern = re.compile(rf"\b{re.escape(title.lower())}")
        lemmatized_title_pattern = re.compile(rf"\b{re.escape(lemmatized_title.lower())}")
        if (title_pattern.search(lemmatized_text)
                or lemmatized_title_pattern.search(lemmatized_text)
                or title_pattern.search(alternative_lemmatized_text) or
                lemmatized_title_pattern.search(alternative_lemmatized_text)):
            return True

    for word in title.split(";"):
        word_pattern = re.compile(rf"\b{re.escape(word)}")
        lemmatized_word = " ".join([token.lemma_ for token in nlp(word)]).lower()
        lemmatized_word_pattern = re.compile(rf"\b{re.escape(lemmatized_word.lower())}")
        if (word_pattern.search(lemmatized_text) or lemmatized_word_pattern.search(lemmatized_text)
            or word_pattern.search(alternative_lemmatized_text) or
                lemmatized_word_pattern.search(alternative_lemmatized_text)):
            return True

    return False


def process_definition(raw_definition: str, mas_config: ParseMASConfig) -> str:
    """
    Process a raw definition string and return a cleaned version without tags and garbage.

    :param raw_definition: The raw definition string.
    :param mas_config: The configuration object.
    :return: The cleaned definition string.
    """
    # Finds the first character in a word that starts with uppercase letter
    # and removes everything before it
    match = re.search(r'[A-ZА-ЯЁ][a-zа-яё\s]+', raw_definition)
    if match:
        raw_definition = raw_definition[match.start():]

    # For words like "активный1" or "актив2", removes the number at the end,
    # but keeps separate numbers
    raw_definition = re.sub(r'\b([A-ZА-Яa-zа-яЁё]+)\d+\b', r'\1', raw_definition)

    if mas_config.remove_tags:
        for tag in mas_config.tags_to_remove:
            # Replace the tag with an empty string, it could be both uppercase and lowercase
            pattern = rf"\b{re.escape(tag)}"
            len_before = len(raw_definition)
            raw_definition = re.sub(pattern, '', raw_definition, flags=re.IGNORECASE)
            if len(raw_definition) != len_before:
                pass

    # Remove things like "во 2 знач." or "в 1 знач."
    raw_definition = re.sub(r'\(?во?\s((\d+,?\s?)+и?\s?\d*\s?)знач\.\)?', '',
                            raw_definition)

    # Remove wrong punctuation from the right side
    raw_definition = re.sub(r'[|:;—-]\s*$', '', raw_definition)

    # Remove empty parentheses
    raw_definition = re.sub(r'\([\s.,;:—-]*\)', '', raw_definition)

    # Remove extra spaces
    raw_definition = re.sub(r'\s*,\s*', ', ', raw_definition)
    raw_definition = re.sub(r',\s*,', ',', raw_definition)
    raw_definition = re.sub(r',\s+$', '', raw_definition)
    raw_definition = re.sub(r'\s+,', ',', raw_definition)
    raw_definition = re.sub(r'\s+\.', '.', raw_definition)
    raw_definition = re.sub(r'\s+;', ';', raw_definition)
    raw_definition = re.sub(r'\s+', ' ', raw_definition)

    # Remove starting commas
    while raw_definition.startswith(", "):
        raw_definition = re.sub(r'^,\s*', '', raw_definition)

    # Remove — at the end
    raw_definition = re.sub(r'—\s*$', '', raw_definition)

    raw_definition = raw_definition.strip()

    if raw_definition.startswith("и"):
        raw_definition = raw_definition[1:]
        raw_definition = raw_definition.strip()

    return raw_definition


def validate_definition(definition_text: str) -> bool:
    """
    Validate an entry by checking if the definition is correct.

    :param definition_text: The definition text.
    :return: True if the entry is valid, False otherwise.
    """
    if not definition_text:
        return False

    prohibited = ["||", "]", "---", "□", " -,", "-;", "…", "-."]
    if any(p in definition_text for p in prohibited):
        return False

    prohibited_pattern = r"-[а-яёо́]+[\b,]|\s[мж]\."
    match = re.search(prohibited_pattern, definition_text)
    if match and match.group() != "-л":
        return False

    with open(ROOT_DIR / "mas_parser" / "last_names.txt", "r",
              encoding="utf-8") as last_names_file:
        last_names = last_names_file.read().split("\n")

    if any(ln in definition_text for ln in last_names):
        return False

    return True


def validate_example(example_text: str, mas_config: ParseMASConfig) -> bool:
    """
    Validate an example by checking if it's correct.

    :param example_text: The example text.
    :param mas_config: The configuration object.
    :return: True if the example is valid, False otherwise.
    """
    for tag in [*mas_config.tags_to_remove, *mas_config.other_tags]:
        pattern = rf"\b{re.escape(tag)}"
        if re.search(pattern, example_text, flags=re.IGNORECASE):
            return False

    return True


def parse_meaning(lines: list[str], title: str, mas_config: ParseMASConfig,
                  nlp: Language, morph: MorphAnalyzer) -> dict[str, dict]:
    """
    Parse a meaning from the given lines.

    :param lines: The lines to parse.
    :param title: The title of the article.
    :param mas_config: The configuration object.
    :param nlp: A spaCy Language object.
    :param morph: A pymorphy3 MorphAnalyzer object.
    :return: A dictionary with the definition and examples.
    """
    if len(lines) < 2:
        return {}

    examples = []

    # If the first line starts with <b>\d\.<b>, but the line continues, split it
    if re.match(r"<b>\d+\.</b>", lines[0]) and len(lines[0]) > 10:
        first_line = lines[0]
        lines = [first_line[:first_line.index("</b>") + 4],
                 first_line[first_line.index("</b>") + 4:]] + lines[1:]

    # If the tag's content is split between two lines, merge them
    for i in range(len(lines) - 1):
        right_i_start = lines[i].rfind("<i>")
        right_i_end = lines[i].rfind("</i>")

        if right_i_start != -1 and right_i_start > right_i_end:
            # Replace the lines[i + 1] with the unsplit version, fix the current line
            lines[i + 1] = "<i>" + lines[i][right_i_start + 3:] + lines[i + 1]
            # Remove the <i> part from the current line
            lines[i] = lines[i][:right_i_start]

        right_parenthesis_start = lines[i].rfind("(")
        right_parenthesis_end = lines[i].rfind(")")

        if right_parenthesis_start != -1 and right_parenthesis_start > right_parenthesis_end:
            # print(lines)
            # Replace the lines[i + 1] with the unsplit version, fix the current line
            lines[i + 1] = "(" + lines[i][right_parenthesis_start + 1:] + lines[i + 1]
            # Remove the <i> part from the current line
            lines[i] = lines[i][:right_parenthesis_start]

    # Remove (<i>сов.</i> word). lines
    pattern = re.compile(r'\(<i>(?:не)?сов\.</i>\s[а-яё]+\)\.?')

    for i in range(len(lines) - 1, -1, -1):
        original_line = lines[i]

        match = re.search(pattern, original_line)

        if not match:
            continue

        # Remove the matched pattern,
        # ensuring a trailing dot after the closing parenthesis is included
        modified_line = re.sub(pattern, '', original_line).strip()

        # If the modified line is empty, it indicates the entire line was just the matched pattern
        if not modified_line:
            lines.pop(i)
        else:
            lines[i] = modified_line

    # Remove (<i>Разг.</i>) like lines
    pattern = re.compile(r'\(<i>[А-ЯЁа-яё]+\.</i>\)')
    for i in range(len(lines) - 1, -1, -1):
        original_line = lines[i]

        match = re.search(pattern, original_line)

        if not match:
            continue

        # print("Removing:", original_line, "because of pattern:", pattern)
        lines[i] = re.sub(pattern, '', original_line).strip()

    # Remove (часто в сочетании: ...) lines
    pattern = re.compile(r'\(<i>часто в сочетании:<\/i>\s<em>[^<]+<\/em>(,\s<em>[^<]+<\/em>)+\)')
    for i in range(len(lines) - 1, -1, -1):
        original_line = lines[i]

        match = re.search(pattern, original_line)

        if not match:
            continue

        lines[i] = re.sub(pattern, '', original_line).strip()

    # Remove (<i>Вышло из употребления ...</i>) lines
    pattern = re.compile(r'\(<i>Вышло из употребления[а-яёА-ЯЁ\s]+</i>\)\.?')
    for i in range(len(lines) - 1, -1, -1):
        original_line = lines[i]

        match = re.search(pattern, original_line)

        if not match:
            continue

        # print("Removing:", original_line, "because of pattern:", pattern)
        lines[i] = re.sub(pattern, '', original_line).strip()

    starting_examples = []

    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    # Check the texts under <i> tags on the second line.
    # If they are examples, move them to the next line
    initial_length = min(len(lines), 4)

    for i in range(1, initial_length):
        for i_tag in BeautifulSoup(lines[i], 'html.parser').find_all('i'):
            if check_if_text_is_example(i_tag.get_text(), title, mas_config, nlp=nlp, morph=morph):
                i_tag_index = lines[i].find(str(i_tag))
                if i_tag_index == -1:
                    return {}
                starting_examples.append(lines[i][i_tag_index:])
                lines[i] = lines[i][:i_tag_index]
                if not lines[i].strip():
                    lines[i] = "<IGNORE>"
                break

    if len(lines) < 2:
        return {}  # Skip if there are less than 2 lines

    second_line_inside_tags = (lines[1].strip().startswith("(") and
                               (lines[1].strip().endswith(")") or lines[1].strip().endswith(").")))

    if ((("<i>" in lines[1] or re.match("-[а-яёо́]+,", lines[1]))
         and not BeautifulSoup(lines[1], "html.parser").find("a")) or
            ("<em>" in lines[1]) or second_line_inside_tags):
        # Extract definition
        if len(lines) > 3 and not lines[3] == "<IGNORE>":
            definition_soup = BeautifulSoup(" ".join([lines[1].replace('\xa0', ' '),
                                                      lines[2].replace('\xa0', ' '),
                                                      lines[3].replace('\xa0', ' ')]),
                                            "html.parser")
        elif len(lines) > 2 and not lines[2] == "<IGNORE>":
            definition_soup = BeautifulSoup(" ".join([lines[1].replace('\xa0', ' '),
                                                      lines[2].replace('\xa0', ' ')]),
                                            "html.parser")
        else:
            definition_soup = BeautifulSoup(lines[1].replace('\xa0', ' '),
                                            "html.parser")

        lines_for_examples = starting_examples + lines[3:]
    else:
        lines_for_examples = starting_examples + lines[2:]
        if len(lines) > 2 and "<i" not in lines[2] and not lines[2] == "<IGNORE>":
            definition_soup = BeautifulSoup(" ".join([lines[1].replace('\xa0', ' '),
                                                      lines[2].replace('\xa0', ' ')]),
                                            "html.parser")
        else:
            definition_soup = BeautifulSoup(lines[1].replace('\xa0', ' '),
                                            "html.parser")

    for em_tag in definition_soup.find_all('em'):
        if em_tag.find("a") or em_tag.find("xa"):
            continue
        index_of_em_tag = definition_soup.decode_contents().find(
            f"<em>{em_tag.decode_contents()}</em>")
        if (index_of_em_tag > 0 and "соответствует по значению" in definition_soup.
            decode_contents().lower()[:index_of_em_tag]):
            continue

        em_tag.extract()

    definition_text = process_definition(definition_soup.get_text(), mas_config)

    examples_text = BeautifulSoup("\n".join(lines_for_examples), 'html.parser')

    for example in examples_text.find_all('i'):
        example_text = example.get_text().strip()
        if validate_example(example_text, mas_config):
            examples.append(example_text)

    for i in range(len(examples) - 2, -1, -1):
        next_example = examples[i + 1].strip()
        example = examples[i].strip()
        if example[-1] != ".":
            examples[i] = example + " " + next_example
            examples.pop(i + 1)

    if not validate_definition(definition_text):
        return {}

    return {"definitions": {definition_text: {"examples": examples}}}


def preprocess_html_content(html_content: str) -> str:
    """
    Preprocess HTML content by removing incorrect blockquotes.

    :param html_content: HTML content as a string.
    :return: Preprocessed HTML content.
    """
    pattern = (r"(?:</i>)?\\n</BLOCKQUOTE><BLOCKQUOTE class=page><P>(?:<p>)"
               r"?<span class=page id=\$p\d+>- \d+ -</span></P>(?:</p>)?\\n(?:<i>)?")

    while re.findall(pattern, html_content)[1:]:
        match = re.findall(pattern, html_content)[1]

        html_content = (html_content[:html_content.index(match)] + " " +
                        html_content[html_content.index(match) + len(match):])

    return html_content


def preprocess_lines(lines: list[str]) -> list[list[str]]:
    """
    Preprocess lines by splitting submeanings into separate sections.

    :param lines: The lines to preprocess.
    :return: A list of preprocessed sections.
    """
    processed_lines: list[list[str]] = []
    current_section: list[str] = []

    for line in lines:
        # Check if the line indicates a submeaning
        if "||" in line:
            # If current_section is not empty, append it to processed_lines
            if current_section:
                processed_lines.append(current_section)
                current_section = []
            # If line starts with "||" but contains more text
            if line.strip() != "||" and line.startswith("||"):
                # Split the line into a marker and the actual content
                current_section.append("||")
                current_section.append(line[2:].strip())
            else:
                # Otherwise, just add the line to the current section
                current_section.append(line)
        else:
            # For regular lines, just add them to the current section
            current_section.append(line)

    # Append the last section to processed_lines if it's not empty
    if current_section:
        processed_lines.append(current_section)

    return processed_lines


def extract_data_from_html(html_content: str, dataset_entry: DatasetEntry,
                           mas_config: ParseMASConfig,
                           nlp: Language, morph: MorphAnalyzer) -> DatasetEntry:
    """
    Extract data from an HTML page's content.

    :param html_content: HTML content as a string.
    :param dataset_entry: An entry to fill.
    :param mas_config: A ParseMASConfig object with configuration options.
    :param nlp: A spaCy Language object.
    :param morph: A pymorphy3 MorphAnalyzer object.
    :return: A dictionary with extracted data.
    """
    html_content = preprocess_html_content(html_content)

    soup = BeautifulSoup(html_content, 'html.parser')

    title = "Unknown"

    # Find the title within <h4> tags
    title_tag = soup.find('h4')
    if title_tag and isinstance(title_tag, Tag):
        title_data = title_tag.get('title')
        title = title_data if isinstance(title_data, str) else "Unknown"
        if title and title[-1].isdigit():
            title = title[:-1]
        if title:
            title = title.replace("'", "")

    dataset_entry.title = title

    if title in mas_config.ignore_entries:
        return dataset_entry

    pos_tags = soup.find_all('blockquote')
    for pos_tag in pos_tags:
        paragraphs = pos_tag.find_all('p', recursive=True)

        for paragraph in paragraphs[2:]:
            if paragraph.find('p'):
                continue

            # Check if paragraph contains a definition (indicated by <b> tags)
            if paragraph.find('b'):
                if "◊" in paragraph.text:  # Means phrases go next, we don't need them
                    break

                lines = paragraph.decode_contents().split("\n")

                if not lines:
                    continue

                preprocessed_sections = preprocess_lines(lines)

                for section in preprocessed_sections:
                    meaning_data = parse_meaning(section, title, mas_config, nlp=nlp, morph=morph)
                    if meaning_data:
                        dataset_entry.definitions.update(meaning_data["definitions"])

    return dataset_entry


def process_jsonl_file(input_file: Path, output_file: Path, mas_config: ParseMASConfig,
                       nlp: Language, morph: MorphAnalyzer) -> None:
    """Process HTML content and append extracted data to a JSON Lines file.

    :param input_file: A Path object pointing to the input JSON Lines file.
    :param output_file: A Path object for the output JSON Lines file.
    :param mas_config: A ParseMASConfig object with configuration options.
    :param nlp: A spaCy Language object.
    :param morph: A pymorphy3 MorphAnalyzer object.
    """
    with (input_file.open('r', encoding='utf-8') as in_f,
          output_file.open('a', encoding='utf-8') as out_f):
        for line in tqdm(in_f.readlines()):
            record = json.loads(line)
            dataset_entry = DatasetEntry(id=record['id'], title="", definitions={})
            dataset_entry = extract_data_from_html(record['html'], dataset_entry, mas_config,
                                                   nlp=nlp, morph=morph)
            out_f.write(json.dumps(asdict(dataset_entry), ensure_ascii=False) + '\n')


def main() -> None:
    """Parse MAS articles."""
    parser = argparse.ArgumentParser(description="Parse MAS articles")
    parser.add_argument("--input-file", type=str,
                        default="mas_parser/data/mas_articles.jsonl",
                        help="The path to the input JSON Lines file")
    parser.add_argument("--output-file", type=str,
                        default="mas_parser/data/mas_definitions.jsonl",
                        help="The path to the output JSON Lines file")
    args = parser.parse_args()

    input_file_path = parse_path(args.input_file)
    output_file_path = parse_path(args.output_file)

    mas_parse_config = load_parse_config(ROOT_DIR / "mas_parser" / "parse_config.json")

    spacy_model = spacy.load("ru_core_news_sm", disable=["ner", "parser", "textcat"])
    morph_analyzer = pymorphy3.MorphAnalyzer()

    process_jsonl_file(input_file_path, output_file_path, mas_parse_config, nlp=spacy_model,
                       morph=morph_analyzer)


if __name__ == "__main__":
    main()
