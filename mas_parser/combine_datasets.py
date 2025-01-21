# pylint: disable=too-many-nested-blocks
"""Module to combine two datasets into one."""
import argparse
import re
from copy import deepcopy

from tqdm import tqdm

from parse_mas_utils import dump_dataset, load_dataset, parse_path


def merge_examples(list1: list[str], list2: list[str]) -> list[str]:
    """Merge two lists of examples, avoiding duplicates.

    :param list1: First list of examples.
    :param list2: Second list of examples.
    :return: A merged list of unique examples.
    """
    merged_list = list1.copy()
    for example in list2:
        similar_found = False
        for other_example in merged_list:
            if is_similar(example, other_example):
                similar_found = True
                break
        if not similar_found:
            merged_list.append(example)
    return merged_list


non_alphanumeric_re = re.compile(r'[^а-яА-ЯЁё\s]')


def is_similar(string_1: str, string_2: str) -> bool:
    """Check if two strings are similar.

    :param string_1: First string.
    :param string_2: Second string.
    :return: True if the strings are similar, False otherwise.
    """
    string_1 = non_alphanumeric_re.sub('', string_1).lower()
    string_2 = non_alphanumeric_re.sub('', string_2).lower()

    return string_1 == string_2


def combine_datasets(dataset1: list[dict], dataset2: list[dict]) -> list[dict]:
    """Combine two datasets into a single dataset.

    :param dataset1: First dataset.
    :param dataset2: Second dataset.
    :return: A combined dataset.
    """
    combined_dataset: list[dict] = []
    seen_titles = set()

    for entry in tqdm(dataset1 + dataset2):
        title = entry['title']
        if title in seen_titles:
            combined_entry = next((item for item in combined_dataset if item['title'] == title),
                                  None)
            if combined_entry:
                for definition, examples_dict in entry['definitions'].items():
                    examples = examples_dict['examples']
                    definition_found = False
                    for combined_definition in combined_entry['definitions']:
                        if is_similar(definition, combined_definition):
                            combined_entry['definitions'][combined_definition]['examples'] = (
                                merge_examples(combined_entry['definitions'][combined_definition]
                                               ['examples'], examples))
                            definition_found = True
                            break
                    if not definition_found:
                        combined_entry['definitions'][definition] = {"examples": examples}
        else:
            combined_dataset.append(deepcopy(entry))
            seen_titles.add(title)

    return combined_dataset


def main():
    """Combine two JSONL datasets into one."""
    parser = argparse.ArgumentParser(description="Combine two JSONL datasets into one.")
    parser.add_argument("file_path1",
                        type=str,
                        help="Path to the first dataset file.")
    parser.add_argument("file_path2",
                        type=str,
                        help="Path to the second dataset file.")
    parser.add_argument("output_file_path",
                        type=str,
                        help="Path to the output combined dataset file.")

    args = parser.parse_args()

    dataset1 = load_dataset(parse_path(args.file_path1))
    dataset2 = load_dataset(parse_path(args.file_path2))
    combined_dataset = combine_datasets(dataset1, dataset2)
    dump_dataset(combined_dataset, parse_path(args.output_file_path))


if __name__ == "__main__":
    main()
