"""Module to create RuShiftEval paired datasets."""
import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from ruscorpora_utils import parse_path, ROOT_DIR, write_results_to_file


def load_data(file_path: Path) -> list[dict]:
    """
    Load JSON Lines file into a list of dictionaries.

    :param file_path: Path to the JSON Lines file.
    :return: List of dictionaries representing the JSON Lines.
    """
    data = []
    with file_path.open('r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Loading data"):
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError:
                continue  # Skip malformed JSON lines
    return data


def map_date_to_period(date_str: str) -> str | None:
    """
    Map the date string to a specific historical period.

    :param date_str: Date string from the entry.
    :return: Period identifier ('pre', 'soviet', 'post') or None if unknown.
    """
    period_mapping = {
        '1700-1916': 'pre',
        '1918-1990': 'soviet',
        '1918-1991': 'soviet',
        '1992-2016': 'post',
        '1992-2017': 'post',
    }
    return period_mapping.get(date_str)


def organize_entries(
    data: list[dict]
) -> dict[str, dict[str, list[dict]]]:
    """
    Organize entries by word and period.

    :param data: List of dictionaries representing the JSON Lines.
    :return: Nested dictionary with structure words[word][period] = list of entries.
    """
    words: dict[str, dict[str, list[dict]]] = {}

    for entry in tqdm(data, desc="Organizing entries"):
        word = entry.get('word')
        date = entry.get('date')
        if not word or not date:
            continue  # Skip entries missing required fields
        period = map_date_to_period(date)
        if period:
            if word not in words:
                words[word] = {}
            if period not in words[word]:
                words[word][period] = []
            words[word][period].append(entry)

    print("Entries have been successfully organized by word and period:")
    for word_key, periods in words.items():
        pre_count = len(periods.get('pre', []))
        soviet_count = len(periods.get('soviet', []))
        post_count = len(periods.get('post', []))
        print(f"{word_key}: pre={pre_count}, soviet={soviet_count}, post={post_count}")

    return words


def create_paired_entries(
    words: dict[str, dict[str, list[dict]]],
    sample_size: int = 100
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Create paired entries for RuShiftEval datasets.

    :param words: Nested dictionary organized by word and period.
    :param sample_size: Number of pairs to create per word per comparison.
    :return: Tuple containing lists for RuShiftEval-1, RuShiftEval-2, and RuShiftEval-3.
    """
    ree1 = []  # Pre-Soviet vs Soviet
    ree2 = []  # Soviet vs Post-Soviet
    ree3 = []  # Pre-Soviet vs Post-Soviet

    for word, periods in tqdm(words.items(), desc="Creating paired entries"):
        pre_entries = periods.get('pre', [])
        soviet_entries = periods.get('soviet', [])
        post_entries = periods.get('post', [])

        min_pairs_ree1 = min(len(pre_entries), len(soviet_entries), sample_size)
        min_pairs_ree2 = min(len(soviet_entries), len(post_entries), sample_size)
        min_pairs_ree3 = min(len(pre_entries), len(post_entries), sample_size)

        for i in range(min_pairs_ree1):  # Pre-Soviet vs Soviet
            pair_id = i + 1
            ree1.append({
                'id': pair_id,
                'sentence_id': 1,
                'word': word,
                'sentence': pre_entries[i]['sentence'],
                'input_text': pre_entries[i]['input_text'],
            })
            ree1.append({
                'id': pair_id,
                'sentence_id': 2,
                'word': word,
                'sentence': soviet_entries[i]['sentence'],
                'input_text': soviet_entries[i]['input_text'],
            })

        for i in range(min_pairs_ree2):  # Soviet vs Post-Soviet
            pair_id = i + 1
            ree2.append({
                'id': pair_id,
                'sentence_id': 1,
                'word': word,
                'sentence': soviet_entries[i]['sentence'],
                'input_text': soviet_entries[i]['input_text'],
            })
            ree2.append({
                'id': pair_id,
                'sentence_id': 2,
                'word': word,
                'sentence': post_entries[i]['sentence'],
                'input_text': post_entries[i]['input_text'],
            })

        for i in range(min_pairs_ree3):  # Pre-Soviet vs Post-Soviet
            pair_id = i + 1
            ree3.append({
                'id': pair_id,
                'sentence_id': 1,
                'word': word,
                'sentence': pre_entries[i]['sentence'],
                'input_text': pre_entries[i]['input_text'],
            })
            ree3.append({
                'id': pair_id,
                'sentence_id': 2,
                'word': word,
                'sentence': post_entries[i]['sentence'],
                'input_text': post_entries[i]['input_text'],
            })

    return ree1, ree2, ree3


def main():
    """Execute the RuShiftEval paired datasets creation based on CLI arguments."""
    parser = ArgumentParser(
        description="Create RuShiftEval paired datasets from a JSON Lines file.")
    parser.add_argument("input_file", help="Path to the input JSON Lines file.")
    parser.add_argument(
        "--output_dir",
        default=ROOT_DIR / "rushifteval" / "data" / "rushifteval",
        help="Directory to save the output RuShiftEval JSON Lines files."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of pairs to create per word per comparison (default: 100)."
    )
    args = parser.parse_args()

    input_path = parse_path(args.input_file)
    output_dir = parse_path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(input_path)

    words = organize_entries(data)

    ree1, ree2, ree3 = create_paired_entries(words, sample_size=args.sample_size)

    ree1_path = output_dir / "rushifteval1_test.jsonl"
    ree2_path = output_dir / "rushifteval2_test.jsonl"
    ree3_path = output_dir / "rushifteval3_test.jsonl"

    write_results_to_file(ree1, output_file=ree1_path)
    write_results_to_file(ree2, output_file=ree2_path)
    write_results_to_file(ree3, output_file=ree3_path)

    print("Paired datasets have been successfully created:")
    print(f"- RuShiftEval-1: {ree1_path}")
    print(f"- RuShiftEval-2: {ree2_path}")
    print(f"- RuShiftEval-3: {ree3_path}")


if __name__ == "__main__":
    main()
