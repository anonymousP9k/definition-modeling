"""A module to test the combine_datasets module."""
import json
import unittest
from pathlib import Path

import pytest

from mas_parser.combine_datasets import combine_datasets
from mas_parser.parse_mas_utils import load_dataset


class CombineDatasetsTest(unittest.TestCase):
    """Tests for combine_datasets module."""
    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "data"
        self.dataset_1_path = self.test_data_dir / "dataset_1.jsonl"
        self.dataset_2_path = self.test_data_dir / "dataset_2.jsonl"
        self.expected_dataset_path = self.test_data_dir / "expected_combined_dataset.jsonl"

        self.output_dataset = self.test_data_dir / "dumped_dataset.jsonl"

    @pytest.mark.mas_parser
    def test_combine_dataset(self):
        """Test for combine_dataset function."""
        with open(self.expected_dataset_path, "r", encoding="utf-8") as expected_file:
            expected = json.load(expected_file)

        dataset_1 = load_dataset(self.dataset_1_path)
        dataset_2 = load_dataset(self.dataset_2_path)
        actual = combine_datasets(dataset_1, dataset_2)

        actual_entry_definitions = actual[0]["definitions"]
        expected_entry_definitions = expected["definitions"]

        for definition in (*actual_entry_definitions.keys(), *expected_entry_definitions.keys()):
            if definition not in actual_entry_definitions:
                assert False, f"Definition {definition} not found in actual: {actual}"
            if definition not in expected_entry_definitions:
                assert False, f"Definition {definition} not found in expected. Actual: {expected}"

            assert set(actual_entry_definitions[definition]) == set(expected_entry_definitions[definition])

    def tearDown(self):
        if self.output_dataset.exists():
            self.output_dataset.unlink()
