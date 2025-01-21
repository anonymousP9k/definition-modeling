"""A module to test the clean_mas_dataset module."""
import json
import unittest
from pathlib import Path

import pytest

from mas_parser.clean_mas_dataset import clean_dataset
from mas_parser.parse_mas_utils import load_cleaning_config, load_dataset, parse_path, ROOT_DIR


class CleanMasDatasetTest(unittest.TestCase):
    """Tests for clean_mas_dataset module."""
    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "data"
        self.config = load_cleaning_config(ROOT_DIR / "mas_parser" / "mas_cleaning_config.json")
        self.dataset_path = self.test_data_dir / "raw_dataset.jsonl"

    @pytest.mark.mas_parser
    def test_clean_dataset(self):
        """Test for clean_dataset function."""
        dataset = load_dataset(parse_path(self.dataset_path))
        actual = clean_dataset(dataset, self.config)

        expected_path = self.test_data_dir / "expected_cleaned_dataset.jsonl"
        with open(expected_path, "r", encoding="utf-8") as expected_file:
            expected = [json.loads(line) for line in expected_file if line.strip()]

        for actual_entry, expected_entry in zip(actual, expected):
            assert actual_entry == expected_entry
