"""A module to test the parse_mas module."""
import json
import unittest
from pathlib import Path

import pymorphy3
import pytest
import spacy

from mas_parser.parse_mas import process_jsonl_file
from mas_parser.parse_mas_utils import load_parse_config


class CleanMasDatasetTest(unittest.TestCase):
    """Tests for clean_mas_dataset module."""
    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "data"
        self.parse_config = load_parse_config(self.test_data_dir / "test_parsing_config.json")
        self.raw_articles_path = self.test_data_dir / "raw_articles.jsonl"
        self.expected_parsed_dataset_path = self.test_data_dir / "expected_parsed_dataset.jsonl"
        self.output_path = self.test_data_dir / "parsed_dataset.jsonl"

    @pytest.mark.mas_parser
    def test_clean_dataset(self):
        """Test for clean_dataset function."""
        nlp = spacy.load("ru_core_news_sm", disable=["ner", "parser", "textcat"])
        morph = pymorphy3.MorphAnalyzer()

        process_jsonl_file(self.raw_articles_path, self.output_path, self.parse_config,
                           nlp=nlp, morph=morph)

        with open(self.output_path, "r", encoding="utf-8") as output_file:
            actual = [json.loads(line) for line in output_file if line.strip()]

        with open(self.expected_parsed_dataset_path, "r", encoding="utf-8") as expected_file:
            expected = [json.loads(line) for line in expected_file if line.strip()]

        for actual_entry, expected_entry in zip(actual, expected):
            assert actual_entry == expected_entry

    def tearDown(self):
        self.output_path.unlink()
