"""Tests for the recreate_rushifteval_pairs module."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ruscorpora.recreate_rushifteval_pairs import (create_paired_entries, load_data,
                                                   map_date_to_period, organize_entries)


class RecreateRushiftevalPairsTest(unittest.TestCase):
    """Tests for the recreate_rushifteval_pairs module."""

    def setUp(self):
        """Set up test data and expected results paths."""
        self.test_data_dir = Path(__file__).parent / "data"
        self.raw_data_path = self.test_data_dir / "raw_data.jsonl"
        self.expected_organized_path = self.test_data_dir / "expected_organized.json"
        self.expected_ree1_path = self.test_data_dir / "expected_ree1.jsonl"
        self.expected_ree2_path = self.test_data_dir / "expected_ree2.jsonl"
        self.expected_ree3_path = self.test_data_dir / "expected_ree3.jsonl"

    def test_load_data(self):
        """
        Test loading JSON Lines data from a file.
        """
        with open(self.raw_data_path, 'r', encoding='utf-8') as file:
            expected = [json.loads(line) for line in file]
        result = load_data(self.raw_data_path)
        self.assertEqual(result, expected)

    def test_map_date_to_period(self):
        """
        Test mapping date strings to historical periods.
        """
        test_cases = [
            ("1700-1916", "pre"),
            ("1918-1990", "soviet"),
            ("1992-2017", "post"),
            ("2000-2020", None),  # Unknown period
            ("1918-1991", "soviet"),
            ("1992-2016", "post"),
        ]

        for date_str, expected_period in test_cases:
            with self.subTest(date_str=date_str):
                period = map_date_to_period(date_str)
                self.assertEqual(period, expected_period)

    def test_organize_entries(self):
        """
        Test organizing entries by word and period.
        """
        data = load_data(self.raw_data_path)
        organized = organize_entries(data)

        with open(self.expected_organized_path, 'r', encoding='utf-8') as file:
            expected = json.load(file)

        self.assertEqual(organized, expected)

    def test_create_paired_entries(self):
        """
        Test creating paired entries for RuShiftEval datasets.
        """
        data = load_data(self.raw_data_path)
        organized = organize_entries(data)
        ree1, ree2, ree3 = create_paired_entries(organized, sample_size=2)

        expected_ree1 = self._load_jsonl(self.expected_ree1_path)
        expected_ree2 = self._load_jsonl(self.expected_ree2_path)
        expected_ree3 = self._load_jsonl(self.expected_ree3_path)

        self.assertEqual(len(ree1), len(expected_ree1))
        self.assertListEqual(ree1, expected_ree1)

        self.assertEqual(len(ree2), len(expected_ree2))
        self.assertListEqual(ree2, expected_ree2)

        self.assertEqual(len(ree3), len(expected_ree3))
        self.assertListEqual(ree3, expected_ree3)

    def test_full_pipeline(self):
        """
        Test the full pipeline of loading, organizing, and creating paired entries.
        """
        with TemporaryDirectory() as temp_dir:
            temp_output_dir = Path(temp_dir)
            data = load_data(self.raw_data_path)
            organized = organize_entries(data)
            ree1, ree2, ree3 = create_paired_entries(organized, sample_size=2)

            ree1_path = temp_output_dir / "rushifteval1_test.jsonl"
            ree2_path = temp_output_dir / "rushifteval2_test.jsonl"
            ree3_path = temp_output_dir / "rushifteval3_test.jsonl"

            self._write_jsonl(ree1, ree1_path)
            self._write_jsonl(ree2, ree2_path)
            self._write_jsonl(ree3, ree3_path)

            actual_ree1 = self._load_jsonl(ree1_path)
            actual_ree2 = self._load_jsonl(ree2_path)
            actual_ree3 = self._load_jsonl(ree3_path)

            self.assertEqual(actual_ree1, ree1)
            self.assertEqual(actual_ree2, ree2)
            self.assertEqual(actual_ree3, ree3)

    def _load_jsonl(self, file_path):
        """
        Helper method to load a JSON Lines file into a list of dictionaries.

        :param file_path: Path to the JSON Lines file.
        :return: List of dictionaries.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file]

    def _write_jsonl(self, data, file_path):
        """
        Helper method to write a list of dictionaries to a JSON Lines file.

        :param data: List of dictionaries to write.
        :param file_path: Path to the output JSON Lines file.
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            for entry in data:
                json.dump(entry, file, ensure_ascii=False)
                file.write('\n')


if __name__ == "__main__":
    unittest.main()
