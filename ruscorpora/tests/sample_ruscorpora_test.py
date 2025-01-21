"""Tests for the ruscorpora sampling functions."""
import json
import unittest
from pathlib import Path

from ruscorpora.ruscorpora_utils import write_results_to_file
from ruscorpora.sample_ruscorpora import load_data, sample_x_rows_per_word


class RuscorporaTest(unittest.TestCase):
    """Tests for the ruscorpora sampling functions."""

    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "data"
        self.raw_data_path = self.test_data_dir / "raw_data.jsonl"
        self.expected_sampled_data_path = self.test_data_dir / "expected_sampled_data.jsonl"
        self.output_file_path = self.test_data_dir / "sampled_data.jsonl"

    def test_load_data(self):
        """
        Test loading JSON Lines data from a file.
        """
        expected = []
        with open(self.raw_data_path, 'r', encoding='utf-8') as file:
            for line in file:
                expected.append(json.loads(line))
        result = load_data(self.raw_data_path)
        print(result)
        self.assertEqual(result, expected)

    def test_sample_x_rows_per_word(self):
        """
        Test sampling X rows for each unique word in the dataset.
        """

        sample_size = 2
        seed = 42

        data = load_data(self.raw_data_path)
        result = sample_x_rows_per_word(data, sample_size, seed)

        with open(self.expected_sampled_data_path, 'r', encoding='utf-8') as file:
            expected_result = [json.loads(line) for line in file]

        result_sorted = sorted(result, key=lambda x: x['id'])
        expected_sorted = sorted(expected_result, key=lambda x: x['id'])
        self.assertEqual(result_sorted, expected_sorted)

    def test_sample_and_save(self):
        """
        Test sampling X rows for each unique word in the dataset and saving the result.
        """
        sample_size = 2
        seed = 42

        data = load_data(self.raw_data_path)
        result = sample_x_rows_per_word(data, sample_size, seed)

        write_results_to_file(result, self.output_file_path)

        self.assertTrue(self.output_file_path.exists())

        with open(self.output_file_path, 'r', encoding='utf-8') as file:
            actual = [json.loads(line) for line in file]

        result_sorted = sorted(result, key=lambda x: x['id'])
        actual_sorted = sorted(actual, key=lambda x: x['id'])
        self.assertEqual(result_sorted, actual_sorted)

    def tearDown(self):
        if self.output_file_path.exists():
            self.output_file_path.unlink()
