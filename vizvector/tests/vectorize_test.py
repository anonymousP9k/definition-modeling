"""Tests for testing vectorize module."""
import json
import unittest
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from vizvector.vectorize import load_json_dataset, save_json_dataset, vectorize_text


class TestVectorizationProcess(unittest.TestCase):
    """Tests for testing vectorize module."""

    def setUp(self):
        self.input_file_path = Path(__file__).parent / "data" / "preds.jsonl"
        self.output_file_path = Path(__file__).parent / "data" / "vectors.jsonl"
        self.expected_file_path = Path(__file__).parent / "data" / "expected_vectors.jsonl"
        self.text_column = "generated_text"
        self.model_name = "cointegrated/rubert-tiny2"

    @pytest.mark.vizvector
    def test_vectorization_process(self):
        """Test functions from vectorize module."""
        data = load_json_dataset(self.input_file_path)

        self.assertEqual(len(data), 5)
        self.assertEqual(data[0]['generated_text'], "значение, влияние кого-, чего-либо")

        vectorized_data = vectorize_text(data, self.text_column, self.model_name)

        save_json_dataset(vectorized_data, self.output_file_path)

        self.assertTrue(self.output_file_path.exists())

        with open(self.output_file_path, 'r', encoding='utf-8') as actual_file:
            actual = [json.loads(line) for line in actual_file]

        with open(self.expected_file_path, "r", encoding="utf-8") as expected_file:
            expected = [json.loads(line) for line in expected_file]

        for actual_item, expected_item in zip(actual, expected):
            for key in actual_item:
                if key != 'vector':
                    self.assertEqual(actual_item[key], expected_item[key])

        similarity_threshold = 0.8
        for actual_item, expected_item in zip(actual, expected):
            actual_vector = np.array(actual_item['vector'])
            expected_vector = np.array(expected_item['vector'])
            similarity = pairwise_distances([actual_vector], [expected_vector],
                                            metric='cosine')[0][0]
            self.assertGreaterEqual(1 - similarity, similarity_threshold)

    def tearDown(self):
        if self.output_file_path.exists():
            self.output_file_path.unlink()
