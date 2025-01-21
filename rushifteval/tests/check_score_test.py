"""Tests for the calculate_test_score function"""
import unittest
from pathlib import Path

import pytest

from rushifteval.check_score import calculate_test_score, Scores


class CheckScoreTest(unittest.TestCase):
    """Tests for calculate_test_score function"""

    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "data"
        self.predictions_path = self.test_data_dir / "generated_testset.tsv"
        self.gold_scores_path = self.test_data_dir / "gold_testset.tsv"

    @pytest.mark.rushifteval
    def test_calculate_test_score(self):
        """
        Test how calculate_test_score computes the correlations.
        """
        scores = calculate_test_score(self.predictions_path, self.gold_scores_path)

        expected_mean_correlation = 0.62
        expected_correlations = {
            'pre-Soviet:Soviet': 0.57,
            'Soviet:post-Soviet': 0.65,
            'pre-Soviet:post-Soviet': 0.62
        }

        self.assertIsInstance(scores, Scores, "Should return a Scores object")
        self.assertTrue(isinstance(scores.mean_correlation, float),
                        "Mean correlation should be a float")
        self.assertTrue(isinstance(scores.correlations, dict),
                        "Correlations should be a dictionary")
        self.assertEqual(len(scores.correlations), 3,
                         "Should contain three correlation values")

        self.assertGreaterEqual(scores.mean_correlation, -1.0,
                                "Mean correlation should be no less than -1")
        self.assertLessEqual(scores.mean_correlation, 1.0,
                             "Mean correlation should be no greater than 1")

        self.assertAlmostEqual(scores.mean_correlation, expected_mean_correlation, places=2,
                               msg="Mean correlation does not match expected value")

        for key, value in expected_correlations.items():
            self.assertAlmostEqual(scores.correlations[key], value, places=2,
                                   msg=f"Correlation for {key} does not match expected value")

    @pytest.mark.rushifteval
    def test_scores_str(self):
        """
        Test how the Scores object is converted to a string.
        """
        scores = Scores(
            mean_correlation=0.62,
            correlations=
            {
                'pre-Soviet:Soviet': 0.57,
                'Soviet:post-Soviet': 0.65,
                'pre-Soviet:post-Soviet': 0.62
            }
        )

        expected_str = ("Mean correlation: 0.62\n"
                        "Correlations for all epoch pairs:\n"
                        "pre-Soviet:Soviet: 0.57\n"
                        "Soviet:post-Soviet: 0.65\n"
                        "pre-Soviet:post-Soviet: 0.62")

        self.assertEqual(str(scores), expected_str,
                         "String representation does not match expected value")
