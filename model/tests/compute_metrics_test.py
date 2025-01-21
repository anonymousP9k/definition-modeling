"""Tests for functions from metrics.py"""
import unittest

import pytest

from model.src.metrics import get_bleu_score, get_rouge_score


class TestComputeMetrics(unittest.TestCase):
    """
    Tests for functions from metrics.py
    """
    def setUp(self):
        self.decoded_labels = ["Ася лежит на кровати, греется на солнышке и вылизывается.",
                               "Ася забралась в коробку и спит."]
        self.decoded_predictions = ["Ася лежит на подоконнике, греется на солнышке и вылизывается.",
                                    "Ася залезла в коробку и спит."]

    @pytest.mark.model
    def test_get_bleu_score_ideal(self):
        """
        get_bleu_score should return the correct BLEU score
        """
        actual = get_bleu_score(self.decoded_predictions, self.decoded_labels)
        expected = 0.68

        self.assertAlmostEqual(actual / 100, expected, places=2)

    @pytest.mark.model
    def test_get_rouge_score_ideal(self):
        """
        get_rouge_score should return the correct ROUGE score
        """
        actual = get_rouge_score(self.decoded_predictions, self.decoded_labels)
        expected = 0.88

        self.assertAlmostEqual(actual, expected, places=2)
