"""Tests for rushifteval_utils module."""
import unittest

from rushifteval.rushifteval_utils import compute_distance


class TestComputeDistance(unittest.TestCase):
    """Tests for compute_distance function."""

    def test_compute_distance_with_valid_inputs(self):
        """Test compute_distance function with valid inputs."""
        vect1 = [1.0, 2.0, 3.0]
        vect2 = [4.0, 5.0, 6.0]

        distance = compute_distance(vect1, vect2, 'cosine', False)
        self.assertAlmostEqual(distance, 0.025368153802923787)

        distance = compute_distance(vect1, vect2, 'manhattan', False)
        self.assertAlmostEqual(distance, 9.0)

        distance = compute_distance(vect1, vect2, 'euclidean', False)
        self.assertAlmostEqual(distance, 5.196152422706632)

        distance = compute_distance(vect1, vect2, 'hamming', False)
        self.assertEqual(distance, 1.0)

        distance = compute_distance(vect1, vect2, 'minkowski', False)
        self.assertAlmostEqual(distance, 4.3267487109222245)

        distance = compute_distance(vect1, vect2, 'dot_product', False)
        self.assertAlmostEqual(distance, 32.0)

        distance = compute_distance(vect1, vect2, 'l2-squared', False)
        self.assertAlmostEqual(distance, 27.0)

    def test_compute_distance_with_normalize_flag(self):
        """Test compute_distance function with normalize_flag=True."""
        vect1 = [1.0, 2.0, 3.0]
        vect2 = [4.0, 5.0, 6.0]

        distance = compute_distance(vect1, vect2, 'cosine', True)
        self.assertAlmostEqual(distance, 0.025368153802923565)

    def test_compute_distance_with_invalid_inputs(self):
        """Test compute_distance function with invalid inputs."""
        with self.assertRaises(ValueError):
            compute_distance(None, [1.0, 2.0, 3.0], 'cosine', False)

        with self.assertRaises(ValueError):
            compute_distance([1.0, 2.0, 3.0], None, 'cosine', False)

        with self.assertRaises(ValueError):
            compute_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 'invalid_metric', False)
