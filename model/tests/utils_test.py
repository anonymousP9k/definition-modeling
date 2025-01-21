"""Tests for load_config function from utils.py"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model.src.utils import get_current_torch_device, load_train_config, parse_path, TrainConfigDTO


class TestLoadConfig(unittest.TestCase):
    """Tests for load_train_config function from utils.py."""

    def setUp(self):
        self.tests_dir = Path(__file__).parent
        self.correct_config_path = self.tests_dir / "data" / "correct_train_config.json"
        self.incorrect_config_path = self.tests_dir / "data" / "incorrect_train_config.json"
        self.incorrect_values_config_path = (self.tests_dir / "data" /
                                             "incorrect_values_train_config.json")

    @pytest.mark.model
    def test_load_config_correct(self):
        """
        load_train_config should return the correct config.
        """
        actual = load_train_config(self.correct_config_path)
        expected = TrainConfigDTO(model_checkpoint="cointegrated/ruT5-small",
                dataset_split_directory="model/data/splits",
                learning_rate=2e-5,
                batch_size=4,
                num_train_epochs=1)

        self.assertEqual(actual, expected)

    @pytest.mark.model
    def test_load_config_incorrect(self):
        """
        load_train_config should raise a ValueError if the config is incorrect.
        """
        with self.assertRaises(TypeError):
            load_train_config(self.incorrect_config_path)

    @pytest.mark.model
    def test_load_config_incorrect_values(self):
        """
        load_train_config should raise a ValueError if the config contains incorrect values.
        """
        with self.assertRaises(ValueError):
            load_train_config(self.incorrect_values_config_path)


class ParsePathTest(unittest.TestCase):
    """Tests for parse_path function from utils.py."""

    def setUp(self):
        self.tests_dir = Path(__file__).parent

    def test_relative_path(self):
        """parse_path should return the absolute path if the path is relative."""
        actual = parse_path("model/tests")
        expected = self.tests_dir
        self.assertEqual(actual, expected)

    def test_absolute_path(self):
        """parse_path should return the absolute path if the path is absolute."""
        actual = parse_path(self.tests_dir)
        expected = self.tests_dir
        self.assertEqual(actual, expected)


class TestTorchDevice(unittest.TestCase):
    """Tests for get_current_torch_device function from utils.py."""

    def test_get_current_torch_device_cuda(self):
        """get_current_torch_device should return 'cuda' if cuda is available."""
        with patch('torch.cuda.is_available', MagicMock(return_value=True)):
            device = get_current_torch_device()
            self.assertEqual(device, "cuda")

    def test_get_current_torch_device_mps(self):
        """get_current_torch_device should return 'mps' if mps is available."""
        with patch('torch.cuda.is_available', MagicMock(return_value=False)):
            with patch('torch.backends.mps.is_available', MagicMock(return_value=True)):
                device = get_current_torch_device()
                self.assertEqual(device, "mps")

    def test_get_current_torch_device_cpu(self):
        """get_current_torch_device should return 'cpu' if cuda and mps are not available."""
        with patch('torch.cuda.is_available', MagicMock(return_value=False)):
            with patch('torch.backends.mps.is_available', MagicMock(return_value=False)):
                device = get_current_torch_device()
                self.assertEqual(device, "cpu")
