"""Tests for prepare_row and prepare_dataset functions."""
import unittest
from pathlib import Path

import pytest

from model.src.dataset_processing import load_dataset_split, load_dump, prepare_dataset, prepare_row


class DatasetProcessingTest(unittest.TestCase):
    """
    Tests for functions that process the dataset
    """
    def setUp(self):
        self.tests_dir = Path(__file__).resolve().parent
        self.dataset_example_filepath = self.tests_dir / "data" / "dataset-sample.jsonl"
        self.result_dataset_split_directory = self.tests_dir / "data" / "splits"

        if self.result_dataset_split_directory.exists():
            for file in self.result_dataset_split_directory.iterdir():
                file.unlink()
            self.result_dataset_split_directory.rmdir()

    @pytest.mark.model
    def test_load_dataset(self):
        """
        load_dataset should return a list of dictionaries
        """
        dataset = load_dump(self.dataset_example_filepath)
        self.assertIsInstance(dataset, list)
        self.assertIsInstance(dataset[0], dict)

    @pytest.mark.model
    def test_prepare_row(self):
        """
        Prepare_row should return a dict with the correct keys and values
        """
        row = {"word": "слово", "definition": "значение", "context": "пример"}
        expected = {"word": "слово",
                    "input_text": "<LM>Контекст: \"пример\" Определение слова \"слово\": ",
                    "target_text": "значение"}
        actual = prepare_row(row)
        self.assertEqual(expected, actual)

    @pytest.mark.model
    def test_prepare_dataset_splits(self):
        """
        Prepare_dataset should return a dataset with the correct splits
        """
        prepare_dataset(self.dataset_example_filepath, self.result_dataset_split_directory)
        self.assertTrue(self.result_dataset_split_directory.exists())
        self.assertTrue(Path(self.result_dataset_split_directory / "train.jsonl").exists())
        self.assertTrue(Path(self.result_dataset_split_directory / "test.jsonl").exists())
        self.assertTrue(Path(self.result_dataset_split_directory / "val.jsonl").exists())

    @pytest.mark.model
    def test_load_dataset_split(self):
        """
        Load_dataset_split should return a dataset with the correct splits
        """
        prepare_dataset(self.dataset_example_filepath, self.result_dataset_split_directory)
        dataset_dict = load_dataset_split(
            self.result_dataset_split_directory)
        train_dataset = dataset_dict["train"]
        test_dataset = dataset_dict["test"]
        val_dataset = dataset_dict["val"]
        self.assertEqual(len(train_dataset), 8)
        self.assertEqual(len(test_dataset), 1)
        self.assertEqual(len(val_dataset), 1)

    @pytest.mark.model
    def test_load_dataset_split_content(self):
        """
        Load_dataset_split should return a dataset with the correct content
        """
        prepare_dataset(self.dataset_example_filepath, self.result_dataset_split_directory)
        dataset_dict = load_dataset_split(
            self.result_dataset_split_directory)
        train_dataset = dataset_dict["train"]
        test_dataset = dataset_dict["test"]
        val_dataset = dataset_dict["val"]
        self.assertEqual(train_dataset[0]["input_text"], "<LM>Контекст: \"Пребывание его в Швеции "
                                                         "продолжалось около двенадцати лет.\" "
                                                         "Определение слова \"Швеция\": ")
        self.assertEqual(train_dataset[0]["target_text"], "государство в Северной Европе")
        self.assertEqual(test_dataset[0]["input_text"], "<LM>Контекст: \"Архитектор "
                                                        "Перестройки.\" Определение слова "
                                                        "\"архитектор\": ")
        self.assertEqual(test_dataset[0]["target_text"], "автор, инициатор и исполнитель "
                                                         "какого-либо проекта")
        self.assertEqual(val_dataset[0]["input_text"], "<LM>Контекст: \"У неё были две "
                                                       "короткохвостые шиншиллы: одна эбонитовая, "
                                                       "а другая — бежевая.\" Определение слова "
                                                       "\"эбонитовый\": ")
        self.assertEqual(val_dataset[0]["target_text"], "очень чёрный")

    @pytest.mark.model
    def test_prepare_dataset_with_invalid_filepath(self):
        """
        Prepare_dataset should raise a FileNotFoundError if the file doesn't exist
        """
        with self.assertRaises(FileNotFoundError):
            prepare_dataset("invalid_filepath", self.result_dataset_split_directory)

    def tearDown(self):
        if self.result_dataset_split_directory.exists():
            for file in self.result_dataset_split_directory.iterdir():
                file.unlink()
            self.result_dataset_split_directory.rmdir()
