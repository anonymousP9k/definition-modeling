# pylint: disable=no-member
"""Tests for plot_training_and_test_loss function"""
import unittest
from pathlib import Path

import cv2
import numpy as np
import pytest

from model.plot import plot_metric, plot_training_and_test_loss


class PlotTrainingAndTestLossTest(unittest.TestCase):
    """
    Tests for functions that create plots
    """
    def setUp(self):
        self.log_history = [{'loss': 3.6603, 'epoch': 1.08, 'step': 10},
                            {'eval_loss': 3.3250460624694824, 'epoch': 1.08, 'step': 10},
                            {'loss': 3.4554, 'epoch': 2.16, 'step': 20},
                            {'eval_loss': 3.1895878314971924, 'epoch': 2.16, 'step': 20},
                            {'loss': 3.5297, 'epoch': 3.24, 'step': 30},
                            {'eval_loss': 3.0954017639160156, 'epoch': 3.24, 'step': 30},
                            {'loss': 3.2921, 'epoch': 4.32, 'step': 40},
                            {'eval_loss': 3.0169196128845215, 'epoch': 4.32, 'step': 40},
                            {'loss': 3.1995, 'epoch': 5.41, 'step': 50},
                            {'eval_loss': 2.964205026626587,'epoch': 5.41, 'step': 50},
                            {'loss': 3.2171, 'epoch': 6.49, 'step': 60},
                            {'eval_loss': 2.9277913570404053, 'epoch': 6.49, 'step': 60},
                            {'loss': 3.158, 'epoch': 7.57, 'step': 70},
                            {'eval_loss': 2.9022018909454346, 'epoch': 7.57, 'step': 70},
                            {'loss': 3.198, 'epoch': 8.65, 'step': 80},
                            {'eval_loss': 2.8873517513275146, 'epoch': 8.65, 'step': 80},
                            {'loss': 3.153, 'epoch': 9.73, 'step': 90},
                            {'eval_loss': 2.882246255874634, 'epoch': 9.73, 'step': 90}]

        self.metric_log_history = [{"eval_rougeL": 0.81, 'epoch': 1, 'step': 10},
                                   {"eval_rougeL": 0.82, 'epoch': 2, 'step': 20},
                                   {"eval_rougeL": 0.83, 'epoch': 3, 'step': 30},
                                   {"eval_rougeL": 0.84, 'epoch': 4, 'step': 40},
                                   {"eval_rougeL": 0.85, 'epoch': 5, 'step': 50},
                                   {"eval_rougeL": 0.86, 'epoch': 6, 'step': 60},
                                   {"eval_rougeL": 0.87, 'epoch': 7, 'step': 70},
                                   {"eval_rougeL": 0.88, 'epoch': 8, 'step': 80},
                                   {"eval_rougeL": 0.89, 'epoch': 9, 'step': 90},
                                   {"eval_rougeL": 0.90, 'epoch': 10, 'step': 100}]

        self.tests_dir = Path(__file__).resolve().parent
        self.expected_loss_epoch_filepath = (self.tests_dir / "data" /
                                             "expected-loss-plot-epoch.png")
        self.expected_loss_step_filepath = (self.tests_dir / "data" /
                                            "expected-loss-plot-step.png")
        self.test_loss_epoch_filepath = self.tests_dir / "data" / "test-loss-plot-epoch.png"
        self.test_loss_step_filepath = self.tests_dir / "data" / "test-loss-plot-step.png"

        self.expected_metric_epoch_filepath = (self.tests_dir / "data" /
                                            "expected-rougeL-plot-epoch.png")
        self.expected_metric_step_filepath = (self.tests_dir / "data" /
                                            "expected-rougeL-plot-step.png")

        self.test_metric_epoch_filepath = (self.tests_dir / "data" /
                                           "test-rougeL-plot-epoch.png")
        self.test_metric_step_filepath = (self.tests_dir / "data" /
                                          "test-rougeL-plot-step.png")

        if self.test_loss_epoch_filepath.exists():
            self.test_loss_epoch_filepath.unlink()

        if self.test_loss_step_filepath.exists():
            self.test_loss_step_filepath.unlink()

        if self.test_metric_epoch_filepath.exists():
            self.test_metric_epoch_filepath.unlink()

        if self.test_metric_step_filepath.exists():
            self.test_metric_step_filepath.unlink()

    def assert_images_similar(self, img_path1: str | Path, img_path2: str | Path,
                              threshold: int = 10):
        """
        Assert that two images are similar using opencv.
        """
        img1 = cv2.imread(str(img_path1))
        img2 = cv2.imread(str(img_path2))

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(img1_gray, img2_gray)

        num_diff_pixels = np.sum(diff > threshold)  # type: ignore

        if num_diff_pixels > 0:
            raise AssertionError(f"Images {img_path1} and {img_path2} differ at "
                                 f"{num_diff_pixels} pixels")

    @pytest.mark.model
    def test_plot_training_and_test_loss_with_epochs_ideal(self):
        """
        Plot_training_and_test_loss should create a png file with the correct plot
        """
        plot_training_and_test_loss(self.log_history, self.test_loss_epoch_filepath)

        self.assertTrue(self.test_loss_epoch_filepath.exists())

        self.assert_images_similar(self.expected_loss_epoch_filepath, self.test_loss_epoch_filepath)

    @pytest.mark.model
    def test_plot_training_and_test_loss_with_steps_ideal(self):
        """
        Plot_training_and_test_loss should create a png file with the correct plot
        """
        plot_training_and_test_loss(self.log_history, self.test_loss_step_filepath,
                                    plot_epochs=False)

        self.assertTrue(self.test_loss_step_filepath.exists())

        self.assert_images_similar(self.expected_loss_step_filepath, self.test_loss_step_filepath)

    @pytest.mark.model
    def test_plot_metric_with_epochs_ideal(self):
        """
        Plot_metric should create a png file with the correct plot
        """
        plot_metric("eval_rougeL", self.metric_log_history,
                    self.test_metric_epoch_filepath, plot_epochs=True)

        self.assertTrue(self.test_metric_epoch_filepath.exists())

        self.assert_images_similar(self.expected_metric_epoch_filepath,
                                   self.test_metric_epoch_filepath)

    @pytest.mark.model
    def test_plot_metric_with_steps_ideal(self):
        """
        Plot_metric should create a png file with the correct plot
        """
        plot_metric("eval_rougeL", self.metric_log_history,
                    self.test_metric_step_filepath, plot_epochs=False)

        self.assertTrue(self.test_metric_step_filepath.exists())

        self.assert_images_similar(self.expected_metric_step_filepath,
                                   self.test_metric_step_filepath)

    def tearDown(self):
        if self.test_loss_epoch_filepath.exists():
            self.test_loss_epoch_filepath.unlink()

        if self.test_loss_step_filepath.exists():
            self.test_loss_step_filepath.unlink()

        if self.test_metric_epoch_filepath.exists():
            self.test_metric_epoch_filepath.unlink()

        if self.test_metric_step_filepath.exists():
            self.test_metric_step_filepath.unlink()
