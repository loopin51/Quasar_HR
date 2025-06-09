import unittest
import os
import tempfile
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg') # Ensure non-interactive backend for tests
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path to allow importing modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tab5_functions import plot_hr_diagram

class TestHRDiagramPlotting(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="astro_test_tab5_")
        self.output_image_path = os.path.join(self.test_dir, "hr_diagram.png")

        # Sample data
        self.sample_magnitudes = np.array([15, 12, 10, 13, 11])
        self.sample_colors = np.array([0.5, 0.2, 0.0, 0.8, 0.3])

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_plot_hr_diagram_successful_creation(self):
        """Test successful creation of an H-R diagram plot."""
        success = plot_hr_diagram(self.sample_magnitudes, self.sample_colors, self.output_image_path)
        self.assertTrue(success, "plot_hr_diagram should return True on success.")
        self.assertTrue(os.path.exists(self.output_image_path), "Output image file should be created.")
        self.assertTrue(os.path.getsize(self.output_image_path) > 0, "Output image file should not be empty.")

    def test_empty_input_magnitudes(self):
        success = plot_hr_diagram([], self.sample_colors, self.output_image_path)
        self.assertFalse(success, "Should return False for empty magnitudes list.")
        self.assertFalse(os.path.exists(self.output_image_path))

    def test_empty_input_colors(self):
        success = plot_hr_diagram(self.sample_magnitudes, [], self.output_image_path)
        self.assertFalse(success, "Should return False for empty colors list.")
        self.assertFalse(os.path.exists(self.output_image_path))

    def test_empty_both_inputs(self):
        success = plot_hr_diagram([], [], self.output_image_path)
        self.assertFalse(success, "Should return False for both input lists empty.")
        self.assertFalse(os.path.exists(self.output_image_path))

    def test_mismatched_input_lengths(self):
        mags = [1, 2, 3]; colors = [0.1, 0.2]
        success = plot_hr_diagram(mags, colors, self.output_image_path)
        self.assertFalse(success, "Should return False for mismatched input lengths.")
        self.assertFalse(os.path.exists(self.output_image_path))

    def test_non_numeric_magnitudes(self):
        mags = [1, 'invalid', 3]; colors = [0.1, 0.2, 0.3]
        success = plot_hr_diagram(mags, colors, self.output_image_path)
        self.assertFalse(success, "Should return False for non-numeric magnitudes.")
        self.assertFalse(os.path.exists(self.output_image_path))

    def test_non_numeric_colors(self):
        mags = [1, 2, 3]; colors = [0.1, 'invalid', 0.3]
        success = plot_hr_diagram(mags, colors, self.output_image_path)
        self.assertFalse(success, "Should return False for non-numeric colors.")
        self.assertFalse(os.path.exists(self.output_image_path))

    def test_output_directory_creation(self):
        nested_output_dir = os.path.join(self.test_dir, "new_subdir")
        nested_output_image_path = os.path.join(nested_output_dir, "hr_in_subdir.png")
        self.assertFalse(os.path.exists(nested_output_dir))
        success = plot_hr_diagram(self.sample_magnitudes, self.sample_colors, nested_output_image_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(nested_output_image_path))
        self.assertTrue(os.path.getsize(nested_output_image_path) > 0)

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_customizations(self, mock_savefig, mock_subplots):
        """Test if plot customizations (title, labels, default color if no colors_data) are applied."""
        mock_fig_instance = MagicMock()
        mock_ax_instance = MagicMock()
        mock_subplots.return_value = (mock_fig_instance, mock_ax_instance)

        custom_title = "My Custom H-R Diagram"
        custom_x_label = "Stellar Color (B-V)"
        custom_y_label = "Apparent Magnitude (V)"

        success = plot_hr_diagram(
            self.sample_magnitudes, self.sample_colors, self.output_image_path,
            title=custom_title,
            x_label=custom_x_label,
            y_label=custom_y_label,
            colors_data=None,
            point_size=25
        )
        self.assertTrue(success, "plot_hr_diagram with customizations should succeed.")

        mock_subplots.assert_called_once_with(figsize=(8, 6))

        mock_ax_instance.scatter.assert_called_once()
        args, kwargs = mock_ax_instance.scatter.call_args
        self.assertEqual(kwargs.get('marker'), 'x')
        self.assertEqual(kwargs.get('s'), 25)
        self.assertEqual(kwargs.get('c'), 'blue')
        self.assertIsNone(kwargs.get('cmap'))

        mock_ax_instance.set_xlabel.assert_called_once_with(custom_x_label)
        mock_ax_instance.set_ylabel.assert_called_once_with(custom_y_label)
        mock_ax_instance.set_title.assert_called_once_with(custom_title)
        mock_ax_instance.invert_yaxis.assert_called_once()
        mock_ax_instance.grid.assert_called_once_with(True, linestyle='--', alpha=0.6)
        mock_savefig.assert_called_once_with(self.output_image_path)

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_with_colormap_and_colorbar(self, mock_savefig, mock_subplots):
        """Test plot generation with colormap based on colors_data and a colorbar."""
        mock_fig_instance = MagicMock()
        mock_ax_instance = MagicMock()
        mock_scatter_return_value = MagicMock()

        mock_subplots.return_value = (mock_fig_instance, mock_ax_instance)
        mock_ax_instance.scatter.return_value = mock_scatter_return_value

        success = plot_hr_diagram(
            magnitudes=self.sample_magnitudes,
            colors=self.sample_colors,
            colors_data=self.sample_colors,
            output_image_path=self.output_image_path,
            title="H-R Diagram with Colormap",
            colormap='viridis'
        )
        self.assertTrue(success, "Plotting with colormap should succeed.")

        mock_subplots.assert_called_once_with(figsize=(8,6))

        mock_ax_instance.scatter.assert_called_once()
        args, kwargs = mock_ax_instance.scatter.call_args
        self.assertTrue(np.array_equal(kwargs.get('c'), self.sample_colors))
        self.assertEqual(kwargs.get('cmap'), 'viridis')
        self.assertEqual(kwargs.get('marker'), 'x')

        mock_fig_instance.colorbar.assert_called_once_with(mock_scatter_return_value, ax=mock_ax_instance, label="B-V Color Index")
        mock_savefig.assert_called_once_with(self.output_image_path)

if __name__ == '__main__':
    unittest.main()
