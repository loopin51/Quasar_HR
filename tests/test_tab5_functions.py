import unittest
import os
import tempfile
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg') # Ensure non-interactive backend for tests
import matplotlib.pyplot as plt

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

        # Optional: Check if file is not empty (basic check)
        self.assertTrue(os.path.getsize(self.output_image_path) > 0, "Output image file should not be empty.")

    def test_empty_input_magnitudes(self):
        """Test with empty magnitudes list."""
        success = plot_hr_diagram([], self.sample_colors, self.output_image_path)
        self.assertFalse(success, "Should return False for empty magnitudes list.")
        self.assertFalse(os.path.exists(self.output_image_path), "Image file should not be created.")

    def test_empty_input_colors(self):
        """Test with empty colors list."""
        success = plot_hr_diagram(self.sample_magnitudes, [], self.output_image_path)
        self.assertFalse(success, "Should return False for empty colors list.")
        self.assertFalse(os.path.exists(self.output_image_path), "Image file should not be created.")

    def test_empty_both_inputs(self):
        """Test with both input lists empty."""
        success = plot_hr_diagram([], [], self.output_image_path)
        self.assertFalse(success, "Should return False for both input lists empty.")
        self.assertFalse(os.path.exists(self.output_image_path), "Image file should not be created.")

    def test_mismatched_input_lengths(self):
        """Test with magnitudes and colors lists of different lengths."""
        mags = [1, 2, 3]
        colors = [0.1, 0.2] # Different length
        success = plot_hr_diagram(mags, colors, self.output_image_path)
        self.assertFalse(success, "Should return False for mismatched input lengths.")
        self.assertFalse(os.path.exists(self.output_image_path), "Image file should not be created.")

    def test_non_numeric_magnitudes(self):
        """Test with non-numeric data in magnitudes list."""
        mags = [1, 'invalid', 3]
        colors = [0.1, 0.2, 0.3]
        success = plot_hr_diagram(mags, colors, self.output_image_path)
        self.assertFalse(success, "Should return False for non-numeric magnitudes.")
        self.assertFalse(os.path.exists(self.output_image_path), "Image file should not be created.")

    def test_non_numeric_colors(self):
        """Test with non-numeric data in colors list."""
        mags = [1, 2, 3]
        colors = [0.1, 'invalid', 0.3]
        success = plot_hr_diagram(mags, colors, self.output_image_path)
        self.assertFalse(success, "Should return False for non-numeric colors.")
        self.assertFalse(os.path.exists(self.output_image_path), "Image file should not be created.")

    def test_output_directory_creation(self):
        """Test if the output directory is created if it doesn't exist."""
        nested_output_dir = os.path.join(self.test_dir, "new_subdir")
        nested_output_image_path = os.path.join(nested_output_dir, "hr_in_subdir.png")

        # Ensure the subdirectory does not exist initially
        self.assertFalse(os.path.exists(nested_output_dir))

        success = plot_hr_diagram(self.sample_magnitudes, self.sample_colors, nested_output_image_path)
        self.assertTrue(success, "plot_hr_diagram should succeed even if output subdir needs creation.")
        self.assertTrue(os.path.exists(nested_output_image_path), "Output image file should be created in new subdirectory.")
        self.assertTrue(os.path.getsize(nested_output_image_path) > 0, "Output image file should not be empty.")

    def test_plot_customizations(self):
        """Test if plot customizations (title, labels) are applied (indirectly by checking execution)."""
        custom_title = "My Custom H-R Diagram"
        custom_x_label = "Stellar Color (B-V)"
        custom_y_label = "Apparent Magnitude (V)"

        success = plot_hr_diagram(
            self.sample_magnitudes, self.sample_colors, self.output_image_path,
            title=custom_title,
            x_label=custom_x_label,
            y_label=custom_y_label,
            point_color='red',
            point_size=20
        )
        self.assertTrue(success, "plot_hr_diagram with customizations should succeed.")
        self.assertTrue(os.path.exists(self.output_image_path), "Customized plot image file should be created.")
        self.assertTrue(os.path.getsize(self.output_image_path) > 0)
        # Note: Verifying actual plot content (e.g., if title is set) is complex and
        # usually part of visual regression testing, not standard unit tests.
        # Here, we just ensure the function runs with these parameters.

if __name__ == '__main__':
    unittest.main()
