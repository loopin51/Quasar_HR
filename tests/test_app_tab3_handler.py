import unittest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
from astropy.io import fits
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path to allow importing modules from there
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import handle_tab3_extinction_from_fits # The handler from app.py
from utils import save_fits_data # For creating test files
from app import _try_remove # Import the helper function

class BaseTestTab3ExtinctionHandler(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="tab3_handler_test_")

        # Mock Gradio request object
        self.mock_request = MagicMock()
        self.mock_request.client.host = "127.0.0.1"
        self.mock_request.client.port = 7860

        # Mock master frame paths (will be created in test methods or a more specific setUp)
        self.master_bias_path = os.path.join(self.test_dir, "master_bias.fits")
        save_fits_data(self.master_bias_path, np.full((10, 10), 10.0, dtype=np.float32), header=fits.Header())

        # Darks for flat processing and light processing
        self.dark_60s_path = os.path.join(self.test_dir, "master_dark_60p0s.fits")
        dark_header_60s = fits.Header({'EXPTIME': 60.0})
        save_fits_data(self.dark_60s_path, np.full((10, 10), 5.0, dtype=np.float32), header=dark_header_60s)

        self.dark_120s_path = os.path.join(self.test_dir, "master_dark_120p0s.fits")
        dark_header_120s = fits.Header({'EXPTIME': 120.0})
        save_fits_data(self.dark_120s_path, np.full((10, 10), 10.0, dtype=np.float32), header=dark_header_120s)

        self.master_dark_paths_state = {
            "60p0s": self.dark_60s_path,
            "120p0s": self.dark_120s_path,
        }

        # Flat for filter 'V' (assume its own exposure is 5s for dark scaling test)
        self.prelim_flat_V_path = os.path.join(self.test_dir, "prelim_flat_V.fits")
        flat_header_V = fits.Header({'FILTER': 'V', 'EXPTIME': 5.0}) # Flat's own exposure
        save_fits_data(self.prelim_flat_V_path, np.full((10, 10), 15000.0, dtype=np.float32), header=flat_header_V)

        self.master_flat_paths_state = {
            "V": self.prelim_flat_V_path
        }

        self.light_file_objs = []

    def _create_dummy_light_fits(self, filename_suffix: str, airmass: float, filter_name: str = 'V', exptime: float = 120.0, star_x=5, star_y=5, star_peak=20000.0, shape=(10,10)):
        """Helper to create a dummy LIGHT FITS file."""
        filepath = os.path.join(self.test_dir, f"light_{filename_suffix}.fits")
        header_dict = {'AIRMASS': airmass, 'FILTER': filter_name, 'EXPTIME': exptime, 'OBJECT': f'TestStar_{filename_suffix}'}

        data = np.full(shape, 100.0, dtype=np.float32) # Background
        if star_peak > 0: # Add a simple star if peak is positive
            data[star_y-1:star_y+2, star_x-1:star_x+2] += star_peak # 3x3 star

        save_fits_data(filepath, data, header=fits.Header(header_dict))

        # Mock a Gradio FileData object
        file_obj_mock = MagicMock()
        file_obj_mock.name = filepath
        return file_obj_mock

    def tearDown(self):
        if hasattr(self, 'light_file_objs'):
            for file_obj in self.light_file_objs:
                if file_obj and hasattr(file_obj, 'name') and os.path.exists(file_obj.name):
                    _try_remove(file_obj.name) # Use the helper

        # Clean up master files created in setUp
        _try_remove(self.master_bias_path)
        _try_remove(self.dark_60s_path)
        _try_remove(self.dark_120s_path)
        _try_remove(self.prelim_flat_V_path)

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

        # Clean up any plot files that might have been created in default paths
        # This depends on where handle_tab3_extinction_from_fits saves plots
        plot_save_dir = os.path.join("calibrated_lights_output", "previews", "tab3_plots")
        if os.path.exists(plot_save_dir):
            for item in os.listdir(plot_save_dir):
                if item.startswith("tab3_extinction_plot_") and item.endswith(".png"):
                    _try_remove(os.path.join(plot_save_dir, item))
            _try_rmdir_if_empty(plot_save_dir)


class TestTab3ExtinctionHandler(BaseTestTab3ExtinctionHandler):

    def test_successful_extinction_calculation(self):
        """Test the full pipeline with valid FITS files producing k and m0."""
        self.light_file_objs = [
            self._create_dummy_light_fits("file1", airmass=1.0, exptime=120.0, star_peak=30000),
            self._create_dummy_light_fits("file2", airmass=1.5, exptime=120.0, star_peak=20000),
            self._create_dummy_light_fits("file3", airmass=2.0, exptime=120.0, star_peak=10000),
        ]

        status, k, k_err, m0, plot_path, df_results = handle_tab3_extinction_from_fits(
            self.light_file_objs,
            self.master_bias_path,
            self.master_dark_paths_state,
            self.master_flat_paths_state,
            self.mock_request
        )

        print(f"Tab 3 Success Test Status: {status}")
        print(f"Tab 3 Results DF:\n{df_results}")

        self.assertTrue("Linear Regression: k =" in status, "Status should indicate successful regression.")
        self.assertTrue(len(k) > 0 and float(k) != 0, "k value should be calculated.")
        self.assertTrue(len(k_err) > 0, "k_err should be calculated.")
        self.assertTrue(len(m0) > 0, "m0 value should be calculated.")
        self.assertIsNotNone(plot_path, "A plot path should be returned.")
        self.assertTrue(os.path.exists(plot_path), f"Plot file should be created at {plot_path}")
        self.assertIsInstance(df_results, pd.DataFrame, "Results should be a Pandas DataFrame.")
        self.assertEqual(len(df_results), 3, "DataFrame should have 3 rows for 3 input files.")
        self.assertFalse(df_results["Instrumental_Magnitude"].isnull().any(), "No null magnitudes expected for valid files.")
        self.assertTrue((df_results["Skipped_Reason"] == "").all(), "No files should be skipped.")

    def test_insufficient_data_points_for_regression(self):
        """Test with only one valid FITS file, insufficient for regression."""
        self.light_file_objs = [
            self._create_dummy_light_fits("file1", airmass=1.0, exptime=120.0, star_peak=30000),
        ]
        status, k, k_err, m0, plot_path, df_results = handle_tab3_extinction_from_fits(
            self.light_file_objs, self.master_bias_path, self.master_dark_paths_state, self.master_flat_paths_state, self.mock_request
        )
        print(f"Tab 3 Insufficient Data Test Status: {status}")
        self.assertTrue("ERROR: Less than 2 data points collected." in status)
        self.assertEqual(k, "")
        self.assertEqual(m0, "")
        self.assertIsNone(plot_path) # No plot if regression fails
        self.assertEqual(len(df_results[df_results["Instrumental_Magnitude"].notna()]), 1, "Should have one valid data point before regression attempt.")

    def test_file_missing_airmass_keyword(self):
        """Test one file missing AIRMASS keyword."""
        self.light_file_objs = [
            self._create_dummy_light_fits("file1_ok", airmass=1.0, exptime=120.0, star_peak=30000),
            self._create_dummy_light_fits("file2_no_airmass", airmass=None, exptime=120.0, star_peak=20000), # Airmass will be removed by helper logic
            self._create_dummy_light_fits("file3_ok", airmass=2.0, exptime=120.0, star_peak=10000),
        ]
        # Modify file2 to remove AIRMASS after creation
        fits_path_no_airmass = self.light_file_objs[1].name
        hdul = fits.open(fits_path_no_airmass, mode='update')
        if 'AIRMASS' in hdul[0].header:
            del hdul[0].header['AIRMASS']
        hdul.flush()
        hdul.close()

        status, k, k_err, m0, plot_path, df_results = handle_tab3_extinction_from_fits(
            self.light_file_objs, self.master_bias_path, self.master_dark_paths_state, self.master_flat_paths_state, self.mock_request
        )
        print(f"Tab 3 Missing Airmass Test Status: {status}")
        print(f"Tab 3 Missing Airmass DF:\n{df_results}")
        self.assertTrue("AIRMASS keyword not found" in status)
        self.assertTrue(len(k) > 0, "k should still be calculated from 2 valid points.") # Regression should proceed with 2 points
        self.assertIsNotNone(plot_path)
        self.assertEqual(len(df_results), 3)
        self.assertTrue(df_results.loc[df_results['Filename'] == 'light_file2_no_airmass_calibrated', 'Skipped_Reason'].iloc[0] == "AIRMASS not found")
        self.assertTrue(df_results.loc[df_results['Filename'] == 'light_file2_no_airmass_calibrated', 'Instrumental_Magnitude'].isnull().all())


    def test_no_stars_detected_in_one_file(self):
        """Test one file where no stars are detected."""
        self.light_file_objs = [
            self._create_dummy_light_fits("file1_ok", airmass=1.0, exptime=120.0, star_peak=30000),
            self._create_dummy_light_fits("file2_no_stars", airmass=1.5, exptime=120.0, star_peak=0), # No star
            self._create_dummy_light_fits("file3_ok", airmass=2.0, exptime=120.0, star_peak=10000),
        ]
        status, k, k_err, m0, plot_path, df_results = handle_tab3_extinction_from_fits(
            self.light_file_objs, self.master_bias_path, self.master_dark_paths_state, self.master_flat_paths_state, self.mock_request
        )
        print(f"Tab 3 No Stars Test Status: {status}")
        self.assertTrue("No sources found. Skipping." in status)
        self.assertTrue(len(k) > 0) # Regression should proceed with 2 points
        self.assertEqual(df_results.loc[df_results['Filename'] == 'light_file2_no_stars_calibrated', 'Skipped_Reason'].iloc[0], "No sources detected")

    def test_calibration_failure_missing_flat(self):
        """Test a file that fails calibration due to a missing flat."""
        self.light_file_objs = [
            self._create_dummy_light_fits("file1_ok_V", airmass=1.0, filter_name='V', exptime=120.0, star_peak=30000),
            self._create_dummy_light_fits("file2_bad_filter_R", airmass=1.5, filter_name='R', exptime=120.0, star_peak=20000), # No R flat in mock state
            self._create_dummy_light_fits("file3_ok_V", airmass=2.0, filter_name='V', exptime=120.0, star_peak=10000),
        ]
        status, k, k_err, m0, plot_path, df_results = handle_tab3_extinction_from_fits(
            self.light_file_objs, self.master_bias_path, self.master_dark_paths_state, self.master_flat_paths_state, self.mock_request
        )
        print(f"Tab 3 Missing Flat Test Status: {status}")
        # The calibration part in handler might not explicitly state "Calibration failed" if flat is just skipped.
        # The check is that the file is skipped and no mag is derived.
        self.assertTrue(len(k) > 0) # Regression should proceed with 2 points
        self.assertEqual(df_results.loc[df_results['Filename'] == 'light_file2_bad_filter_R_calibrated', 'Skipped_Reason'].iloc[0], "Calibration failed")


if __name__ == '__main__':
    unittest.main()
