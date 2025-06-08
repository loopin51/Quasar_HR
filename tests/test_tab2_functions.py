import unittest
import os
import tempfile
import shutil
import numpy as np
from astropy.io import fits

# Add the parent directory to sys.path to allow importing modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tab2_functions import correct_light_frame, get_exposure_time
from utils import load_fits_data, get_fits_header # For verification

class TestLightFrameCorrection(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="astro_test_tab2_")
        self.raw_light_path = os.path.join(self.test_dir, "raw_light.fits")
        self.master_bias_path = os.path.join(self.test_dir, "master_bias.fits")
        self.master_dark_path = os.path.join(self.test_dir, "master_dark.fits")
        self.master_flat_path = os.path.join(self.test_dir, "master_flat.fits")
        self.output_path = os.path.join(self.test_dir, "corrected_light.fits")

        # Default data dimensions
        self.dims = (10, 10)
        self.raw_light_data = np.full(self.dims, 200.0, dtype=np.float32)
        self.master_bias_data = np.full(self.dims, 10.0, dtype=np.float32)
        self.master_dark_data = np.full(self.dims, 20.0, dtype=np.float32) # Exposure 60s for this dark
        self.master_flat_data = np.full(self.dims, 2.0, dtype=np.float32) # Median will be 2.0

        # Create dummy files
        self._create_dummy_fits_file(self.raw_light_path, self.raw_light_data, {'EXPTIME': 30.0, 'INSTRUME': 'TestCam'}) # Light exp time 30s
        self._create_dummy_fits_file(self.master_bias_path, self.master_bias_data)
        self._create_dummy_fits_file(self.master_dark_path, self.master_dark_data, {'EXPTIME': 60.0}) # Dark exp time 60s
        self._create_dummy_fits_file(self.master_flat_path, self.master_flat_data)


    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_dummy_fits_file(self, filepath: str, data: np.ndarray, header_dict: dict = None):
        hdr = fits.Header()
        if header_dict:
            for k, v in header_dict.items():
                hdr[k] = v
        if data.dtype != np.float32: # Ensure float32 for consistency
            data = data.astype(np.float32)
        fits.writeto(filepath, data, header=hdr, overwrite=True)

    def test_get_exposure_time(self):
        hdr1 = fits.Header({'EXPTIME': 10.0})
        self.assertEqual(get_exposure_time(hdr1), 10.0)
        hdr2 = fits.Header({'EXPOSURE': 20.0})
        self.assertEqual(get_exposure_time(hdr2), 20.0)
        hdr3 = fits.Header({'OTHERKEY': 30.0})
        self.assertIsNone(get_exposure_time(hdr3))
        hdr4 = fits.Header({'EXPTIME': 'invalid'})
        self.assertIsNone(get_exposure_time(hdr4))
        self.assertIsNone(get_exposure_time(None))


    def test_full_correction(self):
        success = correct_light_frame(self.raw_light_path, self.output_path,
                                      master_bias_path=self.master_bias_path,
                                      master_dark_path=self.master_dark_path,
                                      master_flat_path=self.master_flat_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.output_path))

        corrected_data = load_fits_data(self.output_path)
        self.assertIsNotNone(corrected_data)

        # Expected: (Raw - Bias - ScaledDark) / NormalizedFlat
        # Raw = 200, Bias = 10
        # Dark = 20 (for 60s), LightExp = 30s. ScaledDark = 20 * (30/60) = 10
        # Flat = 2, NormalizedFlat = Flat / median(Flat) = 2 / 2 = 1
        # Expected_pixel_value = (200 - 10 - 10) / 1 = 180
        expected_data = np.full(self.dims, 180.0, dtype=np.float32)
        self.assertTrue(np.allclose(corrected_data, expected_data))

        header = get_fits_header(self.output_path)
        self.assertIn("CAL_BIAS", header)
        self.assertIn("CAL_DARK", header)
        self.assertIn("CAL_FLAT", header)
        self.assertIn("Subtracted master BIAS", " ".join(header.get('HISTORY', [])))


    def test_bias_only_correction(self):
        success = correct_light_frame(self.raw_light_path, self.output_path, master_bias_path=self.master_bias_path)
        self.assertTrue(success)
        corrected_data = load_fits_data(self.output_path)
        # Expected: Raw - Bias = 200 - 10 = 190
        expected_data = np.full(self.dims, 190.0, dtype=np.float32)
        self.assertTrue(np.allclose(corrected_data, expected_data))
        header = get_fits_header(self.output_path)
        self.assertIn("CAL_BIAS", header)
        self.assertNotIn("CAL_DARK", header)
        self.assertNotIn("CAL_FLAT", header)

    def test_dark_only_correction_scaled(self):
        success = correct_light_frame(self.raw_light_path, self.output_path, master_dark_path=self.master_dark_path)
        self.assertTrue(success)
        corrected_data = load_fits_data(self.output_path)
        # Expected: Raw - ScaledDark = 200 - (20 * 30/60) = 200 - 10 = 190
        expected_data = np.full(self.dims, 190.0, dtype=np.float32)
        self.assertTrue(np.allclose(corrected_data, expected_data))
        header = get_fits_header(self.output_path)
        self.assertIn("CAL_DARK", header)
        self.assertTrue(any("Applied scaled master DARK" in entry for entry in header.get('HISTORY', []))) # Updated expected string


    def test_dark_correction_unscaled(self):
        # Modify master dark to have same exposure time as light
        self._create_dummy_fits_file(self.master_dark_path, self.master_dark_data, {'EXPTIME': 30.0}) # Dark exp = 30s
        success = correct_light_frame(self.raw_light_path, self.output_path, master_dark_path=self.master_dark_path)
        self.assertTrue(success)
        corrected_data = load_fits_data(self.output_path)
        # Expected: Raw - Dark = 200 - 20 = 180 (since exptime matches, scale=1)
        expected_data = np.full(self.dims, 180.0, dtype=np.float32)
        self.assertTrue(np.allclose(corrected_data, expected_data))
        header = get_fits_header(self.output_path)
        self.assertIn("CAL_DARK", header)
        self.assertTrue(any("Applied master DARK (unscaled, exp times equal)" in entry for entry in header.get('HISTORY', []))) # Updated expected string


    def test_flat_only_correction(self):
        success = correct_light_frame(self.raw_light_path, self.output_path, master_flat_path=self.master_flat_path)
        self.assertTrue(success)
        corrected_data = load_fits_data(self.output_path)
        # Expected: Raw / NormalizedFlat = 200 / (2/2) = 200 / 1 = 200
        expected_data = np.full(self.dims, 200.0, dtype=np.float32)
        self.assertTrue(np.allclose(corrected_data, expected_data))
        header = get_fits_header(self.output_path)
        self.assertIn("CAL_FLAT", header)

    def test_no_calibration_files(self):
        success = correct_light_frame(self.raw_light_path, self.output_path)
        self.assertTrue(success)
        corrected_data = load_fits_data(self.output_path)
        # Expected: Raw data (converted to float32, which it already is in test setup)
        self.assertTrue(np.allclose(corrected_data, self.raw_light_data))
        header = get_fits_header(self.output_path)
        self.assertNotIn("CAL_BIAS", header)
        self.assertNotIn("CAL_DARK", header)
        self.assertNotIn("CAL_FLAT", header)
        self.assertTrue(any("LIGHT frame correction started" in entry for entry in header.get('HISTORY',[])))


    def test_missing_master_file(self):
        # Test with a non-existent bias file, but valid dark and flat
        non_existent_bias = os.path.join(self.test_dir, "non_existent_bias.fits")
        success = correct_light_frame(self.raw_light_path, self.output_path,
                                      master_bias_path=non_existent_bias,
                                      master_dark_path=self.master_dark_path,
                                      master_flat_path=self.master_flat_path)
        self.assertTrue(success) # Should still succeed, just skip the missing step
        corrected_data = load_fits_data(self.output_path)
        # Expected: (Raw - ScaledDark) / NormalizedFlat = (200 - 10) / 1 = 190
        expected_data = np.full(self.dims, 190.0, dtype=np.float32)
        self.assertTrue(np.allclose(corrected_data, expected_data))
        header = get_fits_header(self.output_path)
        self.assertNotIn("CAL_BIAS", header) # Bias was skipped
        self.assertIn("CAL_DARK", header)
        self.assertIn("CAL_FLAT", header)
        self.assertTrue(any("BIAS subtraction skipped: Failed to load" in entry for entry in header.get('HISTORY',[])))

    def test_shape_mismatch_master_bias(self):
        # Create a bias with different shape
        mismatched_bias_data = np.full((5,5), 5.0, dtype=np.float32)
        self._create_dummy_fits_file(self.master_bias_path, mismatched_bias_data)
        success = correct_light_frame(self.raw_light_path, self.output_path, master_bias_path=self.master_bias_path)
        self.assertTrue(success) # Skips the step
        header = get_fits_header(self.output_path)
        self.assertTrue(any("BIAS subtraction skipped: Shape mismatch" in entry for entry in header.get('HISTORY',[])))
        # Data should be same as raw, as only bias was provided and it was skipped
        corrected_data = load_fits_data(self.output_path)
        self.assertTrue(np.allclose(corrected_data, self.raw_light_data))


    def test_dark_invalid_exptime(self):
        # Master dark with EXPTIME = 0
        self._create_dummy_fits_file(self.master_dark_path, self.master_dark_data, {'EXPTIME': 0.0})
        success = correct_light_frame(self.raw_light_path, self.output_path, master_dark_path=self.master_dark_path)
        self.assertTrue(success) # Skips the step
        header = get_fits_header(self.output_path)
        self.assertTrue(any("DARK subtraction skipped: Invalid exposure time" in entry for entry in header.get('HISTORY',[])))
        corrected_data = load_fits_data(self.output_path)
        self.assertTrue(np.allclose(corrected_data, self.raw_light_data)) # Dark subtraction skipped

    def test_flat_median_zero(self):
        # Master flat with all zeros (median will be zero)
        zero_flat_data = np.zeros(self.dims, dtype=np.float32)
        self._create_dummy_fits_file(self.master_flat_path, zero_flat_data)
        success = correct_light_frame(self.raw_light_path, self.output_path, master_flat_path=self.master_flat_path)
        self.assertTrue(success) # Skips the step
        header = get_fits_header(self.output_path)
        self.assertTrue(any("FLAT fielding skipped: Median of" in entry for entry in header.get('HISTORY',[])))
        corrected_data = load_fits_data(self.output_path)
        self.assertTrue(np.allclose(corrected_data, self.raw_light_data)) # Flat fielding skipped

if __name__ == '__main__':
    unittest.main()
