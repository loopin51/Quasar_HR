import unittest
import os
import tempfile
import shutil
import numpy as np
from astropy.io import fits

# Add the parent directory to sys.path to allow importing modules from there
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tab1_functions import create_master_frame
from utils import load_fits_data, get_fits_header # For verification

class TestMasterFrameCreation(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory to store test FITS files."""
        self.test_dir = tempfile.mkdtemp(prefix="astro_test_tab1_")
        self.input_files = []
        self.output_path = os.path.join(self.test_dir, "master_frame.fits")

    def tearDown(self):
        """Remove the temporary directory and its contents."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_dummy_fits_file(self, filename_suffix: str, data: np.ndarray, header_dict: dict = None, dtype=np.float32) -> str:
        """Helper to create a dummy FITS file in the test directory."""
        filepath = os.path.join(self.test_dir, f"dummy_{filename_suffix}.fits") # Changed prefix to avoid conflict with test name
        hdr = fits.Header()
        if header_dict:
            for k, v in header_dict.items():
                hdr[k] = v

        # Ensure data is of the specified dtype
        if data.dtype != dtype:
            data = data.astype(dtype)

        fits.writeto(filepath, data, header=hdr, overwrite=True)
        # self.input_files.append(filepath) # Test methods will manage their own file lists
        return filepath

    def test_create_master_bias_median(self):
        """Test creating a master BIAS using median combination."""
        data1 = np.full((10, 10), 100, dtype=np.float32)
        data2 = np.full((10, 10), 102, dtype=np.float32)
        data3 = np.full((10, 10), 104, dtype=np.float32)
        self.input_files.append(self._create_dummy_fits_file("bias1", data1))
        self.input_files.append(self._create_dummy_fits_file("bias2", data2))
        self.input_files.append(self._create_dummy_fits_file("bias3", data3))

        success = create_master_frame(self.input_files, self.output_path, method="median", frame_type="BIAS")
        self.assertTrue(success, "create_master_frame should succeed.")
        self.assertTrue(os.path.exists(self.output_path), "Output master BIAS file should be created.")

        master_data = load_fits_data(self.output_path)
        master_header = get_fits_header(self.output_path)

        self.assertIsNotNone(master_data)
        expected_median_data = np.median(np.stack([data1, data2, data3], axis=0), axis=0)
        self.assertTrue(np.array_equal(master_data, expected_median_data), "Master BIAS data is not the median.")

        self.assertIsNotNone(master_header)
        self.assertEqual(master_header.get("NCOMBINE"), 3)
        self.assertEqual(master_header.get("COMBTYPE"), "MEDIAN")
        self.assertEqual(master_header.get("FRAMTYPE"), "BIAS")
        # Check if any history line contains the expected creation message part
        history_lines = master_header.get("HISTORY", [])
        self.assertTrue(any("Master frame" in line and "created on" in line for line in history_lines),
                        "Expected creation timestamp not found in FITS header HISTORY.")


    def test_create_master_dark_mean(self):
        """Test creating a master DARK using mean combination."""
        data1 = np.full((5, 5), 10.0, dtype=np.float32)
        data2 = np.full((5, 5), 20.0, dtype=np.float32)
        self.input_files.append(self._create_dummy_fits_file("dark1", data1))
        self.input_files.append(self._create_dummy_fits_file("dark2", data2))

        success = create_master_frame(self.input_files, self.output_path, method="mean", frame_type="DARK")
        self.assertTrue(success)

        master_data = load_fits_data(self.output_path)
        expected_mean_data = np.mean(np.stack([data1, data2], axis=0), axis=0)
        self.assertTrue(np.allclose(master_data, expected_mean_data), "Master DARK data is not the mean.") # Use allclose for float

        master_header = get_fits_header(self.output_path)
        self.assertEqual(master_header.get("NCOMBINE"), 2)
        self.assertEqual(master_header.get("COMBTYPE"), "MEAN")
        self.assertEqual(master_header.get("FRAMTYPE"), "DARK")

    def test_create_master_bias_average(self):
        """Test creating a master BIAS using 'average' (mean) combination."""
        data1 = np.full((7, 7), 50.0, dtype=np.float32)
        data2 = np.full((7, 7), 60.0, dtype=np.float32)
        data3 = np.full((7, 7), 70.0, dtype=np.float32)
        self.input_files.append(self._create_dummy_fits_file("bias_avg1", data1))
        self.input_files.append(self._create_dummy_fits_file("bias_avg2", data2))
        self.input_files.append(self._create_dummy_fits_file("bias_avg3", data3))

        success = create_master_frame(self.input_files, self.output_path, method="average", frame_type="BIAS_AVG")
        self.assertTrue(success, "create_master_frame should succeed with 'average' method.")
        self.assertTrue(os.path.exists(self.output_path), "Output master BIAS_AVG file should be created.")

        master_data = load_fits_data(self.output_path)
        self.assertIsNotNone(master_data)

        expected_average_data = np.mean(np.stack([data1, data2, data3], axis=0), axis=0)
        self.assertTrue(np.allclose(master_data, expected_average_data), "Master BIAS_AVG data is not the average of inputs.")

        master_header = get_fits_header(self.output_path)
        self.assertIsNotNone(master_header)
        self.assertEqual(master_header.get("NCOMBINE"), 3)
        self.assertEqual(master_header.get("COMBTYPE"), "AVERAGE") # Check if method string is stored as uppercase
        self.assertEqual(master_header.get("FRAMTYPE"), "BIAS_AVG")

    def test_create_master_flat_sigma_clip_mean(self):
        """Test creating a master FLAT using sigma_clip_mean combination."""
        # Create data with an outlier
        data1 = np.ones((5, 5), dtype=np.float32) * 1.0
        data2 = np.ones((5, 5), dtype=np.float32) * 1.05
        data3 = np.ones((5, 5), dtype=np.float32) * 0.95
        data_with_outlier = np.ones((5, 5), dtype=np.float32) * 1.0
        data_with_outlier[2,2] = 10.0 # Outlier

        self.input_files.append(self._create_dummy_fits_file("flat1", data1))
        self.input_files.append(self._create_dummy_fits_file("flat2", data2))
        self.input_files.append(self._create_dummy_fits_file("flat3", data3))
        self.input_files.append(self._create_dummy_fits_file("flat_outlier", data_with_outlier))

        success = create_master_frame(self.input_files, self.output_path, method="sigma_clip_mean", frame_type="FLAT")
        self.assertTrue(success)

        master_data = load_fits_data(self.output_path)
        self.assertIsNotNone(master_data)

        # For sigma_clip_mean, the outlier should be excluded/downweighted.
        # The mean of data1, data2, data3 is (1.0 + 1.05 + 0.95) / 3 = 1.0.
        # The outlier at [2,2] (10.0) should be clipped.
        # So, master_data[2,2] should be close to 1.0, not (1.0+1.05+0.95+10.0)/4 = 3.25
        self.assertAlmostEqual(master_data[2,2], 1.0, delta=0.1, msg="Sigma clipping did not effectively remove outlier.")
        # Check a non-outlier pixel
        self.assertAlmostEqual(master_data[0,0], (1.0 + 1.05 + 0.95 + 1.0) / 4, delta=0.1,
                               msg="Sigma clipping mean calculation seems off for non-outlier pixels if outlier was the only one clipped.")
        # More precise check for sigma_clip_mean:
        from astropy.stats import sigma_clipped_stats
        stacked_data = np.stack([data1, data2, data3, data_with_outlier], axis=0)
        # Ensure the test calculates the reference mean_clipped using the same robust stdfunc
        mean_clipped, _, _ = sigma_clipped_stats(stacked_data, axis=0, sigma=3.0, stdfunc='mad_std')
        self.assertTrue(np.allclose(master_data, mean_clipped))


        master_header = get_fits_header(self.output_path)
        self.assertEqual(master_header.get("NCOMBINE"), 4)
        self.assertEqual(master_header.get("COMBTYPE"), "SIGMA_CLIP_MEAN")
        self.assertEqual(master_header.get("FRAMTYPE"), "FLAT")


    def test_no_input_files(self):
        """Test behavior when no input files are provided."""
        success = create_master_frame([], self.output_path, method="median", frame_type="BIAS")
        self.assertFalse(success, "Should return False if no input files.")
        self.assertFalse(os.path.exists(self.output_path), "Output file should not be created.")

    def test_one_input_file_not_found(self):
        """Test when one of several input files is not found."""
        data1 = np.full((10, 10), 100, dtype=np.float32)
        # self.input_files is not used by this test method directly after helper change
        valid_file_path = self._create_dummy_fits_file("bias_ok_single", data1, dtype=np.float32)
        non_existent_file = os.path.join(self.test_dir, "non_existent_single.fits")

        test_files_list = [valid_file_path, non_existent_file]

        success = create_master_frame(test_files_list, self.output_path, method="median", frame_type="BIAS")
        self.assertTrue(success, "Should succeed if at least one file is valid.")
        self.assertTrue(os.path.exists(self.output_path))
        master_data = load_fits_data(self.output_path)
        self.assertTrue(np.array_equal(master_data, data1), "Master data should be from the one valid file.")
        master_header = get_fits_header(self.output_path)
        self.assertEqual(master_header.get("NCOMBINE"), 1)


    def test_all_input_files_not_found(self):
        """Test when all input files are not found."""
        non_existent_file1 = os.path.join(self.test_dir, "non_existent1.fits")
        non_existent_file2 = os.path.join(self.test_dir, "non_existent2.fits")
        success = create_master_frame([non_existent_file1, non_existent_file2], self.output_path, method="median", frame_type="BIAS")
        self.assertFalse(success, "Should return False if all input files are invalid/not found.")
        self.assertFalse(os.path.exists(self.output_path))

    def test_mismatched_dimensions(self):
        """Test combining images with different dimensions."""
        data1 = np.full((10, 10), 100, dtype=np.float32)
        data2_diff_shape = np.full((5, 5), 100, dtype=np.float32) # Different shape
        current_files = [
            self._create_dummy_fits_file("bias_shape1", data1),
            self._create_dummy_fits_file("bias_shape2", data2_diff_shape)
        ]
        # This test expects failure because create_master_frame will skip the second file
        # and then fail because only one valid file (or zero if the first was bad) would remain.
        # However, if only one file remains, it should actually succeed.
        # The logic is: if *no* valid files are loaded, it fails. If one is, it uses that one.
        # If multiple *valid* but differently shaped files are given, it uses the shape of the *first* valid one
        # and skips subsequent non-matching ones.

        success = create_master_frame(current_files, self.output_path, method="median", frame_type="BIAS")

        # If data1 is processed, and data2_diff_shape is skipped, NCOMBINE should be 1.
        # The test should check that it *succeeds* but NCOMBINE is 1.
        self.assertTrue(success, "Should succeed, processing only the first validly shaped file.")
        self.assertTrue(os.path.exists(self.output_path), "Output file should be created.")
        master_header = get_fits_header(self.output_path)
        self.assertEqual(master_header.get("NCOMBINE"), 1, "NCOMBINE should be 1 as only the first file is used.")
        master_data = load_fits_data(self.output_path)
        self.assertTrue(np.array_equal(master_data, data1))


    def test_unknown_method(self):
        """Test using an unsupported combination method."""
        file_paths = [self._create_dummy_fits_file("bias_method_test", np.full((10, 10), 100, dtype=np.float32))]
        success = create_master_frame(file_paths, self.output_path, method="unsupported_method", frame_type="BIAS")
        self.assertFalse(success, "Should return False for an unknown combination method.")
        self.assertFalse(os.path.exists(self.output_path))

    def test_unknown_method_actually_unknown(self): # Name is a bit redundant now
        """Test that a truly unsupported method still fails after adding 'average'."""
        file_paths = [self._create_dummy_fits_file("bias_unknown_method", np.full((10, 10), 100, dtype=np.float32))]
        success = create_master_frame(file_paths, self.output_path, method="nonexistent_method", frame_type="BIAS")
        self.assertFalse(success, "Should return False for a truly non-existent combination method.")
        self.assertFalse(os.path.exists(self.output_path))

    def test_create_master_frame_average_multiple_files(self):
        """Test 'average' method with multiple files, type preservation, and shape mismatch handling."""
        num_valid_files = 5 # Test with a number of files that doesn't require complex batching logic
        width, height = 10, 10

        valid_file_paths = []
        all_valid_data_arrays = []

        for i in range(num_valid_files):
            file_data = np.full((height, width), float(i * 10), dtype=np.float32) # Use float for easier averaging
            file_path = self._create_dummy_fits_file(f"avg_img_{i}", file_data, dtype=np.float32)
            valid_file_paths.append(file_path)
            all_valid_data_arrays.append(file_data)

        # 1. Test successful average processing with multiple valid files
        output_path_avg = os.path.join(self.test_dir, "master_avg_multiple.fits")
        success = create_master_frame(valid_file_paths, output_path_avg, method="average", frame_type="TESTAVG_MULTI")

        self.assertTrue(success, "Average processing with multiple files should succeed.")
        self.assertTrue(os.path.exists(output_path_avg), "Output file for average multiple should exist.")

        master_data_avg = load_fits_data(output_path_avg)
        master_header_avg = get_fits_header(output_path_avg)

        self.assertIsNotNone(master_data_avg, "Loaded master data from average multiple should not be None.")
        self.assertEqual(master_data_avg.shape, (height, width), "Output data shape mismatch.")
        self.assertEqual(master_data_avg.dtype, np.float32, "Output data type should be float32.")

        expected_stacked_data = np.stack(all_valid_data_arrays, axis=0)
        expected_mean_data = np.mean(expected_stacked_data, axis=0).astype(np.float32)

        np.testing.assert_allclose(master_data_avg, expected_mean_data, rtol=1e-5, err_msg="Averaged data is incorrect.")

        self.assertEqual(master_header_avg.get("NCOMBINE"), num_valid_files, "NCOMBINE header incorrect.")
        self.assertEqual(master_header_avg.get("COMBTYPE"), "AVERAGE", "COMBTYPE header incorrect.")

        # 2. Test shape mismatch handling
        mismatched_file_data = np.full((height + 1, width), 100.0, dtype=np.float32) # Different shape
        mismatched_file_path = self._create_dummy_fits_file("avg_img_mismatch", mismatched_file_data, dtype=np.float32)

        files_with_mismatch = valid_file_paths[:2] + [mismatched_file_path] + valid_file_paths[2:]

        output_path_mismatch = os.path.join(self.test_dir, "master_avg_mismatch.fits")
        success_mismatch = create_master_frame(files_with_mismatch, output_path_mismatch, method="average", frame_type="TESTAVG_MISMATCH")

        self.assertTrue(success_mismatch, "Average with mismatch should still succeed (by skipping bad file).")
        self.assertTrue(os.path.exists(output_path_mismatch), "Output file for mismatch test should exist.")

        master_data_mismatch = load_fits_data(output_path_mismatch)
        master_header_mismatch = get_fits_header(output_path_mismatch)

        self.assertIsNotNone(master_data_mismatch, "Loaded master data from mismatch test should not be None.")
        # Expected data is the same as the first test (all valid files), as the mismatched file is skipped.
        np.testing.assert_allclose(master_data_mismatch, expected_mean_data, rtol=1e-5, err_msg="Mismatch test data is incorrect (should be like all valid files).")
        self.assertEqual(master_header_mismatch.get("NCOMBINE"), num_valid_files, "NCOMBINE should be num_valid_files (original valid ones).")

if __name__ == '__main__':
    unittest.main()
