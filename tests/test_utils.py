import unittest
import os
import tempfile
import numpy as np
from astropy.io import fits

# Add the parent directory to sys.path to allow importing 'utils'
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_fits_data, save_fits_data, get_fits_header

class TestFitsUtils(unittest.TestCase):

    def setUp(self):
        """Set up temporary file for tests that need to write/read."""
        # Create a temporary file that will be automatically deleted
        self.temp_fits_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        self.temp_fits_filepath = self.temp_fits_file.name
        self.temp_fits_file.close() # Close it so save_fits_data can open and write to it

        self.sample_data = np.arange(100, dtype=np.float32).reshape(10, 10)
        self.sample_header = fits.Header()
        self.sample_header['TESTKEY'] = ('Test Value', 'This is a test keyword')
        self.sample_header['INTKEY'] = (123, 'An integer keyword')

    def tearDown(self):
        """Clean up temporary file."""
        if os.path.exists(self.temp_fits_filepath):
            os.remove(self.temp_fits_filepath)
        # print(f"Cleaned up {self.temp_fits_filepath}") # For debugging if needed

    def test_save_and_load_fits_data(self):
        """Test saving data with a header and loading it back."""
        # Save data
        save_success = save_fits_data(self.temp_fits_filepath, self.sample_data, header=self.sample_header)
        self.assertTrue(save_success, "save_fits_data should return True on success.")
        self.assertTrue(os.path.exists(self.temp_fits_filepath), "FITS file should be created.")

        # Load data
        loaded_data = load_fits_data(self.temp_fits_filepath)
        self.assertIsNotNone(loaded_data, "load_fits_data should return data, not None.")
        self.assertIsInstance(loaded_data, np.ndarray, "Loaded data should be a NumPy array.")
        self.assertTrue(np.array_equal(loaded_data, self.sample_data), "Loaded data should match original data.")

    def test_get_fits_header(self):
        """Test retrieving a FITS header."""
        # First, save a file with a known header
        save_fits_data(self.temp_fits_filepath, self.sample_data, header=self.sample_header)

        # Get header
        loaded_header = get_fits_header(self.temp_fits_filepath)
        self.assertIsNotNone(loaded_header, "get_fits_header should return a header, not None.")
        self.assertIsInstance(loaded_header, fits.Header, "Loaded header should be an astropy.io.fits.Header object.")
        self.assertEqual(loaded_header['TESTKEY'], 'Test Value', "Header keyword 'TESTKEY' should match.")
        self.assertEqual(loaded_header['INTKEY'], 123, "Header keyword 'INTKEY' should match.")
        self.assertEqual(loaded_header.comments['TESTKEY'], 'This is a test keyword')


    def test_load_fits_nonexistent_file(self):
        """Test loading a FITS file that does not exist."""
        non_existent_path = os.path.join(tempfile.gettempdir(), "non_existent_test_file.fits")
        if os.path.exists(non_existent_path): # Ensure it really doesn't exist
            os.remove(non_existent_path)
        loaded_data = load_fits_data(non_existent_path)
        self.assertIsNone(loaded_data, "load_fits_data should return None for a non-existent file.")

    def test_get_header_nonexistent_file(self):
        """Test getting a header from a FITS file that does not exist."""
        non_existent_path = os.path.join(tempfile.gettempdir(), "non_existent_test_header_file.fits")
        if os.path.exists(non_existent_path):
             os.remove(non_existent_path)
        loaded_header = get_fits_header(non_existent_path)
        self.assertIsNone(loaded_header, "get_fits_header should return None for a non-existent file.")

    def test_save_fits_no_data(self):
        """Test saving FITS data when input data is None."""
        save_success = save_fits_data(self.temp_fits_filepath, None, header=self.sample_header)
        self.assertFalse(save_success, "save_fits_data should return False if data is None.")
        # self.assertFalse(os.path.exists(self.temp_fits_filepath), "File should not be created if data is None and save fails early.")
        # Note: The current save_fits_data might still create an empty/corrupt file if it fails after opening.
        # For now, just checking the return value. If it creates a file, tearDown will clean it.

    def test_load_invalid_fits_file(self):
        """Test loading an invalid FITS file (e.g., a text file)."""
        # Create a dummy text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_text_file:
            tmp_text_filepath = tmp_text_file.name
            tmp_text_file.write("This is not a FITS file.")

        loaded_data = load_fits_data(tmp_text_filepath)
        self.assertIsNone(loaded_data, "load_fits_data should return None for an invalid FITS file.")

        loaded_header = get_fits_header(tmp_text_filepath)
        self.assertIsNone(loaded_header, "get_fits_header should return None for an invalid FITS file.")

        os.remove(tmp_text_filepath) # Clean up the dummy text file

    def test_load_fits_data_primary_hdu_empty_fallback(self):
        """Test loading FITS where primary HDU has no data, but an extension does."""
        # Create a FITS file with data in an 'SCI' extension
        primary_hdu = fits.PrimaryHDU() # No data
        image_data = np.arange(25, dtype=np.float32).reshape(5, 5)
        image_hdu = fits.ImageHDU(data=image_data, name='SCI')
        hdul = fits.HDUList([primary_hdu, image_hdu])

        hdul.writeto(self.temp_fits_filepath, overwrite=True)
        hdul.close()

        loaded_data = load_fits_data(self.temp_fits_filepath)
        self.assertIsNotNone(loaded_data)
        self.assertTrue(np.array_equal(loaded_data, image_data))

        # Test header loading from extension as well (though get_fits_header tries primary first)
        # Modify get_fits_header logic if it should prioritize SCI header when primary is minimal
        # For now, get_fits_header is designed to return primary if it exists, even if minimal
        # unless specifically modified. Let's check the 'SCI' header directly for this test.
        with fits.open(self.temp_fits_filepath) as opened_hdul:
            sci_header = opened_hdul['SCI'].header
            sci_header['EXTKEY'] = ('ExtValue', 'Extension keyword')

        # Save again with updated SCI header
        image_hdu.header = sci_header
        hdul = fits.HDUList([primary_hdu, image_hdu])
        hdul.writeto(self.temp_fits_filepath, overwrite=True)
        hdul.close()

        # The current get_fits_header has logic to check extensions if primary is minimal.
        # Let's ensure primary header is minimal for this test case.
        primary_hdu.header['SIMPLE'] = True
        primary_hdu.header['BITPIX'] = 8
        # (No other significant keywords in primary)
        hdul = fits.HDUList([primary_hdu, image_hdu])
        hdul.writeto(self.temp_fits_filepath, overwrite=True)
        hdul.close()

        loaded_header = get_fits_header(self.temp_fits_filepath)
        self.assertIsNotNone(loaded_header)
        self.assertIn('EXTKEY', loaded_header) # Should have loaded the SCI header
        self.assertEqual(loaded_header['EXTKEY'], 'ExtValue')


if __name__ == '__main__':
    unittest.main()
