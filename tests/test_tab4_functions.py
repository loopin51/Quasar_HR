import unittest
import os
import tempfile
import shutil
import numpy as np
from astropy.io import fits
from astropy.table import Table
# from photutils.datasets import make_gaussian_sources_image # This function seems unavailable in photutils 2.0.2

# Add the parent directory to sys.path to allow importing modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tab4_functions import perform_photometry
from utils import save_fits_data # For creating test files

class TestPhotometry(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="astro_test_tab4_")
        self.fits_image_path = os.path.join(self.test_dir, "test_image.fits")

        # Create a synthetic image with Gaussian sources
        shape = (100, 100)
        # Define sources: x, y, peak_value (above background)
        source_params = [
            {'x': 20, 'y': 20, 'peak': 500},
            {'x': 50, 'y': 50, 'peak': 300},
            {'x': 70, 'y': 30, 'peak': 400}
        ]
        self.test_source_coords = [(p['x'], p['y']) for p in source_params]

        # Add a constant background
        self.background_level = 10.0
        image_data = np.full(shape, self.background_level, dtype=np.float32)

        # Add simple "stars" (e.g., 3x3 pixel squares)
        star_size = 1 # half-size, so 3x3 is (2*1+1)x(2*1+1)
        for src in source_params:
            x, y, peak = src['x'], src['y'], src['peak']
            # Ensure coordinates are within bounds for the star placement
            xmin = max(0, x - star_size)
            xmax = min(shape[1] -1, x + star_size)
            ymin = max(0, y - star_size)
            ymax = min(shape[0] -1, y + star_size)
            if xmin <= xmax and ymin <= ymax: # Check if the coordinate is valid for star size
                 image_data[ymin:ymax+1, xmin:xmax+1] += peak # Add peak value to background
            else: # If source is too close to edge to place a 3x3 star, make a single pixel bright
                 image_data[y,x] += peak


        self.image_data = image_data
        save_fits_data(self.fits_image_path, self.image_data, header=None)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_photometry_no_sky(self):
        """Test photometry without local sky subtraction."""
        ap_radius = 5.0
        results = perform_photometry(self.fits_image_path, self.test_source_coords, ap_radius)

        self.assertIsNotNone(results)
        self.assertEqual(len(results), len(self.test_source_coords))

        for i, res in enumerate(results):
            self.assertEqual(res['id'], i + 1)
            self.assertEqual(res['x'], self.test_source_coords[i][0])
            self.assertEqual(res['y'], self.test_source_coords[i][1])
            self.assertTrue(res['aperture_sum_raw'] > 0) # Flux should be positive
            self.assertEqual(res['sky_median_per_pixel'], 0.0)
            self.assertEqual(res['final_flux'], res['aperture_sum_raw'])
            self.assertIsNotNone(res['instrumental_mag'])
            self.assertTrue(res['error_message'] == "No local sky annulus defined." or res['error_message'] == '')


    def test_photometry_with_local_sky(self):
        """Test photometry with local sky annulus subtraction."""
        ap_radius = 5.0
        sky_in = 7.0
        sky_out = 10.0

        results = perform_photometry(self.fits_image_path, self.test_source_coords,
                                     ap_radius, sky_in, sky_out)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), len(self.test_source_coords))

        for res in results:
            self.assertTrue(res['aperture_sum_raw'] > 0)
            # With Gaussian sources on a flat background, sky median should be very close to background_level
            self.assertAlmostEqual(res['sky_median_per_pixel'], self.background_level, delta=1.0,
                                   msg=f"Sky estimation deviates significantly for source ID {res['id']}") # Allow some delta for noise/pixelation
            self.assertTrue(res['sky_sum_in_aperture'] > 0)
            self.assertLess(res['final_flux'], res['aperture_sum_raw']) # Sky subtracted flux should be less
            if res['final_flux'] > 0:
                self.assertIsNotNone(res['instrumental_mag'])
            self.assertEqual(res['error_message'], '')


    def test_photometry_source_near_edge(self):
        """Test photometry for a source near the image edge."""
        # Place a source very close to the edge (0,0)
        edge_coords = [(2, 2)]
        ap_radius = 5.0
        sky_in = 6.0
        sky_out = 8.0 # Annulus might also be partially off

        results = perform_photometry(self.fits_image_path, edge_coords,
                                     ap_radius, sky_in, sky_out)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        res = results[0]

        # Aperture/annulus will be partially off-image. photutils handles this by considering partial pixels.
        # We expect flux and sky to be calculated, possibly with warnings if area is too small.
        # The 'error_message' in our function might capture issues if ApertureStats fails.
        # For this test, just ensure it runs and produces a result.
        self.assertTrue(res['aperture_sum_raw'] >= 0)
        # Sky estimation might be less accurate or fail if annulus is mostly off-image.
        # The current implementation of perform_photometry does not explicitly check for partial apertures
        # beyond what ApertureStats does internally. If ApertureStats returns None for median (e.g. no pixels in annulus)
        # our code would set an error message.
        if "Could not determine median sky value" not in res['error_message']:
             self.assertTrue(res['sky_median_per_pixel'] >= 0) # Sky should be positive or zero

        # Check if a magnitude was computed or if flux became non-positive
        if res['final_flux'] > 0:
            self.assertIsNotNone(res['instrumental_mag'])
        else:
            self.assertIn("Non-positive flux", res['error_message'],
                          "Expected non-positive flux message if mag is None and flux <=0")


    def test_photometry_invalid_annulus(self):
        """Test with invalid sky annulus (inner_radius >= outer_radius)."""
        ap_radius = 5.0
        sky_in_invalid = 10.0
        sky_out_invalid = 8.0 # inner > outer

        results = perform_photometry(self.fits_image_path, [self.test_source_coords[0]],
                                     ap_radius, sky_in_invalid, sky_out_invalid)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertIn("Sky annulus inner radius must be less than outer radius", res['error_message'])
        self.assertEqual(res['sky_median_per_pixel'], 0.0) # Sky subtraction skipped
        self.assertEqual(res['final_flux'], res['aperture_sum_raw']) # Flux should be raw sum


    def test_photometry_non_positive_flux(self):
        """Test scenario leading to non-positive flux after sky subtraction."""
        # Create an image where sky is higher than source + sky in aperture
        high_sky_data = np.full(self.image_data.shape, 200.0, dtype=np.float32) # High constant sky
        source_pos = (50,50)
        high_sky_data[source_pos[1]-2:source_pos[1]+3, source_pos[0]-2:source_pos[0]+3] = 150.0 # Dim source area

        temp_high_sky_path = os.path.join(self.test_dir, "high_sky_image.fits")
        save_fits_data(temp_high_sky_path, high_sky_data, header=None)

        ap_radius = 3.0
        sky_in = 5.0
        sky_out = 8.0

        results = perform_photometry(temp_high_sky_path, [source_pos],
                                     ap_radius, sky_in, sky_out)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        res = results[0]

        self.assertTrue(res['final_flux'] <= 0)
        self.assertIsNone(res['instrumental_mag'])
        self.assertIn("Non-positive flux after sky subtraction", res['error_message'])

    def test_load_fits_fails(self):
        """Test photometry when the FITS file cannot be loaded."""
        non_existent_path = os.path.join(self.test_dir, "does_not_exist.fits")
        results = perform_photometry(non_existent_path, self.test_source_coords, 5.0)
        self.assertIsNone(results)

if __name__ == '__main__':
    unittest.main()


class TestPerformPhotometrySEP(unittest.TestCase):
    def setUp(self):
        # Create dummy FITS file
        image_data = np.zeros((100, 100), dtype=np.float32)
        # Add stars with enough separation to avoid overlap issues with typical aperture sizes
        # Star 1: center (25,25), region [23:28, 23:28] to make it 5x5
        image_data[23:28, 23:28] = 500.0
        # Star 2: center (75,75), region [73:78, 73:78] to make it 5x5
        image_data[73:78, 73:78] = 1000.0

        self.fits_path = os.path.join(os.path.dirname(__file__), 'dummy_test_image.fits') # Ensure it's created in the test directory

        # Add a slight background noise to make it more realistic for SEP
        # and to ensure background subtraction is non-trivial.
        rng = np.random.default_rng(seed=42) # for reproducibility
        image_data += rng.normal(loc=5.0, scale=1.0, size=image_data.shape).astype(np.float32)

        hdu = fits.PrimaryHDU(data=image_data)
        hdul = fits.HDUList([hdu])
        hdul.writeto(self.fits_path, overwrite=True)

    def tearDown(self):
        if os.path.exists(self.fits_path):
            os.remove(self.fits_path)

    def test_perform_photometry_sep_basic(self):
        # This import needs to be valid in the context of the test execution.
        # sys.path has already been modified at the top of this file.
        from tab4_functions import perform_photometry_sep

        # Adjust aperture_radius and extract_thresh based on dummy image characteristics
        # Star flux is concentrated, background is ~5.0. A 3x3 region sum for star 1 is ~500*9 = 4500.
        # A threshold of 5-sigma over background noise (stddev=1) should easily pick them up.
        # SEP's threshold is applied to data - background.
        results = perform_photometry_sep(self.fits_path, aperture_radius=3.0, extract_thresh=5.0)

        self.assertIsNotNone(results, "perform_photometry_sep should return a list or None, not raise an error here.")
        self.assertIsInstance(results, list, "Results should be a list.")

        # Depending on SEP's sensitivity and the exact nature of the dummy image (noise, peak values),
        # the number of detected sources might vary. For the two distinct bright sources, we expect 2.
        # If more are found (e.g., noise peaks), this assertion might need refinement or the dummy image made cleaner.
        self.assertEqual(len(results), 2, f"Should detect 2 sources, but found {len(results)}. Results: {results}")

        expected_keys = ['x', 'y', 'flux', 'fluxerr', 'magnitude', 'sep_flags']
        for res in results:
            for key in expected_keys:
                self.assertIn(key, res, f"Key '{key}' missing in result dictionary.")
            self.assertIsNotNone(res['x'], "Source 'x' coordinate should not be None.")
            self.assertIsNotNone(res['y'], "Source 'y' coordinate should not be None.")
            self.assertGreater(res['flux'], 0, "Flux should be positive for detected astronomical sources.")
            self.assertIsNotNone(res['magnitude'], "Magnitude should be calculated for positive flux.")
            # Check if source positions are reasonable (e.g. near 25,25 and 75,75)
            # SEP's x,y are 0-indexed and can be float values representing centroids.
            # Allow some tolerance for centroiding.
            is_star1 = np.allclose([res['x'], res['y']], [25, 25], atol=3) # Increased tolerance
            is_star2 = np.allclose([res['x'], res['y']], [75, 75], atol=3) # Increased tolerance
            self.assertTrue(is_star1 or is_star2, f"Detected source at ({res['x']},{res['y']}) is not near expected positions.")

    def test_perform_photometry_sep_no_sources_found(self):
        from tab4_functions import perform_photometry_sep
        # Use a very high threshold that should not find any sources.
        # Given peak of brightest star is ~1000 and globalrms is ~1,
        # threshold needs to be > 1000 / globalrms.
        # Setting to 1500.0 ensures it's well above.
        results = perform_photometry_sep(self.fits_path, aperture_radius=3.0, extract_thresh=1500.0)
        self.assertIsNotNone(results)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0, "Should find 0 sources with a very high threshold.")

    def test_perform_photometry_sep_invalid_fits_path(self):
        from tab4_functions import perform_photometry_sep
        results = perform_photometry_sep("non_existent_dummy_image.fits", aperture_radius=3.0, extract_thresh=1.5)
        self.assertIsNone(results, "Should return None for an invalid FITS path.")
