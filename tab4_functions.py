# Functions for Tab 4: Detailed Photometry and Catalog Analysis
import numpy as np
from astropy.io import fits # Though utils primarily handles this
from astropy.stats import SigmaClip
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from photutils.background import Background2D, MedianBackground
from utils import load_fits_data # Assuming utils.py is in PYTHONPATH
import math

def perform_photometry(
    fits_file_path: str,
    source_coords: list[tuple[float, float]],
    aperture_radius: float,
    sky_annulus_inner_radius: float | None = None,
    sky_annulus_outer_radius: float | None = None
) -> list[dict] | None:
    """
    Performs aperture photometry on specified sources in a FITS image.

    Args:
        fits_file_path: Path to the FITS image file.
        source_coords: A list of (x, y) coordinates for the sources.
                       These are 0-indexed (Python/Numpy convention).
        aperture_radius: Radius of the circular aperture for photometry.
        sky_annulus_inner_radius: Optional inner radius for the sky annulus.
        sky_annulus_outer_radius: Optional outer radius for the sky annulus.
                                  If None, or if inner_radius is None, global background might be considered
                                  or photometry done without explicit local sky subtraction.

    Returns:
        A list of dictionaries, where each dictionary contains photometry results
        for a source (e.g., 'id', 'x', 'y', 'aperture_sum', 'sky_median_per_pixel',
        'sky_subtracted_flux', 'instrumental_mag', 'error_message').
        Returns None if the image cannot be loaded.
    """
    image_data = load_fits_data(fits_file_path)
    if image_data is None:
        print(f"Error: Could not load image data from {fits_file_path}.")
        return None

    photometry_results = []

    # For a more robust global background if local sky isn't used for some sources
    # This is a fallback or alternative; local sky is preferred if annuli are defined.
    # bkg_estimator = MedianBackground()
    # try:
    #     bkg = Background2D(image_data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    #     global_background_level = bkg.background_median
    # except ValueError as e: # Handle cases where image is too small for Background2D
    #     print(f"Warning: Could not compute Background2D (image might be too small): {e}. Using global median as fallback background.")
    #     global_background_level = np.median(image_data)
    # print(f"Global median background estimate: {global_background_level}")


    apertures = CircularAperture(source_coords, r=aperture_radius)

    # Perform photometry using ApertureStats for more detailed statistics including median for sky if needed
    # ApertureStats can calculate statistics for multiple apertures at once.
    # We'll use it for the main aperture sum.
    ap_stats = ApertureStats(image_data, apertures, sigma_clip=None) # No sigma clip on source aperture initially
    aperture_sums = ap_stats.sum
    aperture_areas = np.array([ap.area for ap in apertures]) # Get area for each aperture

    idx = 0
    for coord, ap_sum, ap_area in zip(source_coords, aperture_sums, aperture_areas):
        x, y = coord
        result = {
            'id': idx + 1,
            'x': x,
            'y': y,
            'aperture_radius': aperture_radius,
            'aperture_sum_raw': ap_sum,
            'aperture_area': ap_area,
            'sky_median_per_pixel': 0.0, # Default if no sky annulus
            'sky_sum_in_aperture': 0.0,
            'final_flux': ap_sum,
            'instrumental_mag': None,
            'error_message': ''
        }

        if sky_annulus_inner_radius is not None and sky_annulus_outer_radius is not None:
            if sky_annulus_inner_radius >= sky_annulus_outer_radius:
                result['error_message'] = "Sky annulus inner radius must be less than outer radius."
                print(f"Warning for source ({x},{y}): Sky annulus inner radius >= outer. Skipping local sky.")
            else:
                sky_annulus = CircularAnnulus(source_coords,
                                              r_in=sky_annulus_inner_radius,
                                              r_out=sky_annulus_outer_radius)

                # Use ApertureStats for robust sky background estimation (median with sigma clipping)
                # We need to do this per source if annuli are source-specific,
                # or can be done in batch if annuli are same for all (but positions differ)
                # Here, we create a single annulus object centered on the current source for sky estimation.
                current_sky_annulus = CircularAnnulus((x,y),
                                                      r_in=sky_annulus_inner_radius,
                                                      r_out=sky_annulus_outer_radius)
                try:
                    # Sigma clipping for robust median calculation
                    sigma_clip_sky = SigmaClip(sigma=3.0, maxiters=5)
                    sky_stats = ApertureStats(image_data, current_sky_annulus, sigma_clip=sigma_clip_sky)

                    # sky_stats.median for a single aperture is a scalar, not an array
                    if sky_stats.median is not None:
                        sky_median_per_pixel = sky_stats.median
                        result['sky_median_per_pixel'] = sky_median_per_pixel
                        sky_sum_in_aperture = sky_median_per_pixel * ap_area
                        result['sky_sum_in_aperture'] = sky_sum_in_aperture
                        result['final_flux'] = ap_sum - sky_sum_in_aperture
                    else:
                        result['error_message'] = "Could not determine median sky value in annulus (e.g., all pixels masked)."
                        print(f"Warning for source ({x},{y}): Could not get median sky from annulus. Using raw flux.")
                except Exception as e: # Catch errors during sky calculation (e.g. annulus off image)
                    result['error_message'] = f"Error during local sky calculation: {str(e)}"
                    print(f"Warning for source ({x},{y}): Error in local sky calc: {e}. Using raw flux.")
        else:
            # Optional: Use global background if no local sky annulus defined
            # result['sky_median_per_pixel'] = global_background_level
            # sky_sum_in_aperture = global_background_level * ap_area
            # result['sky_sum_in_aperture'] = sky_sum_in_aperture
            # result['final_flux'] = ap_sum - sky_sum_in_aperture
            print(f"Note for source ({x},{y}): No sky annulus defined. Final flux is raw aperture sum.")
            result['error_message'] = "No local sky annulus defined."


        if result['final_flux'] is not None and result['final_flux'] > 0:
            result['instrumental_mag'] = -2.5 * math.log10(result['final_flux'])
        elif result['final_flux'] is not None: # Non-positive flux
            result['error_message'] += " Non-positive flux after sky subtraction; cannot calculate magnitude."
            print(f"Warning for source ({x},{y}): Non-positive flux ({result['final_flux']}). Mag not calculated.")

        photometry_results.append(result)
        idx += 1

    return photometry_results
