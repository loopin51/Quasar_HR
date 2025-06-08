# Functions for Tab 2: LIGHT Frame Correction
import numpy as np
from astropy.io import fits
from utils import load_fits_data, save_fits_data, get_fits_header # Assuming utils.py is in PYTHONPATH
import time
import os # Import the os module

def get_exposure_time(header: fits.Header) -> float | None:
    """
    Attempts to get the exposure time from a FITS header.
    Checks common keywords: EXPTIME, EXPOSURE.
    """
    if header is None:
        return None
    for keyword in ['EXPTIME', 'EXPOSURE']:
        if keyword in header:
            try:
                return float(header[keyword])
            except (ValueError, TypeError):
                print(f"Warning: Could not parse exposure time from keyword {keyword}: {header[keyword]}")
    print("Warning: Exposure time keyword (EXPTIME, EXPOSURE) not found in header.")
    return None

def correct_light_frame(
    raw_light_path: str,
    output_path: str,
    master_bias_path: str | None = None,
    master_dark_path: str | None = None,
    master_flat_path: str | None = None
) -> bool:
    """
    Corrects a raw LIGHT frame using provided master calibration frames.

    Order of operations:
    1. Bias subtraction (if master_bias_path is provided).
    2. Dark subtraction (if master_dark_path is provided, scaled by exposure time).
    3. Flat fielding (if master_flat_path is provided, after normalizing the flat).

    Args:
        raw_light_path: Path to the raw LIGHT FITS file.
        output_path: Path to save the corrected LIGHT FITS file.
        master_bias_path: Optional path to the master BIAS FITS file.
        master_dark_path: Optional path to the master DARK FITS file.
        master_flat_path: Optional path to the master FLAT FITS file.

    Returns:
        True if the LIGHT frame was corrected and saved successfully, False otherwise.
    """
    print(f"Starting correction for LIGHT frame: {raw_light_path}")

    # Load raw light frame data and header
    raw_light_data = load_fits_data(raw_light_path)
    if raw_light_data is None:
        print(f"Error: Could not load raw LIGHT frame from {raw_light_path}. Aborting correction.")
        return False

    raw_light_header = get_fits_header(raw_light_path)
    if raw_light_header is None:
        print(f"Warning: Could not load header from {raw_light_path}. Proceeding with minimal header.")
        raw_light_header = fits.Header() # Create a minimal header

    # Initialize corrected_data as a copy of the raw data (as float32 for processing)
    corrected_data = raw_light_data.astype(np.float32, copy=True)

    history_entries = [f"LIGHT frame correction started: {time.strftime('%Y-%m-%dT%H:%M:%S UTC')}"]
    bias_subtracted_flag = False
    dark_subtracted_flag = False
    flat_fielded_flag = False

    # 1. Bias Subtraction
    if master_bias_path:
        print(f"Loading master BIAS from: {master_bias_path}")
        master_bias_data = load_fits_data(master_bias_path)
        if master_bias_data is None:
            print(f"Warning: Could not load master BIAS from {master_bias_path}. Skipping BIAS subtraction.")
            history_entries.append(f"BIAS subtraction skipped: Failed to load {master_bias_path}")
        elif master_bias_data.shape != corrected_data.shape:
            print(f"Warning: Master BIAS shape {master_bias_data.shape} differs from LIGHT frame shape {corrected_data.shape}. Skipping BIAS subtraction.")
            history_entries.append(f"BIAS subtraction skipped: Shape mismatch for {master_bias_path}")
        else:
            corrected_data -= master_bias_data.astype(np.float32)
            print("Master BIAS subtracted.")
            history_entries.append(f"Subtracted master BIAS: {master_bias_path}")
            bias_subtracted_flag = True

    # 2. Dark Subtraction
    if master_dark_path:
        print(f"Loading master DARK from: {master_dark_path}")
        master_dark_data = load_fits_data(master_dark_path)
        if master_dark_data is None:
            print(f"Warning: Could not load master DARK from {master_dark_path}. Skipping DARK subtraction.")
            history_entries.append(f"DARK subtraction skipped: Failed to load {master_dark_path}")
        elif master_dark_data.shape != corrected_data.shape:
            print(f"Warning: Master DARK shape {master_dark_data.shape} differs from LIGHT frame shape {corrected_data.shape}. Skipping DARK subtraction.")
            history_entries.append(f"DARK subtraction skipped: Shape mismatch for {master_dark_path}")
        else:
            light_exptime = get_exposure_time(raw_light_header)
            master_dark_header = get_fits_header(master_dark_path)
            dark_exptime = get_exposure_time(master_dark_header)

            if light_exptime is not None and dark_exptime is not None and dark_exptime > 0:
                scale_factor = light_exptime / dark_exptime
                corrected_data -= (master_dark_data.astype(np.float32) * scale_factor)
                dark_subtracted_flag = True
                if light_exptime == dark_exptime:
                    print("Applied master DARK (unscaled, exposure times equal).")
                    history_entries.append(f"Applied master DARK (unscaled, exp times equal): {master_dark_path}")
                else:
                    print(f"Applied scaled master DARK by factor {scale_factor:.3f} (Light exp: {light_exptime}s, Dark exp: {dark_exptime}s).")
                    history_entries.append(f"Applied scaled master DARK: {master_dark_path} (Scale: {scale_factor:.3f})")
            elif light_exptime is None or dark_exptime is None:
                print("Warning: Could not determine exposure times for LIGHT or master DARK. Applying DARK without scaling.")
                corrected_data -= master_dark_data.astype(np.float32)
                history_entries.append(f"Applied master DARK (unscaled, exp times missing): {master_dark_path}")
                dark_subtracted_flag = True
            elif dark_exptime <= 0:
                print(f"Warning: Master DARK exposure time is zero or negative ({dark_exptime}s). Skipping DARK subtraction.")
                history_entries.append(f"DARK subtraction skipped: Invalid exposure time for {master_dark_path}")

    # 3. Flat Fielding
    if master_flat_path:
        print(f"Loading master FLAT from: {master_flat_path}")
        master_flat_data = load_fits_data(master_flat_path)
        if master_flat_data is None:
            print(f"Warning: Could not load master FLAT from {master_flat_path}. Skipping FLAT fielding.")
            history_entries.append(f"FLAT fielding skipped: Failed to load {master_flat_path}")
        elif master_flat_data.shape != corrected_data.shape:
            print(f"Warning: Master FLAT shape {master_flat_data.shape} differs from LIGHT frame shape {corrected_data.shape}. Skipping FLAT fielding.")
            history_entries.append(f"FLAT fielding skipped: Shape mismatch for {master_flat_path}")
        else:
            median_flat = np.median(master_flat_data)
            if median_flat == 0:
                print("Warning: Median of master FLAT is zero. Skipping FLAT fielding to avoid division by zero.")
                history_entries.append(f"FLAT fielding skipped: Median of {master_flat_path} is zero.")
            else:
                normalized_flat = master_flat_data.astype(np.float32) / median_flat
                normalized_flat[normalized_flat == 0] = 1.0 # Avoid division by zero in image data
                print(f"Master FLAT normalized by median value: {median_flat:.3f}")
                corrected_data /= normalized_flat
                print("Flat fielding applied.")
                history_entries.append(f"Applied master FLAT: {master_flat_path} (Normalized by median: {median_flat:.3f})")
                flat_fielded_flag = True

    # Update header
    raw_light_header["BZERO"] = 0.0
    raw_light_header["BSCALE"] = 1.0
    raw_light_header.set("DATAMIN", np.min(corrected_data), "Minimum pixel value in corrected data")
    raw_light_header.set("DATAMAX", np.max(corrected_data), "Maximum pixel value in corrected data")
    raw_light_header.add_comment("LIGHT frame processed by custom script.")

    if bias_subtracted_flag:
        raw_light_header["CAL_BIAS"] = (os.path.basename(master_bias_path), "Master BIAS used")
    if dark_subtracted_flag:
        raw_light_header["CAL_DARK"] = (os.path.basename(master_dark_path), "Master DARK used")
    if flat_fielded_flag:
        raw_light_header["CAL_FLAT"] = (os.path.basename(master_flat_path), "Master FLAT used")

    for entry in history_entries:
        raw_light_header.add_history(entry)
    raw_light_header.add_history(f"LIGHT frame correction completed: {time.strftime('%Y-%m-%dT%H:%M:%S UTC')}")

    print(f"Saving corrected LIGHT frame to: {output_path}")
    if save_fits_data(output_path, corrected_data, header=raw_light_header):
        print("Corrected LIGHT frame saved successfully.")
        return True
    else:
        print("Error: Failed to save corrected LIGHT frame.")
        return False
