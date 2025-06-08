# Utility functions for the Astro App
import numpy as np
from astropy.io import fits
import os

def load_fits_data(filepath: str) -> np.ndarray | None:
    """
    Loads data from a FITS file, trying primary HDU then common image extensions.

    Args:
        filepath: Path to the FITS file.

    Returns:
        A NumPy array (typically float32) containing the image data,
        or None if an error occurs or no data is found.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with fits.open(filepath) as hdul:
            # Prioritize primary HDU if it contains data
            if hdul[0].data is not None and hdul[0].data.size > 0:
                data = hdul[0].data
                if data.dtype != np.float32: # Ensure float32 for consistency
                    data = data.astype(np.float32)
                return data
            else:
                # If primary HDU has no data, check common image extensions
                print(f"Primary HDU in {filepath} has no data or is empty. Trying common image extensions (e.g., SCI, IMAGE).")
                for ext_name in ['SCI', 'IMAGE']: # Common names for science image extensions
                    if ext_name in hdul and hdul[ext_name].data is not None and hdul[ext_name].data.size > 0:
                        print(f"Found data in extension '{ext_name}'.")
                        data = hdul[ext_name].data
                        if data.dtype != np.float32:
                            data = data.astype(np.float32)
                        return data

                # If still no data found, and primary HDU was indeed empty, report that.
                # This also handles cases where there's only a primary HDU but it's empty.
                if hdul[0].data is None or hdul[0].data.size == 0 :
                    print(f"Error: No data found in primary HDU or common alternative extensions in {filepath}.")
                    return None
                else:
                    # This case should ideally be caught by the first 'if' - primary HDU has data.
                    # Included for robustness, perhaps data was present but not np.float32 initially.
                    data = hdul[0].data
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    return data


    except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
        print(f"Error: File not found at {filepath}")
        return None
    except OSError as e: # Handles cases like permission errors or file not being a FITS file
        print(f"Error: Could not read FITS file at {filepath}. OS error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading FITS file {filepath}: {e}")
        return None

def save_fits_data(filepath: str, data: np.ndarray, header: fits.Header = None) -> bool:
    """
    Saves NumPy array data to a FITS file.

    Args:
        filepath: Path to save the new FITS file.
        data: NumPy array containing the image data.
        header: Optional astropy.io.fits.Header object to include in the FITS file.

    Returns:
        True if the file was saved successfully, False otherwise.
    """
    if data is None:
        print("Error: No data provided to save.")
        return False
    try:
        # Ensure data is in a FITS-compatible format (e.g., convert boolean arrays to int)
        if data.dtype == bool:
            data = data.astype(np.uint8)
        elif not np.issubdtype(data.dtype, np.number) and not np.issubdtype(data.dtype, bool):
             # Astropy might handle some other types, but explicitly convert non-numeric/non-bool
            print(f"Warning: Data type {data.dtype} might not be directly FITS compatible. Attempting conversion.")
            # Attempt a safe conversion if possible, otherwise it might fail in PrimaryHDU
            try:
                data = data.astype(np.float32) # Default to float32 if unsure
            except ValueError:
                print(f"Error: Could not convert data of type {data.dtype} to a FITS-compatible numeric type.")
                return False

        hdu = fits.PrimaryHDU(data=data, header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(filepath, overwrite=True, output_verify='fix') # Added output_verify
        print(f"FITS file saved successfully to {filepath}")
        return True
    except Exception as e:
        print(f"An error occurred while saving FITS file to {filepath}: {e}")
        return False

def get_fits_header(filepath: str) -> fits.Header | None:
    """
    Loads the header from a FITS file.
    It prioritizes headers from data-carrying HDUs (Primary, 'SCI', 'IMAGE').
    If the primary HDU appears to be data-less (e.g., NAXIS=0), it prefers headers from known image extensions.

    Args:
        filepath: Path to the FITS file.

    Returns:
        An astropy.io.fits.Header object, or None if an error occurs.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with fits.open(filepath) as hdul:
            primary_header = hdul[0].header
            primary_has_data_axes = primary_header.get('NAXIS', 0) > 0 and not (primary_header.get('NAXIS1', 0) == 0 and primary_header.get('NAXIS2', 0) == 0)

            # If primary HDU seems to lack data axes, try common image extensions first
            if not primary_has_data_axes or hdul[0].data is None:
                print(f"Primary HDU in {filepath} appears data-less or NAXIS=0. Checking 'SCI', 'IMAGE' extensions first for header.")
                for ext_name in ['SCI', 'IMAGE']:
                    if ext_name in hdul and hdul[ext_name].header is not None:
                        # Check if this extension itself has data axes
                        ext_header = hdul[ext_name].header
                        if ext_header.get('NAXIS', 0) > 0 and not (ext_header.get('NAXIS1', 0) == 0 and ext_header.get('NAXIS2', 0) == 0):
                            print(f"Found data-carrying header in extension '{ext_name}'.")
                            return ext_header
                print(f"No data-carrying header found in 'SCI' or 'IMAGE'. Falling back to primary HDU header for {filepath}.")

            # Otherwise, or if fallback occurs, return the primary header.
            return primary_header

    except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
        print(f"Error: File not found at {filepath}")
        return None
    except OSError as e:
        print(f"Error: Could not read FITS header from file at {filepath}. OS error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading FITS header from {filepath}: {e}")
        return None
