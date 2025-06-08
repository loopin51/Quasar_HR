# Functions for Tab 1: Master Frame Generation
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats # For sigma clipping if implemented
from utils import load_fits_data, save_fits_data, get_fits_header # Assuming utils.py is in PYTHONPATH
import time # For adding HISTORY entries

def create_master_frame(file_paths: list[str], output_path: str, method: str = "median", frame_type: str = "UNKNOWN") -> bool:
    """
    Creates a master calibration frame (BIAS, DARK, or FLAT) from a list of FITS files.

    Args:
        file_paths: A list of strings, where each string is a path to a FITS file.
        output_path: The path where the combined master FITS file will be saved.
        method: The combination method. Options: "median" (default), "mean", "sigma_clip_mean".
        frame_type: The type of frame being created (e.g., "BIAS", "DARK", "FLAT").
                    This is used to update the FITS header.

    Returns:
        True if the master frame was created and saved successfully, False otherwise.
    """
    if not file_paths:
        print("Error: No input files provided for master frame creation.")
        return False

    loaded_images = []
    first_header = None

    for i, file_path in enumerate(file_paths):
        print(f"Loading file {i+1}/{len(file_paths)}: {file_path}")
        data = load_fits_data(file_path)
        if data is not None:
            loaded_images.append(data)
            if first_header is None: # Get header from the first successfully loaded file
                first_header = get_fits_header(file_path)
        else:
            print(f"Warning: Could not load data from {file_path}. Skipping this file.")

    if not loaded_images:
        print("Error: No valid FITS data could be loaded from the provided files. Master frame not created.")
        return False

    # Ensure all images have the same dimensions
    first_shape = loaded_images[0].shape
    for i, img in enumerate(loaded_images[1:], start=1):
        if img.shape != first_shape:
            print(f"Error: Image dimensions mismatch. {file_paths[0]} has shape {first_shape}, "
                  f"but file {file_paths[i]} has shape {img.shape}. Cannot combine.")
            return False

    try:
        # Stack images into a 3D numpy array (list of 2D arrays)
        # The first dimension will be the number of images.
        image_stack = np.stack(loaded_images, axis=0)
    except Exception as e:
        print(f"Error stacking images: {e}. Check if all images have the same dimensions.")
        return False

    print(f"Combining {len(loaded_images)} frames using '{method}' method...")
    master_image = None
    if method.lower() == "median":
        master_image = np.median(image_stack, axis=0)
    elif method.lower() == "mean":
        master_image = np.mean(image_stack, axis=0)
    elif method.lower() == "sigma_clip_mean":
        # sigma_clipped_stats returns mean, median, stddev of the sigma-clipped data
        # We want the mean of the sigma-clipped stack.
        # This usually applies clipping along the stack axis (axis=0) for each pixel.
        # For direct application, astropy.stats.sigma_clip can be used then np.mean.
        # For simplicity, let's use a common approach:
        mean, median_clipped, std_clipped = sigma_clipped_stats(image_stack, axis=0, sigma=3.0)
        master_image = mean # Using the mean of the sigma-clipped data
        print(f"Sigma clipping applied. Original median: {np.median(image_stack, axis=0)[0,0]}, Clipped mean: {master_image[0,0]}")
    else:
        print(f"Error: Unknown combination method '{method}'. Supported methods are 'median', 'mean', 'sigma_clip_mean'.")
        return False

    if master_image is None:
        print("Error: Master image could not be computed.")
        return False

    # Prepare header for the output file
    if first_header is None:
        # Create a minimal header if no valid header could be read
        print("Warning: Could not retrieve a header from input files. Creating a minimal header.")
        first_header = fits.Header()

    # Update header information
    first_header["NCOMBINE"] = (len(loaded_images), "Number of frames combined")
    first_header["COMBTYPE"] = (method.upper(), "Combination method")
    first_header["FRAMTYPE"] = (frame_type.upper(), "Type of frame (e.g., BIAS, DARK, FLAT)")

    # Add HISTORY entries
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    first_header.add_history(f"Master frame created on {timestamp} UTC")
    first_header.add_history(f"Combined {len(loaded_images)} files using {method} method.")
    first_header.add_history("Input files:")
    for fp in file_paths[:min(len(file_paths), 10)]: # List up to 10 files
        first_header.add_history(f"  {fp}")
    if len(file_paths) > 10:
        first_header.add_history(f"  ... and {len(file_paths) - 10} more files.")

    # Ensure the master image data type is appropriate for FITS (e.g., float32)
    if master_image.dtype != np.float32 and master_image.dtype != np.float64:
        print(f"Converting master image from {master_image.dtype} to float32 for saving.")
        master_image = master_image.astype(np.float32)

    print(f"Saving master {frame_type} frame to {output_path}")
    if save_fits_data(output_path, master_image, header=first_header):
        print(f"Master {frame_type} frame saved successfully.")
        return True
    else:
        print(f"Error: Failed to save master {frame_type} frame.")
        return False
