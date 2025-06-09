# Functions for Tab 1: Master Frame Generation
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats # For sigma clipping if implemented
from utils import load_fits_data, save_fits_data, get_fits_header
import time
import os

def create_master_frame(file_paths: list[str], output_path: str, method: str, frame_type: str = "UNKNOWN") -> bool:
    final_header = None # Precautionary initialization
    if not file_paths:
        print("Error: No input files provided for master frame creation.")
        return False

    valid_methods = ["median", "mean", "average", "sigma_clip_mean"]
    actual_method = method.lower()
    if actual_method == "average": # Treat "average" as "mean"
        actual_method = "mean"

    if actual_method not in valid_methods:
        print(f"Error: Unknown method '{method}'. Supported methods are: {valid_methods}")
        return False

    loaded_images_data = []
    first_header = None # This will serve as the base for the final header
    first_data_shape = None
    # original_dtype will be float32 because load_fits_data converts to it.

    for i, file_path in enumerate(file_paths):
        data = load_fits_data(file_path) # load_fits_data ensures data is float32

        if data is not None:
            if not loaded_images_data: # First valid file
                if data.ndim < 2: # Ensure data is at least 2D
                    print(f"Warning: Data in {file_path} is not at least 2D. Skipping.")
                    continue
                first_data_shape = data.shape
                first_header = get_fits_header(file_path)
                if first_header is None: # Minimal header if first valid file has no proper header
                    first_header = fits.Header()
                    for axis_idx, dim_size in enumerate(data.shape[::-1]): # NAXIS order is FITS standard
                        first_header[f'NAXIS{axis_idx+1}'] = dim_size
                    first_header['NAXIS'] = data.ndim
            elif data.shape != first_data_shape:
                print(f"Warning: Shape mismatch for {file_path} ({data.shape}) vs expected ({first_data_shape}). Skipping.")
                continue
            loaded_images_data.append(data)
        else:
            print(f"Warning: Could not load data from {file_path}. Skipping this file.")

    if not loaded_images_data:
        print(f"Error: No valid FITS data could be loaded/processed for {output_path} using {actual_method} method.")
        return False

    total_files_processed = len(loaded_images_data)

    try:
        image_stack = np.stack(loaded_images_data, axis=0)
    except Exception as e:
        print(f"Error stacking images: {e}. This may happen if shape checks were insufficient or data corrupted.")
        return False

    # print(f"Combining {total_files_processed} frames using '{actual_method}' method...") # Verbose
    combined_data = None
    if actual_method == "median":
        combined_data = np.median(image_stack, axis=0)
    elif actual_method == "mean": # Handles "average"
        combined_data = np.mean(image_stack, axis=0)
    elif actual_method == "sigma_clip_mean":
        mean_val, _, _ = sigma_clipped_stats(image_stack, axis=0, sigma=3.0, stdfunc='mad_std')
        combined_data = mean_val
        # print(f"Sigma clipping applied.") # Verbose

    if combined_data is None:
        print(f"Error: Combined data is None after processing method '{actual_method}'.") # Should not happen
        return False

    # Ensure output is float32, consistent with load_fits_data
    if combined_data.dtype != np.float32:
        combined_data = combined_data.astype(np.float32)

    # Use first_header as the base for the final output header
    final_header = first_header if first_header is not None else fits.Header()
    if not final_header.get('NAXIS',0): # If minimal header didn't get shape info
        for axis_idx, dim_size in enumerate(combined_data.shape[::-1]):
            final_header[f'NAXIS{axis_idx+1}'] = dim_size
        final_header['NAXIS'] = combined_data.ndim


    final_header.set("NCOMBINE", total_files_processed, "Number of frames combined")
    final_header.set("COMBTYPE", method.upper(), "Combination method used")
    final_header.set("FRAMTYPE", frame_type.upper(), "Type of frame")

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    final_header.add_history(f"Master frame ({frame_type}) created on {timestamp} UTC")
    final_header.add_history(f"Combined {total_files_processed} files using {method.upper()} method.")
    final_header.add_history("Input files (up to 10 listed):")
    # Correctly list file paths that were actually used (present in loaded_images_data)
    # This requires mapping loaded_images_data back to file_paths, or changing history logic.
    # For now, using original file_paths for history:
    processed_file_paths = [fp for fp_idx, fp in enumerate(file_paths) if loaded_images_data and fp_idx < len(loaded_images_data)] # Simplified
    for fp_idx, fp_path in enumerate(processed_file_paths):
        if fp_idx < 10:
            final_header.add_history(f"  {os.path.basename(fp_path)}")
        elif fp_idx == 10:
            final_header.add_history(f"  ... and {total_files_processed - 10} more files.")
            break

    if 'BZERO' in final_header: del final_header['BZERO']
    if 'BSCALE' in final_header: del final_header['BSCALE']

    return save_fits_data(output_path, combined_data, header=final_header)
