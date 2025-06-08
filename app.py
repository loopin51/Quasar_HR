from tab4_functions import perform_photometry
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import numpy as np
from astropy.io import fits

# Main application file for the Gradio Astro App
import gradio as gr
import os
import tempfile
from tab1_functions import create_master_frame
from tab2_functions import correct_light_frame
from utils import load_fits_data, get_fits_header, save_fits_data
from tab3_functions import calculate_extinction_coefficient

def handle_extinction_calculation(airmass_str: str, magnitude_str: str) -> tuple[str, str, str, str]:
    """
    Handles the input, calculation, and output for the extinction coefficient tab.
    Returns k, k_err, m0, and any error message.
    """
    try:
        if not airmass_str.strip() or not magnitude_str.strip():
            return "", "", "", "Error: Airmass and Magnitude fields cannot be empty."

        airmasses = [float(x.strip()) for x in airmass_str.split(',')]
        magnitudes = [float(x.strip()) for x in magnitude_str.split(',')]

        if len(airmasses) != len(magnitudes):
            return "", "", "", "Error: Number of airmass values must match number of magnitude values."
        if len(airmasses) < 2:
            return "", "", "", "Error: At least two data points are required for calculation."

        k, k_err = calculate_extinction_coefficient(airmasses, magnitudes)

        # Also, calculate m0 (intercept) for display
        # Using scipy.stats.linregress again, though it's a bit redundant.
        # Alternatively, the main function could be modified to return m0.
        # For now, let's keep it simple and recalculate here or derive if possible.
        # from m = m0 - kX => m0 = m + kX. We need an average or the intercept.
        # The linregress function returns intercept, slope, r_value, p_value, stderr, intercept_stderr
        from scipy.stats import linregress
        regression_result = linregress(airmasses, magnitudes)
        m0 = regression_result.intercept

        return f"{k:.4f}", f"{k_err:.4f}", f"{m0:.4f}", "" # k, k_err, m0, error_message

    except ValueError as ve:
        if "could not convert string to float" in str(ve):
            return "", "", "", "Error: Invalid input. Please enter comma-separated numbers (e.g., 1.1, 1.2, 1.3)."
        return "", "", "", f"Error: {str(ve)}"
    except Exception as e:
        return "", "", "", f"An unexpected error occurred: {str(e)}"

# Dummy handlers for Tab 1 UI (will be replaced with actual logic)
def handle_generate_master_bias(bias_uploads_list, current_master_bias_path_state):
    if not bias_uploads_list:
        return "No BIAS files uploaded. Cannot generate Master BIAS.", current_master_bias_path_state, gr.File(visible=False)

    # Ensure a directory for masters exists
    output_dir = "masters_output"
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            error_msg = f"Error creating output directory {output_dir}: {e}"
            print(error_msg)
            return error_msg, current_master_bias_path_state, gr.File(visible=False)

    output_master_bias_path = os.path.join(output_dir, "master_bias.fits")

    uploaded_file_paths = [f.name for f in bias_uploads_list]

    print(f"Calling create_master_frame for BIAS with files: {uploaded_file_paths} to output: {output_master_bias_path}")
    try:
        success = create_master_frame(
            file_paths=uploaded_file_paths,
            output_path=output_master_bias_path,
            method="median",
            frame_type="BIAS"
        )
    except Exception as e_create:
        # Catch errors from create_master_frame itself (e.g., if it raises exceptions not just returning False)
        error_msg = f"Exception during Master BIAS generation: {e_create}"
        print(error_msg)
        return error_msg, current_master_bias_path_state, gr.File(visible=False)


    if success:
        print(f"Master BIAS generated successfully: {output_master_bias_path}")
        return f"Master BIAS generated: {output_master_bias_path}", output_master_bias_path, gr.File(value=output_master_bias_path, label=f"Download Master BIAS ({os.path.basename(output_master_bias_path)})", visible=True, interactive=True)
    else:
        print("Failed to generate Master BIAS (create_master_frame returned False).")
        # Attempt to remove partially created master bias if it exists and failed
        if os.path.exists(output_master_bias_path):
            try:
                os.remove(output_master_bias_path)
                print(f"Removed incomplete master bias: {output_master_bias_path}")
            except Exception as e_rem:
                print(f"Could not remove incomplete master bias {output_master_bias_path}: {e_rem}")
        return "Failed to generate Master BIAS. Check console/logs.", current_master_bias_path_state, gr.File(visible=False)

def handle_generate_master_dark(dark_uploads_list, master_bias_path, current_master_dark_paths_state):
    print(f"Master BIAS path received in handle_generate_master_dark: {master_bias_path}")
    if not dark_uploads_list:
        return "No DARK files uploaded. Cannot generate Master DARKs.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)

    if not master_bias_path or not os.path.exists(master_bias_path):
        print(f"Master BIAS path invalid or not found: {master_bias_path}")
        return "Master BIAS not available or path invalid. Please generate or upload Master BIAS first.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)

    master_bias_data = load_fits_data(master_bias_path)
    if master_bias_data is None:
        return f"Failed to load Master BIAS data from {master_bias_path}.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)

    output_dir = "masters_output"
    temp_subtracted_dark_dir = os.path.join(output_dir, "temp_subtracted_darks")
    os.makedirs(temp_subtracted_dark_dir, exist_ok=True)
    print(f"Ensured temp directory exists: {temp_subtracted_dark_dir}")

    grouped_darks_by_exp = {}
    raw_dark_paths = [f.name for f in dark_uploads_list]

    for dark_path in raw_dark_paths:
        header = get_fits_header(dark_path)
        if not header:
            print(f"Warning: Could not read header for {dark_path}. Skipping.")
            continue

        exptime_val = header.get('EXPTIME', header.get('EXPOSURE'))
        if exptime_val is None:
            print(f"Warning: Exposure time not found in header for {dark_path}. Skipping.")
            continue
        try:
            exptime_float = float(exptime_val)
            if exptime_float <= 0:
                 print(f"Warning: Invalid exposure time {exptime_float} in {dark_path}. Skipping.")
                 continue
        except ValueError:
            print(f"Warning: Could not convert exposure time '{exptime_val}' to float for {dark_path}. Skipping.")
            continue

        exptime_str = str(exptime_float).replace('.', 'p')

        if exptime_str not in grouped_darks_by_exp:
            grouped_darks_by_exp[exptime_str] = []
        grouped_darks_by_exp[exptime_str].append(dark_path)

    if not grouped_darks_by_exp:
        return "No DARK frames with valid exposure times found.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)

    new_master_dark_paths = (current_master_dark_paths_state or {}).copy()
    status_messages = []

    for exptime_str, file_list in grouped_darks_by_exp.items():
        if not file_list: continue # Should not happen if logic above is correct

        print(f"Processing DARKs for exposure: {exptime_str} ({len(file_list)} files)")
        bias_subtracted_dark_paths = []

        for i, raw_dark_path_iter in enumerate(file_list): # Renamed to avoid conflict with outer scope var if any
            raw_dark_data = load_fits_data(raw_dark_path_iter)
            if raw_dark_data is None:
                status_messages.append(f"Warning: Failed to load raw DARK {raw_dark_path_iter}. Skipping.")
                continue
            if raw_dark_data.shape != master_bias_data.shape:
                status_messages.append(f"Warning: Shape mismatch between DARK {raw_dark_path_iter} {raw_dark_data.shape} and Master BIAS {master_bias_data.shape}. Skipping.")
                continue

            subtracted_data = raw_dark_data - master_bias_data
            temp_dark_filename = f"bias_subtracted_dark_{exptime_str}_{i}.fits"
            temp_path = os.path.join(temp_subtracted_dark_dir, temp_dark_filename)

            original_header = get_fits_header(raw_dark_path_iter)
            if original_header:
                original_header.add_history(f"BIAS subtracted using {os.path.basename(master_bias_path)}")

            if save_fits_data(temp_path, subtracted_data, header=original_header):
                bias_subtracted_dark_paths.append(temp_path)
            else:
                status_messages.append(f"Warning: Failed to save BIAS-subtracted DARK {temp_path}. Skipping.")

        if not bias_subtracted_dark_paths:
            status_messages.append(f"No valid BIAS-subtracted DARKs for exposure {exptime_str}. Master DARK not created for this group.")
            continue

        output_master_dark_path = os.path.join(output_dir, f"master_dark_{exptime_str}.fits") # Corrected filename format

        success = create_master_frame(
            file_paths=bias_subtracted_dark_paths,
            output_path=output_master_dark_path,
            method="median",
            frame_type=f"DARK_{exptime_str}" # Use sanitized exptime_str
        )

        if success:
            new_master_dark_paths[exptime_str] = output_master_dark_path
            status_messages.append(f"Master DARK for {exptime_str} generated: {output_master_dark_path}")
        else:
            status_messages.append(f"Failed to generate Master DARK for {exptime_str}.")
            if os.path.exists(output_master_dark_path):
                try: os.remove(output_master_dark_path)
                except Exception: pass

        for temp_f_path in bias_subtracted_dark_paths:
            try:
                if os.path.exists(temp_f_path): os.remove(temp_f_path)
            except Exception as e_clean:
                print(f"Warning: Could not delete temp file {temp_f_path}: {e_clean}")

    try:
        if not os.listdir(temp_subtracted_dark_dir):
            os.rmdir(temp_subtracted_dark_dir)
        # else: Consider removing the whole tree if files might be left on failure
        # import shutil; shutil.rmtree(temp_subtracted_dark_dir) # Be cautious
    except Exception as e_rmdir:
         print(f"Could not remove temp dir {temp_subtracted_dark_dir}: {e_rmdir}")

    final_status = "
".join(status_messages) if status_messages else "Processing completed."
    if not new_master_dark_paths:
        final_status += "
No Master DARKs were successfully generated."

    dark_paths_display_text = "Generated Master DARKs:
" + "
".join([f"{exp}: {p}" for exp, p in new_master_dark_paths.items()])
    return final_status, new_master_dark_paths, gr.Textbox(value=dark_paths_display_text, label="Generated Master DARK Paths", visible=True, interactive=False)

def handle_generate_master_flat(flat_uploads_list, current_master_flat_paths_state):
    if not flat_uploads_list:
        return "No FLAT files uploaded. Cannot generate Preliminary Master FLATs.", current_master_flat_paths_state or {}, gr.Textbox(visible=False)

    output_dir = "masters_output" # Same as for BIAS/DARK
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    grouped_flats_by_filter = {}
    raw_flat_paths = [f.name for f in flat_uploads_list]

    for flat_path in raw_flat_paths:
        header = get_fits_header(flat_path) # Assumes get_fits_header is imported via utils
        if not header:
            print(f"Warning: Could not read header for {flat_path}. Skipping.")
            continue

        filter_name_val = header.get('FILTER', header.get('FILTER1', header.get('FILTNAME')))
        if filter_name_val is None:
            print(f"Warning: Filter name not found in header for {flat_path}. Skipping.")
            continue
        filter_name = str(filter_name_val).strip().replace(" ", "_")
        if not filter_name:
             print(f"Warning: Filter name is empty for {flat_path} after sanitization. Skipping.")
             continue

        if filter_name not in grouped_flats_by_filter:
            grouped_flats_by_filter[filter_name] = []
        grouped_flats_by_filter[filter_name].append(flat_path)

    if not grouped_flats_by_filter:
        return "No FLAT frames with valid filter names found.", current_master_flat_paths_state or {}, gr.Textbox(visible=False)

    new_master_flat_paths = (current_master_flat_paths_state or {}).copy()
    status_messages = []

    for filter_name_key, file_list in grouped_flats_by_filter.items():
        if not file_list: continue

        print(f"Processing Prelim Master FLATs for filter: {filter_name_key} ({len(file_list)} files)")

        output_master_flat_path = os.path.join(output_dir, f"prelim_master_flat_{filter_name_key}.fits")

        try:
            success = create_master_frame(
                file_paths=file_list,
                output_path=output_master_flat_path,
                method="median",
                frame_type=f"PRELIM_FLAT_{filter_name_key.upper()}"
            )
        except Exception as e_create_flat:
            error_msg = f"Exception during Prelim Master FLAT generation for filter {filter_name_key}: {e_create_flat}"
            print(error_msg)
            status_messages.append(error_msg)
            success = False


        if success:
            new_master_flat_paths[filter_name_key] = output_master_flat_path
            status_messages.append(f"Preliminary Master FLAT for filter '{filter_name_key}' generated: {output_master_flat_path}")
        else:
            status_messages.append(f"Failed to generate Preliminary Master FLAT for filter '{filter_name_key}'.")
            if os.path.exists(output_master_flat_path):
                try: os.remove(output_master_flat_path)
                except Exception: pass

    final_status = "
".join(status_messages) if status_messages else "Flat processing completed."
    if not new_master_flat_paths and not status_messages: # If no groups or all groups failed silently before message
        final_status = "No Preliminary Master FLATs were successfully generated or no valid flats found."
    elif not new_master_flat_paths: # Some errors occurred, but no successes
        final_status += "
No Preliminary Master FLATs were successfully generated."


    flat_paths_display_text = "Generated Preliminary Master FLATs:
" + "
".join([f"{filt}: {p}" for filt, p in new_master_flat_paths.items()])
    return final_status, new_master_flat_paths, gr.Textbox(value=flat_paths_display_text, label="Generated Prelim. Master FLAT Paths", visible=True, interactive=False)

def handle_upload_master_bias(uploaded_master_bias_file_obj, current_master_bias_path_state):
    if not uploaded_master_bias_file_obj:
        return "No Master BIAS file provided for upload.", current_master_bias_path_state, gr.File(visible=False)

    output_dir = "masters_output"
    os.makedirs(output_dir, exist_ok=True)

    destination_filename = "uploaded_master_bias.fits" # Using a fixed name
    destination_path = os.path.join(output_dir, destination_filename)

    try:
        # uploaded_master_bias_file_obj.name is the temporary path of the uploaded file
        shutil.copy(uploaded_master_bias_file_obj.name, destination_path)
        status_msg = f"Uploaded Master BIAS saved to: {destination_path}"
        print(status_msg)
        return status_msg, destination_path, gr.File(value=destination_path, label=f"Download {destination_filename}", visible=True, interactive=True)
    except Exception as e:
        error_msg = f"Error copying uploaded Master BIAS: {e}"
        print(error_msg)
        return error_msg, current_master_bias_path_state, gr.File(visible=False)

def handle_upload_master_darks(uploaded_master_darks_list, current_master_dark_paths_state):
    if not uploaded_master_darks_list:
        return "No Master DARK files provided for upload.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)

    output_dir = "masters_output"
    os.makedirs(output_dir, exist_ok=True)

    new_master_dark_paths = (current_master_dark_paths_state or {}).copy()
    status_messages = []

    for file_obj in uploaded_master_darks_list:
        try:
            header = get_fits_header(file_obj.name)
            exp_time_key = "unknown_exptime"
            if header:
                exptime_val = header.get('EXPTIME', header.get('EXPOSURE'))
                if exptime_val is not None:
                    try:
                        exp_time_key = str(float(exptime_val)).replace('.', 'p') + "s"
                    except ValueError:
                        exp_time_key = f"invalid_exptime_{os.path.basename(file_obj.name).split('.')[0]}"
            else:
                 exp_time_key = f"noheader_{os.path.basename(file_obj.name).split('.')[0]}"

            base_name_part = os.path.basename(file_obj.name)
            safe_base_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name_part)
            destination_filename = f"uploaded_master_dark_{exp_time_key}_{safe_base_name}"
            destination_path = os.path.join(output_dir, destination_filename)

            shutil.copy(file_obj.name, destination_path)
            new_master_dark_paths[exp_time_key] = destination_path # Could overwrite if same key, desired for updates
            status_messages.append(f"Uploaded Master DARK '{base_name_part}' (key: {exp_time_key}) saved to: {destination_path}")
            print(f"Uploaded Master DARK '{base_name_part}' saved to {destination_path} with key {exp_time_key}")

        except Exception as e:
            error_msg = f"Error copying uploaded Master DARK {file_obj.name}: {e}"
            status_messages.append(error_msg)
            print(error_msg)

    final_status = "
".join(status_messages) if status_messages else "No files processed or error."
    if not new_master_dark_paths and not status_messages : # No files processed, no errors
        final_status = "No dark files were processed."
    elif not new_master_dark_paths and status_messages: # Only errors
        pass # final_status already has the errors
    elif not status_messages and new_master_dark_paths : # All success
        final_status = "All dark files uploaded successfully."


    dark_paths_display_text = "Uploaded/Updated Master DARKs:
" + "
".join([f"{exp}: {p}" for exp, p in new_master_dark_paths.items()])
    return final_status, new_master_dark_paths, gr.Textbox(value=dark_paths_display_text, label="Uploaded Master DARK Paths", visible=True, interactive=False)

def handle_upload_master_flats(uploaded_master_flats_list, current_master_flat_paths_state):
    if not uploaded_master_flats_list:
        return "No Master FLAT files provided for upload.", current_master_flat_paths_state or {}, gr.Textbox(visible=False)

    output_dir = "masters_output"
    os.makedirs(output_dir, exist_ok=True)

    new_master_flat_paths = (current_master_flat_paths_state or {}).copy()
    status_messages = []

    for file_obj in uploaded_master_flats_list:
        try:
            header = get_fits_header(file_obj.name)
            filter_key = "unknown_filter"
            if header:
                filter_val = header.get('FILTER', header.get('FILTER1', header.get('FILTNAME')))
                if filter_val is not None:
                    filter_key = str(filter_val).strip().replace(" ", "_")
                    if not filter_key: filter_key = "empty_filter"
            else:
                filter_key = f"noheader_{os.path.basename(file_obj.name).split('.')[0]}"

            base_name_part = os.path.basename(file_obj.name)
            safe_base_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name_part)
            destination_filename = f"uploaded_master_flat_{filter_key}_{safe_base_name}"
            destination_path = os.path.join(output_dir, destination_filename)

            shutil.copy(file_obj.name, destination_path)
            new_master_flat_paths[filter_key] = destination_path # Overwrites if same filter key, desired
            status_messages.append(f"Uploaded Master FLAT '{base_name_part}' (key: {filter_key}) saved to: {destination_path}")
            print(f"Uploaded Master FLAT '{base_name_part}' saved to {destination_path} with key {filter_key}")

        except Exception as e:
            error_msg = f"Error copying uploaded Master FLAT {file_obj.name}: {e}"
            status_messages.append(error_msg)
            print(error_msg)

    final_status = "
".join(status_messages) if status_messages else "No files processed or error."
    if not new_master_flat_paths and not status_messages :
        final_status = "No flat files were processed."
    elif not new_master_flat_paths and status_messages:
        pass
    elif not status_messages and new_master_flat_paths :
        final_status = "All flat files uploaded successfully."

    flat_paths_display_text = "Uploaded/Updated Master FLATs:
" + "
".join([f"{filt}: {p}" for filt, p in new_master_flat_paths.items()])
    return final_status, new_master_flat_paths, gr.Textbox(value=flat_paths_display_text, label="Uploaded Master FLAT Paths", visible=True, interactive=False)

# Dummy handler for Tab 4 button (actual handler to be implemented in a later step)
# Renamed handler for Tab 4
def handle_tab4_photometry(
    tab4_b_file_obj, tab4_v_file_obj,
    tab4_std_star_file_obj, tab4_std_b_mag_str, tab4_std_v_mag_str,
    tab4_roi_str, tab4_k_value_str,
    master_bias_path_state, master_dark_paths_state, master_flat_paths_state, # States from Tab 1
    request: gr.Request  # Added request: gr.Request type hint
):
    status_messages = ["Starting Tab 4 Photometry Analysis..."]
    # Ensure all globally needed functions like load_fits_data, get_fits_header,
    # save_fits_data, correct_light_frame are accessible (imported at module level in app.py)

    if not tab4_b_file_obj or not tab4_v_file_obj:
        status_messages.append("Error: Both B-filter and V-filter LIGHT frames must be uploaded.")
        return "
".join(status_messages), None, None, None # status, df, preview_img, csv_dl

    # Ensure master_dark_paths_state and master_flat_paths_state are dictionaries
    if master_dark_paths_state is None: master_dark_paths_state = {}
    if master_flat_paths_state is None: master_flat_paths_state = {}


    raw_b_path = tab4_b_file_obj.name
    raw_v_path = tab4_v_file_obj.name
    status_messages.append(f"B-frame: {os.path.basename(raw_b_path)}, V-frame: {os.path.basename(raw_v_path)}")

    b_data_check = load_fits_data(raw_b_path) # Check data directly, not just assign
    b_header = get_fits_header(raw_b_path)
    v_data_check = load_fits_data(raw_v_path) # Check data directly
    v_header = get_fits_header(raw_v_path)

    if b_data_check is None or b_header is None:
        status_messages.append(f"Error: Could not load B-filter frame or its header from {raw_b_path}.")
        return "
".join(status_messages), None, None, None
    if v_data_check is None or v_header is None:
        status_messages.append(f"Error: Could not load V-filter frame or its header from {raw_v_path}.")
        return "
".join(status_messages), None, None, None
    status_messages.append("B and V LIGHT frames loaded successfully.")

    if not master_bias_path_state or not os.path.exists(master_bias_path_state):
        status_messages.append("Error: Master BIAS not found or path invalid. Please prepare/upload in Tab 1.")
        return "
".join(status_messages), None, None, None
    master_bias_data = load_fits_data(master_bias_path_state)
    if master_bias_data is None:
        status_messages.append(f"Error: Failed to load Master BIAS from {master_bias_path_state}.")
        return "
".join(status_messages), None, None, None
    status_messages.append(f"Master BIAS loaded (shape {master_bias_data.shape}).")

    temp_dir = os.path.join("masters_output", "temp_final_flats_tab4")
    calibrated_dir = os.path.join("calibrated_lights_output", "tab4_corrected")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(calibrated_dir, exist_ok=True)

    corrected_b_path = None
    try:
        b_exptime_val = b_header.get('EXPTIME', b_header.get('EXPOSURE'))
        b_filter_name = b_header.get('FILTER', b_header.get('FILTER1', b_header.get('FILTNAME')))
        if b_exptime_val is None: raise ValueError("B-frame missing EXPTIME/EXPOSURE keyword.")
        if b_filter_name is None: raise ValueError("B-frame missing FILTER/FILTER1/FILTNAME keyword.")

        b_exptime_float = float(b_exptime_val)
        b_filter_key = str(b_filter_name).strip().replace(" ", "_")

        b_dark_exp_key = str(b_exptime_float).replace('.', 'p') + "s"
        b_master_dark_path = master_dark_paths_state.get(b_dark_exp_key)
        b_master_dark_data = None # Define before conditional assignment
        if b_master_dark_path and os.path.exists(b_master_dark_path):
            b_master_dark_data = load_fits_data(b_master_dark_path)
            if b_master_dark_data is None: status_messages.append(f"Warning: Failed to load B-DARK from {b_master_dark_path} for exptime {b_exptime_float}s. Proceeding without dark correction for B-flat.")
            else: status_messages.append(f"Using B-DARK {b_master_dark_path} for B-flat processing.")
        else: status_messages.append(f"Warning: B-DARK for exp {b_exptime_float}s (key: {b_dark_exp_key}) not found in {list(master_dark_paths_state.keys())}. Proceeding without dark correction for B-flat.")

        b_prelim_flat_path = master_flat_paths_state.get(b_filter_key)
        path_to_final_b_flat = None
        if b_prelim_flat_path and os.path.exists(b_prelim_flat_path):
            b_prelim_flat_data = load_fits_data(b_prelim_flat_path)
            if b_prelim_flat_data is not None:
                if b_prelim_flat_data.shape != master_bias_data.shape: raise ValueError(f"B-PrelimFlat shape {b_prelim_flat_data.shape} mismatch with MasterBias {master_bias_data.shape}")
                flat_temp1 = b_prelim_flat_data - master_bias_data
                flat_temp2 = flat_temp1
                if b_master_dark_data is not None:
                    if flat_temp1.shape != b_master_dark_data.shape: raise ValueError(f"B-Flat(bias-sub) shape {flat_temp1.shape} mismatch with B-Dark {b_master_dark_data.shape}")
                    flat_temp2 = flat_temp1 - b_master_dark_data

                median_flat_temp2 = np.median(flat_temp2)
                if median_flat_temp2 == 0: raise ValueError("Median of processed B-FLAT (after bias/dark) is zero.")
                final_b_flat_data = flat_temp2 / median_flat_temp2

                path_to_final_b_flat = os.path.join(temp_dir, f"final_B_flat_{b_filter_key}.fits")
                # Try to get header from prelim flat, or create minimal one
                temp_flat_header = get_fits_header(b_prelim_flat_path) if get_fits_header(b_prelim_flat_path) else fits.Header()
                temp_flat_header['HISTORY'] = 'Generated final B-flat for Tab 4 (bias/dark subtracted, normalized)'
                save_fits_data(path_to_final_b_flat, final_b_flat_data, header=temp_flat_header)
                status_messages.append(f"Generated temporary final B-FLAT: {path_to_final_b_flat}")
            else: status_messages.append(f"Warning: Failed to load Prelim B-FLAT from {b_prelim_flat_path}. B-frame will not be flat-fielded.")
        else: status_messages.append(f"Warning: Prelim B-FLAT for filter '{b_filter_key}' not found in {list(master_flat_paths_state.keys())}. B-frame will not be flat-fielded.")

        base_b_name = os.path.splitext(os.path.basename(raw_b_path))[0]
        corrected_b_path_val = os.path.join(calibrated_dir, f"{base_b_name}_cal_B.fits")
        # Use the specific master dark for this B-frame if available, otherwise None
        actual_b_master_dark_for_light = b_master_dark_path if (b_master_dark_path and os.path.exists(b_master_dark_path)) else None
        if correct_light_frame(raw_b_path, corrected_b_path_val, master_bias_path_state, actual_b_master_dark_for_light, path_to_final_b_flat):
            status_messages.append(f"B-frame corrected: {corrected_b_path_val}")
            corrected_b_path = corrected_b_path_val # Store path for later use
        else: status_messages.append(f"Error: Failed to correct B-frame {raw_b_path}.")
        # Clean up temp final B-flat
        if path_to_final_b_flat and os.path.exists(path_to_final_b_flat):
            try: os.remove(path_to_final_b_flat)
            except Exception as e_rem_b: status_messages.append(f"Warning: Could not remove temp B-flat {path_to_final_b_flat}: {e_rem_b}")
    except Exception as e: status_messages.append(f"Error during B-frame processing: {str(e)}")


    corrected_v_path = None # Initialize for V-frame
    try:
        v_exptime_val = v_header.get('EXPTIME', v_header.get('EXPOSURE'))
        v_filter_name = v_header.get('FILTER', v_header.get('FILTER1', v_header.get('FILTNAME')))
        if v_exptime_val is None: raise ValueError("V-frame missing EXPTIME/EXPOSURE keyword.")
        if v_filter_name is None: raise ValueError("V-frame missing FILTER/FILTER1/FILTNAME keyword.")

        v_exptime_float = float(v_exptime_val)
        v_filter_key = str(v_filter_name).strip().replace(" ", "_")
        v_dark_exp_key = str(v_exptime_float).replace('.', 'p') + "s"
        v_master_dark_path = master_dark_paths_state.get(v_dark_exp_key)
        v_master_dark_data = None # Define before conditional assignment
        if v_master_dark_path and os.path.exists(v_master_dark_path):
            v_master_dark_data = load_fits_data(v_master_dark_path)
            if v_master_dark_data is None: status_messages.append(f"Warning: Failed to load V-DARK from {v_master_dark_path} for exptime {v_exptime_float}s. Proceeding without dark correction for V-flat.")
            else: status_messages.append(f"Using V-DARK {v_master_dark_path} for V-flat processing.")
        else: status_messages.append(f"Warning: V-DARK for exp {v_exptime_float}s (key: {v_dark_exp_key}) not found in {list(master_dark_paths_state.keys())}. Proceeding without dark correction for V-flat.")

        v_prelim_flat_path = master_flat_paths_state.get(v_filter_key)
        path_to_final_v_flat = None # Initialize path
        if v_prelim_flat_path and os.path.exists(v_prelim_flat_path):
            v_prelim_flat_data = load_fits_data(v_prelim_flat_path)
            if v_prelim_flat_data is not None:
                if v_prelim_flat_data.shape != master_bias_data.shape: raise ValueError(f"V-PrelimFlat shape {v_prelim_flat_data.shape} mismatch with MasterBias {master_bias_data.shape}")
                flat_temp1_v = v_prelim_flat_data - master_bias_data
                flat_temp2_v = flat_temp1_v
                if v_master_dark_data is not None:
                    if flat_temp1_v.shape != v_master_dark_data.shape: raise ValueError(f"V-Flat(bias-sub) shape {flat_temp1_v.shape} mismatch with V-Dark {v_master_dark_data.shape}")
                    flat_temp2_v = flat_temp1_v - v_master_dark_data

                median_flat_temp2_v = np.median(flat_temp2_v)
                if median_flat_temp2_v == 0: raise ValueError("Median of processed V-FLAT (after bias/dark) is zero.")
                final_v_flat_data = flat_temp2_v / median_flat_temp2_v

                path_to_final_v_flat = os.path.join(temp_dir, f"final_V_flat_{v_filter_key}.fits")
                temp_flat_header_v = get_fits_header(v_prelim_flat_path) if get_fits_header(v_prelim_flat_path) else fits.Header()
                temp_flat_header_v['HISTORY'] = 'Generated final V-flat for Tab 4 (bias/dark subtracted, normalized)'
                save_fits_data(path_to_final_v_flat, final_v_flat_data, header=temp_flat_header_v)
                status_messages.append(f"Generated temporary final V-FLAT: {path_to_final_v_flat}")
            else: status_messages.append(f"Warning: Failed to load Prelim V-FLAT from {v_prelim_flat_path}. V-frame will not be flat-fielded.")
        else: status_messages.append(f"Warning: Prelim V-FLAT for filter '{v_filter_key}' not found in {list(master_flat_paths_state.keys())}. V-frame will not be flat-fielded.")

        base_v_name = os.path.splitext(os.path.basename(raw_v_path))[0]
        corrected_v_path_val = os.path.join(calibrated_dir, f"{base_v_name}_cal_V.fits")
        actual_v_master_dark_for_light = v_master_dark_path if (v_master_dark_path and os.path.exists(v_master_dark_path)) else None
        if correct_light_frame(raw_v_path, corrected_v_path_val, master_bias_path_state, actual_v_master_dark_for_light, path_to_final_v_flat):
            status_messages.append(f"V-frame corrected: {corrected_v_path_val}")
            corrected_v_path = corrected_v_path_val # Store path for later use
        else: status_messages.append(f"Error: Failed to correct V-frame {raw_v_path}.")
        # Clean up temp final V-flat
        if path_to_final_v_flat and os.path.exists(path_to_final_v_flat):
            try: os.remove(path_to_final_v_flat)
            except Exception as e_rem_v: status_messages.append(f"Warning: Could not remove temp V-flat {path_to_final_v_flat}: {e_rem_v}")
    except Exception as e: status_messages.append(f"Error during V-frame processing: {str(e)}")

    if not corrected_b_path or not corrected_v_path:
        status_messages.append("One or both LIGHT frames failed to calibrate. Cannot proceed with photometry.")
        return "
".join(status_messages), None, None, None

    status_messages.append("
Photometry Part 1 (Calibration) finished. Corrected frames (if successful):")
    status_messages.append(f"Corrected B: {corrected_b_path if corrected_b_path else 'N/A'}")
    status_messages.append(f"Corrected V: {corrected_v_path if corrected_v_path else 'N/A'}")
    status_messages.append("
Source detection, photometry, and catalog matching not yet implemented in this part.")

    # For now, return None for DataFrame, preview image, and CSV download path
    # Preview could show the corrected B-frame if successful
    preview_image_path = corrected_b_path # or None if it failed

    return "
".join(status_messages), None, preview_image_path, None

master_dark_paths_state = gr.State({})
master_flat_paths_state = gr.State({})

            gr.Markdown("## Create Master Calibration Frames or Upload Existing Ones")

                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("### Option 1: Generate Master Frames from Raw Files")
                                bias_uploads = gr.Files(label="Upload BIAS Frames (FITS)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab1_bias_uploads")
                                dark_uploads = gr.Files(label="Upload DARK Frames (FITS)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab1_dark_uploads")
                                flat_uploads = gr.Files(label="Upload FLAT Frames (FITS)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab1_flat_uploads")

                                with gr.Row():
                                    generate_master_bias_button = gr.Button("Generate Master BIAS", elem_id="tab1_gen_mbias_btn")
                                    generate_master_dark_button = gr.Button("Generate Master DARKs", elem_id="tab1_gen_mdark_btn")
                                    generate_master_flat_button = gr.Button("Generate Prelim. Master FLATs", elem_id="tab1_gen_mflat_btn")

                            with gr.Column(scale=1):
                                gr.Markdown("### Option 2: Upload Existing Master Frames")
                                upload_master_bias = gr.File(label="Upload Master BIAS (FITS)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab1_upload_mbias")
                                upload_master_darks = gr.Files(label="Upload Master DARKs (FITS, one per exposure)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab1_upload_mdarks")
                                upload_master_flats = gr.Files(label="Upload Prelim. Master FLATs (FITS, one per filter)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab1_upload_mflats")

                        gr.Markdown("### Processing Status & Results")
                        tab1_status_display = gr.Textbox(label="Status", interactive=False, lines=5, elem_id="tab1_status_disp")

                        with gr.Row():
                            # These will be updated to be File components for download later
                            download_master_bias = gr.File(label="Download Master BIAS", interactive=False, visible=False, elem_id="tab1_dl_mbias") # Placeholder
                            download_master_darks_display = gr.Textbox(label="Generated Master DARK Paths", interactive=False, visible=False, lines=3, elem_id="tab1_dl_mdarks_txt") # Placeholder
                            download_master_flats_display = gr.Textbox(label="Generated Prelim. Master FLAT Paths", interactive=False, visible=False, lines=3, elem_id="tab1_dl_mflats_txt") # Placeholder

                        # Connect buttons to handlers
                        generate_master_bias_button.click(fn=handle_generate_master_bias, inputs=[bias_uploads, master_bias_path_state], outputs=[tab1_status_display, master_bias_path_state, download_master_bias])
                        generate_master_dark_button.click(fn=handle_generate_master_dark, inputs=[dark_uploads, master_bias_path_state, master_dark_paths_state], outputs=[tab1_status_display, master_dark_paths_state, download_master_darks_display])
                        generate_master_flat_button.click(fn=handle_generate_master_flat, inputs=[flat_uploads, master_flat_paths_state], outputs=[tab1_status_display, master_flat_paths_state, download_master_flats_display])

                        upload_master_bias.upload(fn=handle_upload_master_bias, inputs=[upload_master_bias, master_bias_path_state], outputs=[tab1_status_display, master_bias_path_state, download_master_bias])
                        upload_master_darks.upload(fn=handle_upload_master_darks, inputs=[upload_master_darks, master_dark_paths_state], outputs=[tab1_status_display, master_dark_paths_state, download_master_darks_display])
                        upload_master_flats.upload(fn=handle_upload_master_flats, inputs=[upload_master_flats, master_flat_paths_state], outputs=[tab1_status_display, master_flat_paths_state, download_master_flats_display])
            # TODO: Add UI components for Tab 1

        with gr.TabItem("LIGHT Frame Correction (Tab 2)"):
            gr.Markdown("## Calibrate Raw LIGHT Frames")

                        with gr.Row():
                            with gr.Column(scale=1):
                                light_frame_uploads = gr.Files(label="Upload Raw LIGHT Frames (FITS)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab2_light_uploads")
                                # Assuming master_bias_path_state, master_dark_paths_state, master_flat_paths_state are accessible
                                # from the parent gr.Blocks() context where they are defined (in Tab 1).
                                calibrate_lights_button = gr.Button("Calibrate Uploaded LIGHT Frames", elem_id="tab2_calibrate_btn")

                            with gr.Column(scale=2):
                                tab2_status_display = gr.Textbox(label="Calibration Status", interactive=False, lines=10, elem_id="tab2_status_disp") # Increased lines

                        gr.Markdown("### Calibrated Image Preview") # Singular for now
                        # Output for a single image preview (e.g., the first calibrated image)
                        calibrated_light_preview = gr.Image(label="Calibrated LIGHT Frame Preview (PNG)", type="filepath", interactive=False, elem_id="tab2_preview_img", height=400, visible=False)

                        gr.Markdown("### Download Calibrated LIGHT Frame") # Singular for now
                        # Output for downloading a single calibrated FITS file (e.g., the first one, or a ZIP if multiple)
                        download_calibrated_light = gr.File(label="Download Calibrated LIGHT Frame (FITS)", interactive=False, visible=False, elem_id="tab2_download_fits")

                        # Connect button to handler
                        # The request object is implicitly passed if the handler includes it in its signature.
                        # Gradio checks the handler's signature.
                        calibrate_lights_button.click(
                            fn=handle_calibrate_lights,
                            inputs=[light_frame_uploads, master_bias_path_state, master_dark_paths_state, master_flat_paths_state],
                            outputs=[tab2_status_display, calibrated_light_preview, download_calibrated_light]
                        )
            # TODO: Add UI components for Tab 2

        with gr.TabItem("Extinction Coefficient (Tab 3)"):
            gr.Markdown("## Calculate Atmospheric Extinction Coefficient (k)")
            gr.Markdown(
                "Enter comma-separated airmass values and their corresponding instrumental magnitudes. "
                "The relationship is `m = m0 - kX`, where `m` is observed magnitude, `m0` is magnitude at zero airmass, "
                "`k` is the extinction coefficient, and `X` is airmass."
            )
            with gr.Row():
                airmass_input = gr.Textbox(label="Airmass Values (comma-separated)", placeholder="e.g., 1.0, 1.5, 2.0, 2.5, 3.0")
                magnitude_input = gr.Textbox(label="Instrumental Magnitudes (comma-separated)", placeholder="e.g., 12.5, 12.8, 13.1, 13.4, 13.7")

            calculate_button_tab3 = gr.Button("Calculate Extinction Coefficient")

            with gr.Row():
                k_output = gr.Textbox(label="Extinction Coefficient (k)", interactive=False)
                k_err_output = gr.Textbox(label="Uncertainty in k (k_err)", interactive=False)
                m0_output = gr.Textbox(label="Magnitude at Zero Airmass (m0)", interactive=False)

            error_output_tab3 = gr.Textbox(label="Status/Errors", interactive=False)

            calculate_button_tab3.click(
                fn=handle_extinction_calculation,
                inputs=[airmass_input, magnitude_input],
                outputs=[k_output, k_err_output, m0_output, error_output_tab3]
            )

        with gr.TabItem("Detailed Photometry (Tab 4)"):

            gr.Markdown("## Tab 4: Detailed Photometry and Catalog Analysis")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input LIGHT Frames")
                    tab4_b_frame_upload = gr.File(label="Upload B-filter LIGHT Frame (FITS)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab4_b_upload")
                    tab4_v_frame_upload = gr.File(label="Upload V-filter LIGHT Frame (FITS)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab4_v_upload")

                    gr.Markdown("### Region of Interest (ROI)")
                    tab4_roi_input = gr.Textbox(label="ROI (center_x, center_y, radius_pixels)", placeholder="e.g., 512,512,100 or leave blank for full image", elem_id="tab4_roi")

                    gr.Markdown("### Atmospheric Extinction")
                    tab4_k_value_input = gr.Textbox(label="Extinction Coefficient (k)", value="0.15", elem_id="tab4_k_val") # Default k value

                with gr.Column(scale=1):
                    gr.Markdown("### Optional: Standard Star Information")
                    tab4_std_star_fits_upload = gr.File(label="Upload Standard Star FITS File (optional)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab4_std_fits")
                    tab4_std_b_mag_input = gr.Textbox(label="Std. Star Known B Mag (e.g., 12.34)", placeholder="Required if Std FITS provided", elem_id="tab4_std_b")
                    tab4_std_v_mag_input = gr.Textbox(label="Std. Star Known V Mag (e.g., 12.01)", placeholder="Required if Std FITS provided", elem_id="tab4_std_v")

            with gr.Row():
                tab4_run_button = gr.Button("Run Photometry Analysis", elem_id="tab4_run_phot_btn", variant="primary")

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Results Table")
                    # Increased rows for better display, added wrap=True
                    tab4_results_table = gr.DataFrame(label="Photometry Results", headers=["ID", "X", "Y", "RA", "Dec", "InstrMag_B", "InstrMag_V", "StdMag_B", "StdMag_V", "B-V"], interactive=False, wrap=True, max_rows=10, overflow_row_behaviour='paginate', elem_id="tab4_results_df")
                    tab4_csv_download = gr.File(label="Download Results as CSV", interactive=False, visible=False, elem_id="tab4_csv_dl")
                with gr.Column(scale=1):
                    gr.Markdown("### Preview (B-filter)")
                    tab4_preview_image = gr.Image(label="B-filter Preview with Detections/ROI", type="filepath", interactive=False, height=400, visible=False, elem_id="tab4_preview_img_b")

            tab4_status_display = gr.Textbox(label="Status / Errors", lines=5, interactive=False, elem_id="tab4_status_text")

            # Dummy handler for the button, to be properly implemented later.
            # This is just to make the app runnable without error if button is clicked.
            def handle_run_photometry_analysis_dummy(b_frame, v_frame, roi, k_val, std_fits, std_b, std_v):
                return "Photometry analysis not fully implemented yet.", None, None, None

            tab4_run_button.click(
                fn=handle_run_photometry_analysis_dummy,
                inputs=[tab4_b_frame_upload, tab4_v_frame_upload, tab4_roi_input, tab4_k_value_input, tab4_std_star_fits_upload, tab4_std_b_mag_input, tab4_std_v_mag_input],
                outputs=[tab4_status_display, tab4_results_table, tab4_csv_download, tab4_preview_image]
            )
            # TODO: Add UI components for Tab 4

        with gr.TabItem("H-R Diagram (Tab 5)"):
            gr.Markdown("Placeholder for Tab 5: H-R Diagram Plotting")
            # TODO: Add UI components for Tab 5



# Handler for Tab 2: LIGHT Frame Correction
def handle_calibrate_lights(light_uploads_list, mbias_path, mdark_paths_dict, mflat_paths_dict, request: gr.Request):
    # request: gr.Request to potentially get session_hash for unique temp filenames if needed

    # Initial checks
    if not light_uploads_list:
        return "No LIGHT frames uploaded for calibration.", None, None # status, preview, download

    # For now, just acknowledge and list master paths from state
    status_messages = [f"Received {len(light_uploads_list)} LIGHT frames for calibration."]
    status_messages.append(f"Master BIAS path from state: {mbias_path}")
    status_messages.append(f"Master DARK paths from state: {mdark_paths_dict}") # This is a dict
    status_messages.append(f"Prelim. Master FLAT paths from state: {mflat_paths_dict}") # This is a dict
    status_messages.append("\n(Full calibration logic not yet implemented in this handler.)") # Escaped newline for string literal

    # Ensure the return signature matches what Gradio expects for the outputs
    # (status_textbox, image_preview_output, download_corrected_light_output)
    # For now, returning None for image and file outputs.
    return "\n".join(status_messages), None, None


if __name__ == '__main__':
    # Ensure necessary packages are installed
    try:
        import scipy
    except ImportError:
        print("SCIPY NOT INSTALLED. PLEASE INSTALL IT: pip install scipy")
        # In a real scenario, you might try to install it here or provide clearer instructions.

    astro_app.launch()
