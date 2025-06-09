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

    final_status = "\n".join(status_messages) if status_messages else "Processing completed."
    if not new_master_dark_paths:
        final_status += "\nNo Master DARKs were successfully generated."

    dark_paths_display_text = "Generated Master DARKs:\n" + "\n".join([f"{exp}: {p}" for exp, p in new_master_dark_paths.items()])
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

    final_status = "\n".join(status_messages) if status_messages else "Flat processing completed."
    if not new_master_flat_paths and not status_messages: # If no groups or all groups failed silently before message
        final_status = "No Preliminary Master FLATs were successfully generated or no valid flats found."
    elif not new_master_flat_paths: # Some errors occurred, but no successes
        final_status += "\nNo Preliminary Master FLATs were successfully generated."


    flat_paths_display_text = "Generated Preliminary Master FLATs:\n" + "\n".join([f"{filt}: {p}" for filt, p in new_master_flat_paths.items()])
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

    final_status = "\n".join(status_messages) if status_messages else "No files processed or error."
    if not new_master_dark_paths and not status_messages : # No files processed, no errors
        final_status = "No dark files were processed."
    elif not new_master_dark_paths and status_messages: # Only errors
        pass # final_status already has the errors
    elif not status_messages and new_master_dark_paths : # All success
        final_status = "All dark files uploaded successfully."


    dark_paths_display_text = "Uploaded/Updated Master DARKs:\n" + "\n".join([f"{exp}: {p}" for exp, p in new_master_dark_paths.items()])
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

    final_status = "\n".join(status_messages) if status_messages else "No files processed or error."
    if not new_master_flat_paths and not status_messages :
        final_status = "No flat files were processed."
    elif not new_master_flat_paths and status_messages:
        pass
    elif not status_messages and new_master_flat_paths :
        final_status = "All flat files uploaded successfully."

    flat_paths_display_text = "Uploaded/Updated Master FLATs:\n" + "\n".join([f"{filt}: {p}" for filt, p in new_master_flat_paths.items()])
    return final_status, new_master_flat_paths, gr.Textbox(value=flat_paths_display_text, label="Uploaded Master FLAT Paths", visible=True, interactive=False)

with gr.Blocks() as astro_app:
    gr.Markdown("# Astro App")

    with gr.Tabs():
        with gr.TabItem("Master Frame Generation (Tab 1)"):
            master_bias_path_state = gr.State(None)
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
            gr.Markdown("Placeholder for Tab 4: Detailed Photometry and Catalog Analysis")
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
