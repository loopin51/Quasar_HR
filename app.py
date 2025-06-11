import gradio as gr
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.visualization import ZScaleInterval, ImageNormalize, PercentileInterval # For PNG preview
from astropy.wcs import WCS
from astropy.table import Table # DAOStarFinder might return this
from astropy.coordinates import SkyCoord # For cross-matching
import astropy.units as u # For SkyCoord units
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats


from utils import load_fits_data, get_fits_header, save_fits_data
from tab1_functions import create_master_frame
from tab2_functions import correct_light_frame
# from tab3_functions import calculate_extinction_coefficient # Old function, will be replaced by logic in app.py
from tab4_functions import perform_photometry # May use for simplified photometry if needed
from tab5_functions import plot_hr_diagram
from scipy.stats import linregress # For Tab 3
import time # For unique filenames and history
import re # For sanitizing filenames in upload handlers

# --- Helper for cleanup in handlers ---
def _try_remove(file_path):
    if file_path and os.path.exists(file_path):
        try: os.remove(file_path)
        except Exception as e: print(f"Warning: Could not remove temp file {file_path}: {e}")

def _try_rmdir_if_empty(dir_path):
    if dir_path and os.path.exists(dir_path) and not os.listdir(dir_path): # Check if empty
        try: shutil.rmtree(dir_path) # Use shutil.rmtree for directories
        except Exception as e: print(f"Warning: Could not remove temp dir {dir_path}: {e}")

# --- Handler for Tab 1: Master BIAS ---
def handle_generate_master_bias(bias_uploads_list, current_master_bias_path_state):
    if not bias_uploads_list:
        return "No BIAS files uploaded. Cannot generate Master BIAS.", current_master_bias_path_state, gr.File(visible=False)
    output_dir = "masters_output"
    if not os.path.exists(output_dir):
        try: os.makedirs(output_dir); print(f"Created output directory: {output_dir}")
        except OSError as e: return f"Error creating output directory {output_dir}: {e}", current_master_bias_path_state, gr.File(visible=False)
    output_master_bias_path = os.path.join(output_dir, "master_bias.fits")
    uploaded_file_paths = [f.name for f in bias_uploads_list]
    # print(f"Calling create_master_frame for BIAS with files: {uploaded_file_paths} to output: {output_master_bias_path}") # Verbose
    try:
        success = create_master_frame(file_paths=uploaded_file_paths, output_path=output_master_bias_path, method="average", frame_type="BIAS")
    except Exception as e_create: return f"Exception during Master BIAS generation: {e_create}", current_master_bias_path_state, gr.File(visible=False)
    if success:
        # print(f"Master BIAS generated successfully: {output_master_bias_path}") # Verbose
        return f"Master BIAS generated: {output_master_bias_path}", output_master_bias_path, gr.File(value=output_master_bias_path, label=f"Download Master BIAS ({os.path.basename(output_master_bias_path)})", visible=True, interactive=True)
    else:
        _try_remove(output_master_bias_path)
        return "Failed to generate Master BIAS. Check console/logs.", current_master_bias_path_state, gr.File(visible=False)

# --- Handler for Tab 1: Master DARKs ---
def handle_generate_master_dark(dark_uploads_list, master_bias_path, current_master_dark_paths_state):
    # print(f"Master BIAS path received in handle_generate_master_dark: {master_bias_path}") # Verbose
    if not dark_uploads_list: return "No DARK files uploaded.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)
    if not master_bias_path or not os.path.exists(master_bias_path): return "Master BIAS not available or path invalid.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)
    master_bias_data_for_darks = load_fits_data(master_bias_path)
    if master_bias_data_for_darks is None: return f"Failed to load Master BIAS data from {master_bias_path}.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)
    output_dir = "masters_output"; temp_subtracted_dark_dir = os.path.join(output_dir, "temp_subtracted_darks")
    os.makedirs(temp_subtracted_dark_dir, exist_ok=True) # print(f"Ensured temp directory exists: {temp_subtracted_dark_dir}") # Verbose
    grouped_darks_by_exp = {}; raw_dark_paths = [f.name for f in dark_uploads_list]
    for dark_path in raw_dark_paths:
        header = get_fits_header(dark_path)
        if not header: print(f"Warning: Could not read header for {dark_path}. Skipping."); continue
        exptime_val = header.get('EXPTIME', header.get('EXPOSURE'))
        if exptime_val is None: print(f"Warning: Exposure time not found for {dark_path}. Skipping."); continue
        try:
            exptime_float = float(exptime_val)
            if exptime_float <= 0: print(f"Warning: Invalid exposure time {exptime_float} in {dark_path}. Skipping."); continue
        except ValueError: print(f"Warning: Could not convert exposure time '{exptime_val}' for {dark_path}. Skipping."); continue
        exptime_str = str(exptime_float).replace('.', 'p')
        if exptime_str not in grouped_darks_by_exp: grouped_darks_by_exp[exptime_str] = []
        grouped_darks_by_exp[exptime_str].append(dark_path)
    if not grouped_darks_by_exp: return "No DARK frames with valid exposure times.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)
    new_master_dark_paths = (current_master_dark_paths_state or {}).copy(); status_messages = []
    for exptime_str, file_list in grouped_darks_by_exp.items():
        if not file_list: continue
        # print(f"Processing DARKs for exposure: {exptime_str} ({len(file_list)} files)"); # Verbose
        bias_subtracted_dark_paths = []
        for i, raw_dark_path_iter in enumerate(file_list):
            raw_dark_data = load_fits_data(raw_dark_path_iter)
            if raw_dark_data is None: status_messages.append(f"Warning: Failed to load raw DARK {raw_dark_path_iter}."); continue
            if raw_dark_data.shape != master_bias_data_for_darks.shape: status_messages.append(f"Warning: Shape mismatch DARK {raw_dark_path_iter} vs BIAS."); continue
            subtracted_data = raw_dark_data - master_bias_data_for_darks
            temp_dark_filename = f"bias_subtracted_dark_{exptime_str}_{i}.fits"; temp_path = os.path.join(temp_subtracted_dark_dir, temp_dark_filename)
            original_header = get_fits_header(raw_dark_path_iter)
            if original_header: original_header.add_history(f"BIAS subtracted using {os.path.basename(master_bias_path)}")
            if save_fits_data(temp_path, subtracted_data, header=original_header): bias_subtracted_dark_paths.append(temp_path)
            else: status_messages.append(f"Warning: Failed to save BIAS-subtracted DARK {temp_path}.")
        if not bias_subtracted_dark_paths: status_messages.append(f"No valid BIAS-subtracted DARKs for exp {exptime_str}."); continue
        output_master_dark_path = os.path.join(output_dir, f"master_dark_{exptime_str}.fits")
        success = create_master_frame(file_paths=bias_subtracted_dark_paths, output_path=output_master_dark_path, method="average", frame_type=f"DARK_{exptime_str}")
        if success: new_master_dark_paths[exptime_str] = output_master_dark_path; status_messages.append(f"Master DARK {exptime_str} gen: {output_master_dark_path}")
        else: status_messages.append(f"Failed Master DARK gen for {exptime_str}."); _try_remove(output_master_dark_path)
        for temp_f_path in bias_subtracted_dark_paths: _try_remove(temp_f_path)
    _try_rmdir_if_empty(temp_subtracted_dark_dir)
    final_status = "\\n".join(status_messages) if status_messages else "Processing completed."
    if not new_master_dark_paths and not status_messages: final_status += "\\nNo Master DARKs generated."
    elif not new_master_dark_paths: final_status += "\\nNo Master DARKs were successfully generated."
    # dark_paths_display_text = "Generated Master DARKs:\\n" + "\\n".join([f"{exp}: {p}" for exp, p in new_master_dark_paths.items()])
    dark_file_paths_list = list(new_master_dark_paths.values()) if new_master_dark_paths else []
    return final_status, new_master_dark_paths, gr.Files(value=dark_file_paths_list, visible=len(dark_file_paths_list) > 0)

# --- Handler for Tab 1: Master FLATs ---
def handle_generate_master_flat(flat_uploads_list, current_master_flat_paths_state):
    if not flat_uploads_list: return "No FLAT files uploaded.", current_master_flat_paths_state or {}, gr.Textbox(visible=False)
    output_dir = "masters_output"; os.makedirs(output_dir, exist_ok=True) # print(f"Ensured output directory exists: {output_dir}") # Verbose
    grouped_flats_by_filter = {}; raw_flat_paths = [f.name for f in flat_uploads_list]
    for flat_path in raw_flat_paths:
        header = get_fits_header(flat_path)
        if not header: print(f"Warning: Could not read header for {flat_path}. Skipping."); continue
        filter_name_val = header.get('FILTER', header.get('FILTER1', header.get('FILTNAME')))
        if filter_name_val is None: print(f"Warning: Filter name not found for {flat_path}. Skipping."); continue
        filter_name = str(filter_name_val).strip().replace(" ", "_")
        if not filter_name: print(f"Warning: Filter name empty for {flat_path}. Skipping."); continue
        if filter_name not in grouped_flats_by_filter: grouped_flats_by_filter[filter_name] = []
        grouped_flats_by_filter[filter_name].append(flat_path)
    if not grouped_flats_by_filter: return "No FLATs with valid filter names.", current_master_flat_paths_state or {}, gr.Textbox(visible=False)
    new_master_flat_paths = (current_master_flat_paths_state or {}).copy(); status_messages = []
    for filter_name_key, file_list in grouped_flats_by_filter.items():
        if not file_list: continue
        # print(f"Processing Prelim Master FLATs for filter: {filter_name_key} ({len(file_list)} files)") # Verbose
        output_master_flat_path = os.path.join(output_dir, f"prelim_master_flat_{filter_name_key}.fits")
        try:
            success = create_master_frame(file_paths=file_list, output_path=output_master_flat_path, method="median", frame_type=f"PRELIM_FLAT_{filter_name_key.upper()}")
        except Exception as e_create_flat: error_msg = f"Exception for Prelim FLAT {filter_name_key}: {e_create_flat}"; print(error_msg); status_messages.append(error_msg); success = False
        if success: new_master_flat_paths[filter_name_key] = output_master_flat_path; status_messages.append(f"Prelim Master FLAT '{filter_name_key}' gen: {output_master_flat_path}")
        else: status_messages.append(f"Failed Prelim Master FLAT for '{filter_name_key}'."); _try_remove(output_master_flat_path)
    final_status = "\\n".join(status_messages) if status_messages else "Flat processing completed."
    if not new_master_flat_paths and not status_messages: final_status = "No Prelim Master FLATs generated or no valid flats."
    elif not new_master_flat_paths: final_status += "\\nNo Prelim Master FLATs successfully generated."
    # flat_paths_display_text = "Generated Prelim. Master FLATs:\\n" + "\\n".join([f"{filt}: {p}" for filt, p in new_master_flat_paths.items()])
    flat_file_paths_list = list(new_master_flat_paths.values()) if new_master_flat_paths else []
    return final_status, new_master_flat_paths, gr.Files(value=flat_file_paths_list, visible=len(flat_file_paths_list) > 0)

# --- Handlers for Uploading Master Frames (Tab 1) ---
def handle_upload_master_bias(uploaded_master_bias_file_obj, current_master_bias_path_state):
    if not uploaded_master_bias_file_obj: return "No Master BIAS provided.", current_master_bias_path_state, gr.File(visible=False)
    output_dir = "masters_output"; os.makedirs(output_dir, exist_ok=True)
    destination_filename = "uploaded_master_bias.fits"; destination_path = os.path.join(output_dir, destination_filename)
    try:
        shutil.copy(uploaded_master_bias_file_obj.name, destination_path); status_msg = f"Uploaded Master BIAS: {destination_path}"; print(status_msg)
        return status_msg, destination_path, gr.File(value=destination_path, label=f"Download {destination_filename}", visible=True, interactive=True)
    except Exception as e: error_msg = f"Error copying Master BIAS: {e}"; print(error_msg); return error_msg, current_master_bias_path_state, gr.File(visible=False)

def handle_upload_master_darks(uploaded_master_darks_list, current_master_dark_paths_state):
    if not uploaded_master_darks_list: return "No Master DARKs provided.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)
    output_dir = "masters_output"; os.makedirs(output_dir, exist_ok=True)
    new_master_dark_paths = (current_master_dark_paths_state or {}).copy(); status_messages = []
    for file_obj in uploaded_master_darks_list:
        try:
            header = get_fits_header(file_obj.name); exp_time_key = "unknown_exptime"
            if header:
                exptime_val = header.get('EXPTIME', header.get('EXPOSURE'))
                if exptime_val is not None:
                    try: exp_time_key = str(float(exptime_val)).replace('.', 'p') + "s"
                    except ValueError: exp_time_key = f"invalid_exptime_{os.path.basename(file_obj.name).split('.')[0]}"
            else: exp_time_key = f"noheader_{os.path.basename(file_obj.name).split('.')[0]}"
            base_name_part = os.path.basename(file_obj.name); safe_base_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name_part) # Sanitize
            destination_filename = f"uploaded_master_dark_{exp_time_key}_{safe_base_name}"; destination_path = os.path.join(output_dir, destination_filename)
            shutil.copy(file_obj.name, destination_path); new_master_dark_paths[exp_time_key] = destination_path
            status_messages.append(f"Uploaded DARK '{base_name_part}' (key: {exp_time_key}): {destination_path}"); # print(f"Uploaded DARK '{base_name_part}' -> {destination_path} (key {exp_time_key})") # Verbose
        except Exception as e: error_msg = f"Error copying DARK {file_obj.name}: {e}"; status_messages.append(error_msg); print(error_msg)
    final_status = "\\n".join(status_messages) if status_messages else "No files processed."
    if not new_master_dark_paths and status_messages: final_status = "\\n".join(status_messages)
    elif not new_master_dark_paths and not status_messages: final_status = "No dark files processed."
    elif not status_messages and new_master_dark_paths: final_status = "All darks uploaded."
    # dark_paths_display_text = "Uploaded/Updated Master DARKs:\\n" + "\\n".join([f"{exp}: {p}" for exp, p in new_master_dark_paths.items()])
    dark_file_paths_list = list(new_master_dark_paths.values()) if new_master_dark_paths else []
    return final_status, new_master_dark_paths, gr.Files(value=dark_file_paths_list, visible=len(dark_file_paths_list) > 0)

def handle_upload_master_flats(uploaded_master_flats_list, current_master_flat_paths_state):
    if not uploaded_master_flats_list: return "No Master FLATs provided.", current_master_flat_paths_state or {}, gr.Textbox(visible=False)
    output_dir = "masters_output"; os.makedirs(output_dir, exist_ok=True)
    new_master_flat_paths = (current_master_flat_paths_state or {}).copy(); status_messages = []
    for file_obj in uploaded_master_flats_list:
        try:
            header = get_fits_header(file_obj.name); filter_key = "unknown_filter"
            if header:
                filter_val = header.get('FILTER', header.get('FILTER1', header.get('FILTNAME')))
                if filter_val is not None: filter_key = str(filter_val).strip().replace(" ", "_");
                if not filter_key: filter_key = "empty_filter"
            else: filter_key = f"noheader_{os.path.basename(file_obj.name).split('.')[0]}"
            base_name_part = os.path.basename(file_obj.name); safe_base_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name_part)
            destination_filename = f"uploaded_master_flat_{filter_key}_{safe_base_name}"; destination_path = os.path.join(output_dir, destination_filename)
            shutil.copy(file_obj.name, destination_path); new_master_flat_paths[filter_key] = destination_path
            status_messages.append(f"Uploaded FLAT '{base_name_part}' (key: {filter_key}): {destination_path}"); # print(f"Uploaded FLAT '{base_name_part}' -> {destination_path} (key {filter_key})") # Verbose
        except Exception as e: error_msg = f"Error copying FLAT {file_obj.name}: {e}"; status_messages.append(error_msg); print(error_msg)
    final_status = "\\n".join(status_messages) if status_messages else "No files processed."
    if not new_master_flat_paths and status_messages: final_status = "\\n".join(status_messages)
    elif not new_master_flat_paths and not status_messages: final_status = "No flat files processed."
    elif not status_messages and new_master_flat_paths: final_status = "All flats uploaded."
    # flat_paths_display_text = "Uploaded/Updated Master FLATs:\\n" + "\\n".join([f"{filt}: {p}" for filt, p in new_master_flat_paths.items()])
    flat_file_paths_list = list(new_master_flat_paths.values()) if new_master_flat_paths else []
    return final_status, new_master_flat_paths, gr.Files(value=flat_file_paths_list, visible=len(flat_file_paths_list) > 0)

# --- Helper function for PNG preview generation ---
def create_png_preview(fits_data, output_png_path, stretch_mode='zscale', percentile=99.5, min_val=None, max_val=None):
    if fits_data is None: return None
    plt.figure(figsize=(6, 6))
    if stretch_mode == 'zscale': norm = ImageNormalize(fits_data, interval=ZScaleInterval())
    elif stretch_mode == 'minmax':
        if min_val is None: min_val = np.nanmin(fits_data)
        if max_val is None: max_val = np.nanmax(fits_data)
        norm = ImageNormalize(fits_data, vmin=min_val, vmax=max_val)
    elif stretch_mode == 'percentile': norm = ImageNormalize(fits_data, interval=PercentileInterval(percentile))
    else: norm = ImageNormalize(fits_data, interval=ZScaleInterval())
    plt.imshow(fits_data, cmap='grey', origin='lower', norm=norm)
    plt.colorbar(fraction=0.046, pad=0.04); plt.xlabel("X Pixel"); plt.ylabel("Y Pixel"); plt.title("Calibrated Image Preview"); plt.tight_layout()
    try: plt.savefig(output_png_path, dpi=100); plt.close(); return output_png_path
    except Exception as e: print(f"Error saving PNG preview: {e}"); plt.close(); return None

# --- Handler for Tab 2: LIGHT Frame Correction ---
def handle_calibrate_lights(light_uploads_list, mbias_path, mdark_paths_dict, mflat_paths_dict, stretch_option_dropdown): # Removed request: gr.Request
    if not light_uploads_list: return "No LIGHT frames uploaded for calibration.", None, None, gr.File(visible=False)
    status_messages = [f"Processing {len(light_uploads_list)} LIGHT frame(s)..."]
    calibrated_dir = os.path.join("calibrated_lights_output", "tab2_corrected")
    preview_dir = os.path.join(calibrated_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    if mdark_paths_dict is None: mdark_paths_dict = {}
    if mflat_paths_dict is None: mflat_paths_dict = {}
    master_bias_data_for_flat_processing = load_fits_data(mbias_path) if mbias_path and os.path.exists(mbias_path) else None
    if mbias_path and master_bias_data_for_flat_processing is None: status_messages.append(f"Warning: Failed to load master BIAS from {mbias_path} for flat processing.")

    first_calibrated_fits_path = None; first_preview_png_path = None; all_calibrated_paths = []

    for i, light_file_obj in enumerate(light_uploads_list):
        raw_light_path = light_file_obj.name
        base_name = os.path.splitext(os.path.basename(raw_light_path))[0]
        corrected_fits_path = os.path.join(calibrated_dir, f"{base_name}_calibrated.fits")
        status_messages.append(f"\\nProcessing: {os.path.basename(raw_light_path)}")
        raw_header = get_fits_header(raw_light_path)
        if not raw_header: status_messages.append(f"ERROR: Could not read header for {os.path.basename(raw_light_path)}. Skipping."); continue

        light_exptime_val = raw_header.get('EXPTIME', raw_header.get('EXPOSURE'))
        selected_dark_for_light_path = None
        if light_exptime_val is not None:
            try:
                light_exptime_float = float(light_exptime_val)
                dark_exp_key_s = str(light_exptime_float).replace('.', 'p') + "s"
                dark_exp_key_no_s = str(light_exptime_float).replace('.', 'p')
                if dark_exp_key_s in mdark_paths_dict: selected_dark_for_light_path = mdark_paths_dict[dark_exp_key_s]
                elif dark_exp_key_no_s in mdark_paths_dict: selected_dark_for_light_path = mdark_paths_dict[dark_exp_key_no_s]
                if selected_dark_for_light_path: status_messages.append(f"Found Master DARK for LIGHT: {os.path.basename(selected_dark_for_light_path)} (exp {light_exptime_float}s).")
                else: status_messages.append(f"Warning: No matching Master DARK for LIGHT exposure {light_exptime_float}s. Dark keys: {list(mdark_paths_dict.keys())}")
            except ValueError: status_messages.append(f"Warning: Could not parse EXPTIME '{light_exptime_val}' for LIGHT. Dark subtraction may be affected.")
        else: status_messages.append(f"Warning: EXPTIME not found in LIGHT. Dark subtraction may be compromised.")

        light_filter_name = str(raw_header.get('FILTER', raw_header.get('FILTER1', raw_header.get('FILTNAME', '')))).strip().replace(" ", "_")
        path_to_use_for_final_flat = None; temp_processed_flat_path = None

        if not light_filter_name: status_messages.append(f"Warning: FILTER name not found in {os.path.basename(raw_light_path)}. Flat fielding skipped.")
        else:
            prelim_flat_path = mflat_paths_dict.get(light_filter_name)
            if not prelim_flat_path or not os.path.exists(prelim_flat_path):
                status_messages.append(f"Warning: Prelim. Master FLAT for filter '{light_filter_name}' not found or path invalid. Flat fielding skipped.")
            else:
                status_messages.append(f"Processing Prelim. FLAT {os.path.basename(prelim_flat_path)} for filter '{light_filter_name}'...")
                prelim_flat_data = load_fits_data(prelim_flat_path)
                prelim_flat_header = get_fits_header(prelim_flat_path)
                if prelim_flat_data is None: status_messages.append(f"ERROR: Failed to load Prelim. FLAT data from {prelim_flat_path}. Flat fielding skipped.")
                else:
                    processed_flat_data = prelim_flat_data.astype(np.float32, copy=True)
                    if master_bias_data_for_flat_processing is not None:
                        if master_bias_data_for_flat_processing.shape == processed_flat_data.shape:
                            processed_flat_data -= master_bias_data_for_flat_processing.astype(np.float32)
                            status_messages.append(f"Subtracted Master BIAS from Prelim. FLAT.")
                        else: status_messages.append(f"Warning: Shape mismatch Master BIAS vs Prelim. FLAT. BIAS not subtracted from flat.")
                    else: status_messages.append(f"Info: No Master BIAS for flat processing. Skipping BIAS subtraction from Prelim. FLAT.")

                    if prelim_flat_header:
                        flat_exptime_val = prelim_flat_header.get('EXPTIME', prelim_flat_header.get('EXPOSURE'))
                        if flat_exptime_val is not None:
                            try:
                                flat_exptime_float = float(flat_exptime_val)
                                flat_dark_key_s = str(flat_exptime_float).replace('.', 'p') + "s"
                                flat_dark_key_no_s = str(flat_exptime_float).replace('.', 'p')
                                dark_for_flat_path = mdark_paths_dict.get(flat_dark_key_s) or mdark_paths_dict.get(flat_dark_key_no_s)
                                if dark_for_flat_path and os.path.exists(dark_for_flat_path):
                                    dark_for_flat_data = load_fits_data(dark_for_flat_path)
                                    if dark_for_flat_data is not None and dark_for_flat_data.shape == processed_flat_data.shape:
                                        dark_for_flat_header = get_fits_header(dark_for_flat_path)
                                        dark_for_flat_exptime_val = dark_for_flat_header.get('EXPTIME', dark_for_flat_header.get('EXPOSURE')) if dark_for_flat_header else None
                                        if dark_for_flat_exptime_val is not None:
                                            try:
                                                dark_for_flat_exptime_float = float(dark_for_flat_exptime_val)
                                                if dark_for_flat_exptime_float > 0:
                                                    scale = flat_exptime_float / dark_for_flat_exptime_float
                                                    processed_flat_data -= (dark_for_flat_data.astype(np.float32) * scale)
                                                    status_messages.append(f"Subtracted scaled Master DARK ({os.path.basename(dark_for_flat_path)}, factor {scale:.3f}) from Prelim. FLAT.")
                                                else:
                                                    processed_flat_data -= dark_for_flat_data.astype(np.float32)
                                                    status_messages.append(f"Warn: Master DARK for Prelim. FLAT ({os.path.basename(dark_for_flat_path)}) has zero/negative EXPTIME. Applied unscaled.")
                                            except ValueError: status_messages.append(f"Warn: Could not parse EXPTIME for Master DARK for Prelim. FLAT. Applied unscaled."); processed_flat_data -= dark_for_flat_data.astype(np.float32)
                                        else: status_messages.append(f"Warn: Master DARK for Prelim. FLAT missing EXPTIME. Applied unscaled."); processed_flat_data -= dark_for_flat_data.astype(np.float32)
                                    else: status_messages.append(f"Warn: Failed to load or shape mismatch for Master DARK for Prelim. FLAT. Dark not subtracted from flat.")
                                else: status_messages.append(f"Warn: No matching Master DARK for Prelim. FLAT exposure {flat_exptime_float}s. Dark not subtracted from flat.")
                            except ValueError: status_messages.append(f"Warn: Could not parse EXPTIME '{flat_exptime_val}' for Prelim. FLAT. Dark not subtracted from flat.")
                        else: status_messages.append(f"Warn: Prelim. FLAT missing EXPTIME. Dark not subtracted from flat.")
                    else: status_messages.append(f"Warn: Could not read header for Prelim. FLAT. Dark not subtracted from flat.")

                    median_flat = np.median(processed_flat_data)
                    if median_flat == 0: status_messages.append(f"ERROR: Median of processed flat for '{light_filter_name}' is zero. Flat fielding skipped.")
                    else:
                        processed_flat_data /= median_flat; status_messages.append(f"Processed flat for '{light_filter_name}' normalized.")
                        temp_flat_filename = f"temp_proc_flat_{light_filter_name}_{base_name}_{i}.fits"
                        temp_processed_flat_path = os.path.join(preview_dir, temp_flat_filename)
                        temp_flat_header = fits.Header({'FILTER': light_filter_name, 'EXPTIME': flat_exptime_val if prelim_flat_header and flat_exptime_val else 'UNKNOWN'})
                        temp_flat_header.add_history("Temporary processed flat for Tab 2 calibration")
                        if save_fits_data(temp_processed_flat_path, processed_flat_data, header=temp_flat_header):
                            path_to_use_for_final_flat = temp_processed_flat_path
                            status_messages.append(f"Saved temp processed flat: {os.path.basename(temp_processed_flat_path)}")
                        else: status_messages.append(f"ERROR: Failed to save temp processed flat. Using prelim flat if available."); path_to_use_for_final_flat = prelim_flat_path

        success = correct_light_frame(raw_light_path, corrected_fits_path, mbias_path, selected_dark_for_light_path, path_to_use_for_final_flat)
        if temp_processed_flat_path and os.path.exists(temp_processed_flat_path): _try_remove(temp_processed_flat_path)

        if success:
            status_messages.append(f"Successfully calibrated: {os.path.basename(corrected_fits_path)}")
            all_calibrated_paths.append(corrected_fits_path)
            if first_calibrated_fits_path is None:
                first_calibrated_fits_path = corrected_fits_path
                calibrated_data = load_fits_data(first_calibrated_fits_path)
                if calibrated_data is not None:
                    preview_png_name = f"{base_name}_calibrated_preview.png"
                    preview_png_full_path = os.path.join(preview_dir, preview_png_name)
                    stretch_mode_ui = stretch_option_dropdown.lower(); actual_stretch_mode = 'zscale'; percentile_val = 99.5
                    if "minmax" in stretch_mode_ui: actual_stretch_mode = 'minmax'
                    elif "percentile" in stretch_mode_ui:
                        actual_stretch_mode = 'percentile'
                        try: percentile_val = float(stretch_mode_ui.split()[-1].replace('%',''))
                        except ValueError: status_messages.append(f"Warning: Could not parse percentile from '{stretch_option_dropdown}'. Defaulting to {percentile_val}%.")
                    elif "zscale" in stretch_mode_ui: actual_stretch_mode = 'zscale'
                    status_messages.append(f"Generating preview with stretch: {actual_stretch_mode}" + (f", percentile: {percentile_val}" if actual_stretch_mode == 'percentile' else ""))
                    first_preview_png_path = create_png_preview(calibrated_data, preview_png_full_path, actual_stretch_mode, percentile_val)
                    if first_preview_png_path: status_messages.append(f"Preview generated: {os.path.basename(first_preview_png_path)}")
                    else: status_messages.append(f"Error: Failed to generate preview for {os.path.basename(first_calibrated_fits_path)}.")
                else: status_messages.append(f"Error: Could not load data from {os.path.basename(first_calibrated_fits_path)} for preview.")
        else:
            status_messages.append(f"ERROR: Failed to calibrate {os.path.basename(raw_light_path)}.")
            _try_remove(corrected_fits_path)

    final_status = "\\n".join(status_messages)
    if not all_calibrated_paths: return final_status, None, gr.File(visible=False), gr.Textbox(value="No calibrated FITS files generated.", visible=True)
    download_label = f"Download {os.path.basename(first_calibrated_fits_path)}" if first_calibrated_fits_path else "No FITS available"
    calibrated_files_summary = "Successfully Calibrated Files:\\n" + "\\n".join([os.path.basename(p) for p in all_calibrated_paths]) + f"\\n\\nTotal: {len(all_calibrated_paths)} file(s)."
    return final_status, gr.Image(value=first_preview_png_path, visible=first_preview_png_path is not None), gr.File(value=first_calibrated_fits_path, label=download_label, visible=first_calibrated_fits_path is not None), gr.Textbox(value=calibrated_files_summary, label="Calibrated Files List", visible=True)

# --- Handler for Tab 3: Extinction Coefficient --- (NEW: handle_tab3_extinction_from_fits)
def handle_tab3_extinction_from_fits(
    light_files_for_extinction,
    mbias_path,
    mdark_paths_dict,
    mflat_paths_dict,
    request: gr.Request
):
    if not light_files_for_extinction:
        return "No FITS files uploaded for extinction analysis.", "", "", "", None, None

    status_messages = [f"Processing {len(light_files_for_extinction)} FITS file(s) for extinction analysis..."]
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="tab3_extinction_")
    temp_dir_path = temp_dir_obj.name

    master_bias_data_for_flat_proc = load_fits_data(mbias_path) if mbias_path and os.path.exists(mbias_path) else None
    if mbias_path and master_bias_data_for_flat_proc is None:
        status_messages.append(f"Warning: Master BIAS for flat processing ({mbias_path}) could not be loaded. Flat calibration may be affected.")

    airmass_mag_pairs = []
    results_data = [] # For the DataFrame

    # Fixed settings for DAOStarFinder and Photometry for Tab 3
    FWHM_FIXED = 5.0
    APERTURE_RADIUS_FIXED = 1.5 * FWHM_FIXED
    SKY_INNER_FIXED = 2.5 * FWHM_FIXED
    SKY_OUTER_FIXED = 3.5 * FWHM_FIXED
    DETECTION_THRESHOLD_SIGMA = 5.0

    for i, fits_file_obj in enumerate(light_files_for_extinction):
        raw_light_path = fits_file_obj.name
        base_name = os.path.splitext(os.path.basename(raw_light_path))[0]
        current_status_prefix = f"File {i+1} ({base_name}): "

        try:
            raw_header = get_fits_header(raw_light_path)
            if not raw_header:
                status_messages.append(current_status_prefix + "ERROR: Could not read header. Skipping.")
                results_data.append({"Filename": base_name, "Airmass": np.nan, "Instrumental_Magnitude": np.nan, "Raw_Flux": np.nan, "Net_Flux": np.nan, "X_cen": np.nan, "Y_cen": np.nan, "Skipped_Reason": "Header read error"})
                continue

            calibrated_fits_path = os.path.join(temp_dir_path, f"{base_name}_calibrated.fits")

            light_exptime_val = raw_header.get('EXPTIME', raw_header.get('EXPOSURE'))
            selected_dark_for_light_path = None
            if light_exptime_val is not None:
                try:
                    light_exptime_float = float(light_exptime_val)
                    dark_exp_key_s = str(light_exptime_float).replace('.', 'p') + "s"
                    dark_exp_key_no_s = str(light_exptime_float).replace('.', 'p')
                    if mdark_paths_dict and dark_exp_key_s in mdark_paths_dict: selected_dark_for_light_path = mdark_paths_dict[dark_exp_key_s]
                    elif mdark_paths_dict and dark_exp_key_no_s in mdark_paths_dict: selected_dark_for_light_path = mdark_paths_dict[dark_exp_key_no_s]
                except ValueError: pass

            light_filter_name = str(raw_header.get('FILTER', raw_header.get('FILTER1', raw_header.get('FILTNAME', '')))).strip().replace(" ", "_")
            path_to_use_for_final_flat = None
            temp_processed_flat_path_tab3 = None

            if light_filter_name and mflat_paths_dict and mflat_paths_dict.get(light_filter_name) and os.path.exists(mflat_paths_dict.get(light_filter_name)):
                prelim_flat_path = mflat_paths_dict.get(light_filter_name)
                prelim_flat_data = load_fits_data(prelim_flat_path)
                prelim_flat_header = get_fits_header(prelim_flat_path)
                if prelim_flat_data is not None:
                    processed_flat_data = prelim_flat_data.astype(np.float32, copy=True)
                    if master_bias_data_for_flat_proc is not None and master_bias_data_for_flat_proc.shape == processed_flat_data.shape:
                        processed_flat_data -= master_bias_data_for_flat_proc.astype(np.float32)

                    if prelim_flat_header:
                        flat_exptime_val = prelim_flat_header.get('EXPTIME', prelim_flat_header.get('EXPOSURE'))
                        if flat_exptime_val is not None:
                            try:
                                flat_exptime_float = float(flat_exptime_val)
                                flat_dark_key_s = str(flat_exptime_float).replace('.', 'p') + "s"
                                flat_dark_key_no_s = str(flat_exptime_float).replace('.', 'p')
                                dark_for_flat_path = None
                                if mdark_paths_dict and flat_dark_key_s in mdark_paths_dict: dark_for_flat_path = mdark_paths_dict[flat_dark_key_s]
                                elif mdark_paths_dict and flat_dark_key_no_s in mdark_paths_dict: dark_for_flat_path = mdark_paths_dict[flat_dark_key_no_s]

                                if dark_for_flat_path and os.path.exists(dark_for_flat_path):
                                    dark_for_flat_data = load_fits_data(dark_for_flat_path)
                                    if dark_for_flat_data is not None and dark_for_flat_data.shape == processed_flat_data.shape:
                                        dark_for_flat_header = get_fits_header(dark_for_flat_path)
                                        dark_for_flat_exptime_val = dark_for_flat_header.get('EXPTIME', dark_for_flat_header.get('EXPOSURE')) if dark_for_flat_header else None
                                        if dark_for_flat_exptime_val is not None:
                                            try:
                                                dark_for_flat_exptime_float = float(dark_for_flat_exptime_val)
                                                if dark_for_flat_exptime_float > 0:
                                                    scale = flat_exptime_float / dark_for_flat_exptime_float
                                                    processed_flat_data -= (dark_for_flat_data.astype(np.float32) * scale)
                                            except ValueError: processed_flat_data -= dark_for_flat_data.astype(np.float32)
                                        else: processed_flat_data -= dark_for_flat_data.astype(np.float32)
                            except ValueError: pass

                    median_flat = np.median(processed_flat_data)
                    if median_flat != 0: processed_flat_data /= median_flat
                    else: processed_flat_data = None

                    if processed_flat_data is not None:
                        temp_flat_filename = f"temp_tab3_proc_flat_{light_filter_name}_{base_name}.fits"
                        temp_processed_flat_path_tab3 = os.path.join(temp_dir_path, temp_flat_filename)
                        temp_flat_header = fits.Header({'FILTER': light_filter_name})
                        if save_fits_data(temp_processed_flat_path_tab3, processed_flat_data, header=temp_flat_header):
                            path_to_use_for_final_flat = temp_processed_flat_path_tab3

            # Check if flat fielding was expected but not performed
            flat_skipped_due_to_missing_master = False
            if light_filter_name and (not mflat_paths_dict or not mflat_paths_dict.get(light_filter_name) or not os.path.exists(mflat_paths_dict.get(light_filter_name))):
                # A filter was present, implying a flat was desired, but the master flat for it was missing.
                # And path_to_use_for_final_flat would be None in this case if not processed above.
                if path_to_use_for_final_flat is None:
                    flat_skipped_due_to_missing_master = True
                    status_messages.append(current_status_prefix + f"Warning: Required flat for filter '{light_filter_name}' was not found. Marking as calibration failed.")

            if flat_skipped_due_to_missing_master:
                results_data.append({"Filename": base_name, "Airmass": np.nan, "Instrumental_Magnitude": np.nan, "Raw_Flux": np.nan, "Net_Flux": np.nan, "X_cen": np.nan, "Y_cen": np.nan, "Skipped_Reason": "Calibration failed"})
                if temp_processed_flat_path_tab3: _try_remove(temp_processed_flat_path_tab3)
                _try_remove(calibrated_fits_path) # Also remove any intermediate calibrated file
                continue

            if not correct_light_frame(raw_light_path, calibrated_fits_path, mbias_path, selected_dark_for_light_path, path_to_use_for_final_flat):
                status_messages.append(current_status_prefix + f"ERROR: correct_light_frame returned false. Skipping.")
                results_data.append({"Filename": base_name, "Airmass": np.nan, "Instrumental_Magnitude": np.nan, "Raw_Flux": np.nan, "Net_Flux": np.nan, "X_cen": np.nan, "Y_cen": np.nan, "Skipped_Reason": "Calibration failed"})
                if temp_processed_flat_path_tab3: _try_remove(temp_processed_flat_path_tab3)
                continue
            status_messages.append(current_status_prefix + f"Calibrated: {os.path.basename(calibrated_fits_path)}")
            if temp_processed_flat_path_tab3: _try_remove(temp_processed_flat_path_tab3)

            airmass_val = raw_header.get('AIRMASS')
            if airmass_val is None:
                status_messages.append(current_status_prefix + "ERROR: AIRMASS keyword not found in header. Skipping.")
                results_data.append({"Filename": base_name, "Airmass": np.nan, "Instrumental_Magnitude": np.nan, "Raw_Flux": np.nan, "Net_Flux": np.nan, "X_cen": np.nan, "Y_cen": np.nan, "Skipped_Reason": "AIRMASS not found"})
                continue
            try:
                airmass = float(airmass_val)
            except ValueError:
                status_messages.append(current_status_prefix + f"ERROR: Invalid AIRMASS value '{airmass_val}'. Skipping.")
                results_data.append({"Filename": base_name, "Airmass": airmass_val, "Instrumental_Magnitude": np.nan, "Raw_Flux": np.nan, "Net_Flux": np.nan, "X_cen": np.nan, "Y_cen": np.nan, "Skipped_Reason": "Invalid AIRMASS value"})
                continue

            calibrated_data = load_fits_data(calibrated_fits_path)
            if calibrated_data is None:
                status_messages.append(current_status_prefix + "ERROR: Could not load calibrated data for photometry. Skipping.")
                results_data.append({"Filename": base_name, "Airmass": airmass, "Instrumental_Magnitude": np.nan, "Raw_Flux": np.nan, "Net_Flux": np.nan, "X_cen": np.nan, "Y_cen": np.nan, "Skipped_Reason": "Failed to load calibrated data"})
                continue

            mean, median, std = sigma_clipped_stats(calibrated_data, sigma=3.0)
            daofind = DAOStarFinder(fwhm=FWHM_FIXED, threshold=DETECTION_THRESHOLD_SIGMA * std)
            sources_table = daofind(calibrated_data - median)

            if not sources_table:
                status_messages.append(current_status_prefix + "No sources found. Skipping.")
                results_data.append({"Filename": base_name, "Airmass": airmass, "Instrumental_Magnitude": np.nan, "Raw_Flux": np.nan, "Net_Flux": np.nan, "X_cen": np.nan, "Y_cen": np.nan, "Skipped_Reason": "No sources detected"})
                continue

            sources_table.sort('flux', reverse=True)
            brightest_source = sources_table[0]
            x_cen, y_cen = brightest_source['xcentroid'], brightest_source['ycentroid']

            aperture = CircularAperture((x_cen, y_cen), r=APERTURE_RADIUS_FIXED)
            annulus_aperture = CircularAnnulus((x_cen, y_cen), r_in=SKY_INNER_FIXED, r_out=SKY_OUTER_FIXED)

            ap_stats = ApertureStats(calibrated_data, aperture)
            ann_stats = ApertureStats(calibrated_data, annulus_aperture, sigma_clip=SigmaClip(sigma=3.0))

            if ap_stats.sum is None:
                status_messages.append(current_status_prefix + "Photometry failed (aperture sum is None). Skipping.")
                results_data.append({"Filename": base_name, "Airmass": airmass, "Instrumental_Magnitude": np.nan, "Raw_Flux": np.nan, "Net_Flux": np.nan, "X_cen": x_cen, "Y_cen": y_cen, "Skipped_Reason": "Aperture sum None"})
                continue

            raw_flux = ap_stats.sum

            # CRITICAL FIX: Use aperture.area instead of ap_stats.area for sky sum calculation.
            if ann_stats.median is None or ann_stats.sum is None or aperture.area == 0:
                status_messages.append(current_status_prefix + "Warning: Sky annulus stats invalid or aperture area is zero. Using zero sky subtraction.")
                sky_median_per_pixel = 0.0
            else:
                sky_median_per_pixel = ann_stats.median

            sky_sum_in_aperture = sky_median_per_pixel * aperture.area # Corrected line
            net_flux = raw_flux - sky_sum_in_aperture

            if net_flux <= 0:
                status_messages.append(current_status_prefix + f"Net flux is not positive ({net_flux:.2f}). Skipping.")
                results_data.append({"Filename": base_name, "Airmass": airmass, "Instrumental_Magnitude": np.nan, "Raw_Flux": raw_flux, "Net_Flux": net_flux, "X_cen": x_cen, "Y_cen": y_cen, "Skipped_Reason": "Non-positive net flux"})
                continue

            instrumental_mag = -2.5 * np.log10(net_flux)
            airmass_mag_pairs.append((airmass, instrumental_mag))
            results_data.append({"Filename": base_name, "Airmass": airmass, "Instrumental_Magnitude": instrumental_mag, "Raw_Flux": raw_flux, "Net_Flux": net_flux, "X_cen": x_cen, "Y_cen": y_cen, "Skipped_Reason": ""})
            status_messages.append(current_status_prefix + f"Airmass: {airmass:.3f}, InstrMag: {instrumental_mag:.3f} (Net Flux: {net_flux:.2f})")

        except Exception as e:
            status_messages.append(current_status_prefix + f"ERROR during processing: {e}. Skipping.")
            results_data.append({"Filename": base_name, "Airmass": np.nan, "Instrumental_Magnitude": np.nan, "Raw_Flux": np.nan, "Net_Flux": np.nan, "X_cen": np.nan, "Y_cen": np.nan, "Skipped_Reason": str(e)})
            continue
        finally:
            _try_remove(calibrated_fits_path)

    df_results = pd.DataFrame(results_data)

    if len(airmass_mag_pairs) < 2:
        status_messages.append("ERROR: Less than 2 data points collected. Cannot perform linear regression.")
        if temp_dir_obj: temp_dir_obj.cleanup()
        return "\\n".join(status_messages), "", "", "", None, df_results

    airmasses_arr, magnitudes_arr = zip(*airmass_mag_pairs)
    regression = linregress(airmasses_arr, magnitudes_arr)

    k_val = -regression.slope if regression.slope is not None else np.nan
    m0_val = regression.intercept if regression.intercept is not None else np.nan
    k_err_val = regression.stderr if regression.stderr is not None else np.nan

    status_messages.append(f"Linear Regression: k = {k_val:.4f}, m0 = {m0_val:.4f}, k_err = {k_err_val:.4f}")

    plot_path = None
    plot_filename = f"tab3_extinction_plot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plot_save_dir = os.path.join("calibrated_lights_output", "previews", "tab3_plots")
    os.makedirs(plot_save_dir, exist_ok=True)
    plot_path = os.path.join(plot_save_dir, plot_filename)

    plt.figure(figsize=(8, 6))
    plt.scatter(airmasses_arr, magnitudes_arr, color='blue', label='Data Points')
    if regression.slope is not None and regression.intercept is not None:
        fit_line_y = regression.intercept + regression.slope * np.array(airmasses_arr)
        plt.plot(airmasses_arr, fit_line_y, color='red', label=f'Fit: m = {m0_val:.3f} - {k_val:.3f}X')
    plt.xlabel("Airmass (X)")
    plt.ylabel("Instrumental Magnitude (m)")
    plt.title("Atmospheric Extinction Plot (Tab 3)")
    plt.legend(); plt.grid(True); plt.gca().invert_yaxis(); plt.tight_layout()
    try:
        plt.savefig(plot_path); plt.close()
        status_messages.append(f"Plot generated: {os.path.basename(plot_path)}")
    except Exception as e_plot:
        plt.close()
        status_messages.append(f"Plot generation failed: {e_plot}")
        plot_path = None

    if temp_dir_obj: temp_dir_obj.cleanup()

    return ("\\n".join(status_messages),
            f"{k_val:.4f}" if not np.isnan(k_val) else "",
            f"{k_err_val:.4f}" if not np.isnan(k_err_val) else "",
            f"{m0_val:.4f}" if not np.isnan(m0_val) else "",
            plot_path,
            df_results)


# --- Handler for Tab 4: Photometry ---
def handle_tab4_photometry(
    tab4_b_file_obj, tab4_v_file_obj,
    tab4_std_star_file_obj, tab4_std_b_mag_str, tab4_std_v_mag_str,
    tab4_roi_str,
    tab4_fwhm_input_val, tab4_aperture_radius_input_val,
    tab4_sky_inner_input_val, tab4_sky_outer_input_val,
    tab4_k_b_value_str, tab4_k_v_value_str, # Updated k-value parameters
    master_bias_path_state, master_dark_paths_state, master_flat_paths_state
):
    status_messages = [
        "Starting Tab 4 Photometry Analysis...",
        f"Settings: FWHM={tab4_fwhm_input_val}px, Aperture Radius={tab4_aperture_radius_input_val}px",
        f"Sky Annulus: Inner={tab4_sky_inner_input_val}px, Outer={tab4_sky_outer_input_val}px"
    ]
    final_results_df_for_ui = None; preview_image_path_for_ui = None; csv_file_path_for_ui = None
    display_columns = ['ID', 'X_B', 'Y_B', 'RA_deg', 'Dec_deg', 'InstrMag_B', 'InstrMag_V', 'StdMag_B', 'StdMag_V', 'B-V']
    if master_dark_paths_state is None: master_dark_paths_state = {}
    if master_flat_paths_state is None: master_flat_paths_state = {}
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="tab4_phot_") # Use tempfile for auto-cleanup
    temp_dir_path = temp_dir_obj.name
    calibrated_dir = os.path.join(temp_dir_path, "calibrated") # Store intermediate calibrated files here
    previews_dir = os.path.join("calibrated_lights_output", "previews", "tab4_previews") # User-facing previews
    csv_output_dir = os.path.join("calibrated_lights_output", "csv_results", "tab4_results") # User-facing CSVs

    os.makedirs(calibrated_dir, exist_ok=True)
    os.makedirs(previews_dir, exist_ok=True)
    os.makedirs(csv_output_dir, exist_ok=True)

    try:
        if not tab4_b_file_obj or not tab4_v_file_obj: raise ValueError("Both B and V frames must be uploaded.")
        raw_b_path = tab4_b_file_obj.name; raw_v_path = tab4_v_file_obj.name
        status_messages.append(f"B: {os.path.basename(raw_b_path)}, V: {os.path.basename(raw_v_path)}")
        b_header = get_fits_header(raw_b_path)
        v_header = get_fits_header(raw_v_path)
        if b_header is None: raise ValueError(f"Could not load B-frame header: {raw_b_path}")
        if v_header is None: raise ValueError(f"Could not load V-frame header: {raw_v_path}")
        status_messages.append("B,V LIGHT headers loaded.")

        master_bias_data_tab4 = load_fits_data(master_bias_path_state) if master_bias_path_state and os.path.exists(master_bias_path_state) else None
        if master_bias_path_state and master_bias_data_tab4 is None : status_messages.append(f"Warning: Failed to load Master BIAS from {master_bias_path_state}. Calibration may be affected.")
        elif master_bias_data_tab4 is not None: status_messages.append(f"Master BIAS loaded for Tab 4 processing.")

        def _correct_science_frame(label, raw_path, header):
            status_messages.append(f"Correcting {label}-frame...");
            science_frame_exptime_val = header.get('EXPTIME', header.get('EXPOSURE'))
            science_frame_filter_name = header.get('FILTER', header.get('FILTER1', header.get('FILTNAME')))
            if science_frame_exptime_val is None: raise ValueError(f"{label}-frame missing EXPTIME.");
            if science_frame_filter_name is None: raise ValueError(f"{label}-frame missing FILTER.")

            science_frame_exptime_float = float(science_frame_exptime_val)
            science_frame_filter_key = str(science_frame_filter_name).strip().replace(" ", "_")

            actual_dark_for_light_path = None
            if master_dark_paths_state:
                dark_exp_key_s = str(science_frame_exptime_float).replace('.', 'p') + "s"
                dark_exp_key_no_s = str(science_frame_exptime_float).replace('.', 'p')
                if dark_exp_key_s in master_dark_paths_state: actual_dark_for_light_path = master_dark_paths_state[dark_exp_key_s]
                elif dark_exp_key_no_s in master_dark_paths_state: actual_dark_for_light_path = master_dark_paths_state[dark_exp_key_no_s]

            final_flat_path_for_science = None; temp_flat_generated_path = None
            if master_flat_paths_state and science_frame_filter_key in master_flat_paths_state:
                prelim_flat_p = master_flat_paths_state[science_frame_filter_key]
                if prelim_flat_p and os.path.exists(prelim_flat_p):
                    prelim_flat_d = load_fits_data(prelim_flat_p)
                    prelim_flat_header = get_fits_header(prelim_flat_p)
                    if prelim_flat_d is not None:
                        flat_cal_data = prelim_flat_d.astype(np.float32, copy=True)
                        if master_bias_data_tab4 is not None:
                            if prelim_flat_d.shape!=master_bias_data_tab4.shape:
                                status_messages.append(f"Warn: Prelim. {label}-FLAT ({os.path.basename(prelim_flat_p)}) shape mismatch vs Master BIAS. BIAS not sub from flat.")
                            else: flat_cal_data -= master_bias_data_tab4.astype(np.float32); status_messages.append(f"Subtracted Master BIAS from Prelim. {label}-FLAT {os.path.basename(prelim_flat_p)}.")

                        if prelim_flat_header:
                            flat_exptime_val = prelim_flat_header.get('EXPTIME', prelim_flat_header.get('EXPOSURE'))
                            if flat_exptime_val is not None:
                                try:
                                    flat_exptime_float = float(flat_exptime_val)
                                    dark_for_flat_path = None
                                    if master_dark_paths_state:
                                        flat_dark_exp_key_s = str(flat_exptime_float).replace('.', 'p') + "s"
                                        flat_dark_exp_key_no_s = str(flat_exptime_float).replace('.', 'p')
                                        if flat_dark_exp_key_s in master_dark_paths_state: dark_for_flat_path = master_dark_paths_state[flat_dark_exp_key_s]
                                        elif flat_dark_exp_key_no_s in master_dark_paths_state: dark_for_flat_path = master_dark_paths_state[flat_dark_exp_key_no_s]

                                    if dark_for_flat_path and os.path.exists(dark_for_flat_path):
                                        dark_for_flat_data = load_fits_data(dark_for_flat_path)
                                        if dark_for_flat_data is not None and dark_for_flat_data.shape == flat_cal_data.shape:
                                            dark_for_flat_hdr = get_fits_header(dark_for_flat_path)
                                            dark_for_flat_exp_val = dark_for_flat_hdr.get('EXPTIME',dark_for_flat_hdr.get('EXPOSURE')) if dark_for_flat_hdr else None
                                            if dark_for_flat_exp_val is not None:
                                                try:
                                                    dark_for_flat_exp_float = float(dark_for_flat_exp_val)
                                                    if dark_for_flat_exp_float > 0:
                                                        s_factor = flat_exptime_float / dark_for_flat_exp_float
                                                        flat_cal_data -= (dark_for_flat_data.astype(np.float32) * s_factor)
                                                        status_messages.append(f"Subtracted scaled Master DARK ({os.path.basename(dark_for_flat_path)}, factor {s_factor:.3f}) from Prelim. {label}-FLAT.")
                                                    else: status_messages.append(f"Warn: Master DARK for Prelim. {label}-FLAT ({os.path.basename(dark_for_flat_path)}) has zero/neg EXPTIME. Applied unscaled."); flat_cal_data -= dark_for_flat_data.astype(np.float32)
                                                except ValueError: status_messages.append(f"Warn: Could not parse EXPTIME for Master DARK for Prelim. {label}-FLAT. Applied unscaled."); flat_cal_data -= dark_for_flat_data.astype(np.float32)
                                            else: status_messages.append(f"Warn: Master DARK for Prelim. {label}-FLAT missing EXPTIME. Applied unscaled."); flat_cal_data -= dark_for_flat_data.astype(np.float32)
                                        else: status_messages.append(f"Warn: Failed/shape mismatch Master DARK for Prelim. {label}-FLAT. Dark not subtracted from flat.")
                                    else: status_messages.append(f"Warn: No matching Master DARK for Prelim. {label}-FLAT exp {flat_exptime_float}s. Dark not subtracted from flat.")
                                except ValueError: status_messages.append(f"Warn: Could not parse EXPTIME for Prelim. {label}-FLAT. Dark not subtracted from flat.")
                            else: status_messages.append(f"Warn: Prelim. {label}-FLAT missing EXPTIME. Dark not subtracted from flat.")
                        else: status_messages.append(f"Warn: Could not read header for Prelim. {label}-FLAT. Dark not subtracted from flat.")

                        median_flat_cal = np.median(flat_cal_data)
                        if median_flat_cal == 0: raise ValueError(f"Median of processed Prelim. {label}-FLAT is zero.")
                        final_flat_d = flat_cal_data / median_flat_cal
                        temp_flat_generated_path = os.path.join(temp_dir_path, f"final_{label}_flat_{science_frame_filter_key}.fits")
                        tmp_flat_h = get_fits_header(prelim_flat_p) if get_fits_header(prelim_flat_p) else fits.Header()
                        tmp_flat_h['HISTORY'] = f'Final {label}-flat for Tab4'; save_fits_data(temp_flat_generated_path, final_flat_d, header=tmp_flat_h)
                        final_flat_path_for_science = temp_flat_generated_path
                        status_messages.append(f"Temp final {label}-FLAT: {os.path.basename(final_flat_path_for_science)}")
                    elif master_bias_data_tab4 is None : status_messages.append(f"Warn: Master BIAS not available. {label}-frame not flat-fielded with full calibration.")
                    else: status_messages.append(f"Warn: Failed load Prelim {label}-FLAT. {label}-frame not flat-fielded.")
            else: status_messages.append(f"Warn: Prelim {label}-FLAT for '{science_frame_filter_key}' not found. {label}-frame not flat-fielded.")

            base_n = os.path.splitext(os.path.basename(raw_path))[0]; corrected_p = os.path.join(calibrated_dir, f"{base_n}_cal_{label}.fits")
            if not correct_light_frame(raw_path, corrected_p, master_bias_path_state, actual_dark_for_light_path, final_flat_path_for_science):
                raise ValueError(f"Failed to correct {label}-frame.")
            status_messages.append(f"{label}-frame corrected: {os.path.basename(corrected_p)}");
            # _try_remove(temp_flat_generated_path) # Keep for now, cleanup with temp_dir_obj
            return corrected_p

        corrected_b_path = _correct_science_frame("B", raw_b_path, b_header)
        corrected_v_path = _correct_science_frame("V", raw_v_path, v_header)
        corrected_b_data = load_fits_data(corrected_b_path)
        corrected_v_data = load_fits_data(corrected_v_path)
        if corrected_b_data is None or corrected_v_data is None: raise ValueError("Failed to load corrected B or V frame data.")

        m0_B, m0_V = 25.0, 25.0; status_messages.append(f"Default m0_B={m0_B:.3f}, m0_V={m0_V:.3f}.")
        if tab4_std_star_file_obj and tab4_std_b_mag_str.strip() and tab4_std_v_mag_str.strip():
            status_messages.append("Std Star Processing...");
            try:
                std_path = tab4_std_star_file_obj.name; std_b_known = float(tab4_std_b_mag_str.strip()); std_v_known = float(tab4_std_v_mag_str.strip())
                std_hdr_raw = get_fits_header(std_path);
                if std_hdr_raw is None: raise ValueError("Cannot read std star header.")
                std_filt = std_hdr_raw.get('FILTER',std_hdr_raw.get('FILTER1',std_hdr_raw.get('FILTNAME')));
                if std_filt is None: raise ValueError("Std Star FITS missing FILTER.")
                std_filt_key = str(std_filt).strip().upper()
                corr_std_p = _correct_science_frame("STD",std_path,std_hdr_raw)
                if not corr_std_p: raise ValueError("Std star correction failed.")
                corr_std_d = load_fits_data(corr_std_p);
                if corr_std_d is None: raise ValueError("Failed load corrected std star data.")
                mean_s,med_s,std_s_val=sigma_clipped_stats(corr_std_d,sigma=3.0)
                dao_s=DAOStarFinder(fwhm=tab4_fwhm_input_val,threshold=5.*std_s_val)
                srcs_s_tbl=dao_s(corr_std_d-med_s);
                if not srcs_s_tbl or len(srcs_s_tbl)==0: raise ValueError("No sources in std star img.")
                srcs_s_tbl.sort('flux',reverse=True); std_x,std_y=srcs_s_tbl[0]['xcentroid'],srcs_s_tbl[0]['ycentroid']
                std_phot_res_list=perform_photometry(corr_std_d,[(std_x,std_y)], tab4_aperture_radius_input_val, tab4_sky_inner_input_val, tab4_sky_outer_input_val)
                if not std_phot_res_list or 'instrumental_mag' not in std_phot_res_list[0] or std_phot_res_list[0]['instrumental_mag'] is None: raise ValueError("Photometry failed for std star.")
                instr_mag_s=std_phot_res_list[0]['instrumental_mag']
                # k_val is no longer a single value, use k_b_val or k_v_val
                std_airmass=float(std_hdr_raw.get('AIRMASS',1.0))
                if std_filt_key.startswith('B'): m0_B=std_b_known-instr_mag_s+(k_b_val*std_airmass); status_messages.append(f"Calib m0_B={m0_B:.3f} using k(B)={k_b_val:.2f}")
                elif std_filt_key.startswith('V'): m0_V=std_v_known-instr_mag_s+(k_v_val*std_airmass); status_messages.append(f"Calib m0_V={m0_V:.3f} using k(V)={k_v_val:.2f}")
                else: status_messages.append(f"Warn: Std star filt '{std_filt_key}' not B/V. Using default m0 for this band if applicable, or specific k if defined for it.")
                # _try_remove(corr_std_p) # Let temp_dir_obj handle cleanup
            except Exception as e_std: status_messages.append(f"Std star error: {e_std}. Using default m0s.")

        srcs_b_tbl=None;
        if corrected_b_data is not None:
            _,med_b,std_b_val=sigma_clipped_stats(corrected_b_data,sigma=3.0)
            daofind_b=DAOStarFinder(fwhm=tab4_fwhm_input_val,threshold=5.*std_b_val)
            srcs_b_tbl=daofind_b(corrected_b_data-med_b)
            status_messages.append(f"B-srcs: {len(srcs_b_tbl) if srcs_b_tbl else 0}")
        srcs_v_tbl=None;
        if corrected_v_data is not None:
            _,med_v,std_v_val=sigma_clipped_stats(corrected_v_data,sigma=3.0)
            daofind_v=DAOStarFinder(fwhm=tab4_fwhm_input_val,threshold=5.*std_v_val)
            srcs_v_tbl=daofind_v(corrected_v_data-med_v)
            status_messages.append(f"V-srcs: {len(srcs_v_tbl) if srcs_v_tbl else 0}")

        phot_b, phot_v = [], []
        roi_x,roi_y,roi_r = None,None,None
        if tab4_roi_str and tab4_roi_str.strip():
            try:
                parts=[float(p.strip()) for p in tab4_roi_str.split(',')];
                if len(parts)==3: roi_x,roi_y,roi_r=parts;
                if srcs_b_tbl is not None: dist_b=np.sqrt((srcs_b_tbl['xcentroid']-roi_x)**2+(srcs_b_tbl['ycentroid']-roi_y)**2); srcs_b_tbl=srcs_b_tbl[dist_b<=roi_r]
                if srcs_v_tbl is not None: dist_v=np.sqrt((srcs_v_tbl['xcentroid']-roi_x)**2+(srcs_v_tbl['ycentroid']-roi_y)**2); srcs_v_tbl=srcs_v_tbl[dist_v<=roi_r]
                status_messages.append(f"ROI: {len(srcs_b_tbl) if srcs_b_tbl is not None else 0} B, {len(srcs_v_tbl) if srcs_v_tbl is not None else 0} V srcs.")
            except Exception as e_roi: status_messages.append(f"Warn: ROI error: {e_roi}.")

        if srcs_b_tbl and len(srcs_b_tbl)>0:
            coords_b=np.array([(s['xcentroid'],s['ycentroid']) for s in srcs_b_tbl])
            phot_b=perform_photometry(corrected_b_data,coords_b,tab4_aperture_radius_input_val, tab4_sky_inner_input_val, tab4_sky_outer_input_val)
        if srcs_v_tbl and len(srcs_v_tbl)>0:
            coords_v=np.array([(s['xcentroid'],s['ycentroid']) for s in srcs_v_tbl])
            phot_v=perform_photometry(corrected_v_data,coords_v,tab4_aperture_radius_input_val, tab4_sky_inner_input_val, tab4_sky_outer_input_val)

        try:
            k_b_val = float(tab4_k_b_value_str.strip()) if tab4_k_b_value_str.strip() else 0.22
        except ValueError:
            status_messages.append(f"Warning: Invalid k(B) value '{tab4_k_b_value_str}'. Using default 0.22.")
            k_b_val = 0.22
        try:
            k_v_val = float(tab4_k_v_value_str.strip()) if tab4_k_v_value_str.strip() else 0.12
        except ValueError:
            status_messages.append(f"Warning: Invalid k(V) value '{tab4_k_v_value_str}'. Using default 0.12.")
            k_v_val = 0.12
        status_messages.append(f"Using k(B)={k_b_val:.2f}, k(V)={k_v_val:.2f}")

        air_b=float(b_header.get('AIRMASS',1.0)); air_v=float(v_header.get('AIRMASS',1.0))
        for r_d in phot_b:
            if r_d.get('instrumental_mag') is not None: r_d['StdMag_B']=r_d['instrumental_mag']+m0_B-(k_b_val*air_b) # Use k_b_val
            if b_header:
                try:
                    w=WCS(b_header)
                    if w.is_celestial:
                        sky_coord = w.pixel_to_world(r_d['x'],r_d['y'])
                        r_d.update({'ra_deg': sky_coord.ra.deg, 'dec_deg': sky_coord.dec.deg})
                except Exception as e_wcs_b:
                    status_messages.append(f"Warning: WCS conversion failed for a B-frame source: {e_wcs_b}")
        for r_d in phot_v:
            if r_d.get('instrumental_mag') is not None: r_d['StdMag_V']=r_d['instrumental_mag']+m0_V-(k_v_val*air_v) # Use k_v_val
            if v_header:
                try:
                    w=WCS(v_header)
                    if w.is_celestial:
                        sky_coord = w.pixel_to_world(r_d['x'],r_d['y'])
                        r_d.update({'ra_deg': sky_coord.ra.deg, 'dec_deg': sky_coord.dec.deg})
                except Exception as e_wcs_v:
                    status_messages.append(f"Warning: WCS conversion failed for a V-frame source: {e_wcs_v}")
        status_messages.append("Std Mags Calc.")

        results,matched_prev = [],[]; match_r_px=3.0
        use_sky = all(p.get('ra_deg') is not None for p in phot_b) and all(p.get('ra_deg') is not None for p in phot_v)
        if use_sky:
            status_messages.append("RA/Dec cross-match."); sky_b=SkyCoord([p['ra_deg'] for p in phot_b]*u.deg,[p['dec_deg'] for p in phot_b]*u.deg); sky_v=SkyCoord([p['ra_deg'] for p in phot_v]*u.deg,[p['dec_deg'] for p in phot_v]*u.deg)
            idx,d2d,_=sky_b.match_to_catalog_sky(sky_v); p_scale=abs(b_header.get('CDELT1',b_header.get('CD1_1',0.5))*3600); tol=match_r_px*p_scale*u.arcsec; v_matched_indices=set()
            for i,s_b in enumerate(phot_b):
                e={'ID':i+1,'X_B':s_b.get('x'),'Y_B':s_b.get('y'),'RA_deg':s_b.get('ra_deg'),'Dec_deg':s_b.get('dec_deg'),'InstrMag_B':s_b.get('instrumental_mag'),'StdMag_B':s_b.get('StdMag_B'),'InstrMag_V':None,'StdMag_V':None,'B-V':None,'X_V':None,'Y_V':None}
                if idx[i] < len(phot_v) and d2d[i]<=tol : # Check idx bounds
                    m_v_idx=idx[i];
                    if m_v_idx not in v_matched_indices: s_v=phot_v[m_v_idx];e.update({'InstrMag_V':s_v.get('instrumental_mag'),'StdMag_V':s_v.get('StdMag_V'),'X_V':s_v.get('x'),'Y_V':s_v.get('y')});v_matched_indices.add(m_v_idx)
                if e['StdMag_B'] is not None and e['StdMag_V'] is not None: e['B-V']=e['StdMag_B']-e['StdMag_V']
                if e['InstrMag_V'] is not None: matched_prev.append({'x':s_b['x'],'y':s_b['y'],'id':e['ID']})
                results.append(e)
        else:
            status_messages.append("Pixel X/Y cross-match."); avail_v=[dict(s) for s in phot_v]
            for i,s_b in enumerate(phot_b):
                s_b['id']=s_b.get('id',i+1); best_m_v=None;min_d=float('inf');pop_idx=-1
                for i_v,s_v_a in enumerate(avail_v):
                    d=np.sqrt((s_b['x']-s_v_a['x'])**2+(s_b['y']-s_v_a['y'])**2)
                    if d<min_d and d<=match_r_px:min_d=d;best_m_v=s_v_a;pop_idx=i_v
                e={'ID':s_b['id'],'X_B':s_b.get('x'),'Y_B':s_b.get('y'),'RA_deg':s_b.get('ra_deg'),'Dec_deg':s_b.get('dec_deg'),'InstrMag_B':s_b.get('instrumental_mag'),'StdMag_B':s_b.get('StdMag_B'),'InstrMag_V':None,'StdMag_V':None,'B-V':None,'X_V':None,'Y_V':None}
                if best_m_v: e.update({'InstrMag_V':best_m_v.get('instrumental_mag'),'StdMag_V':best_m_v.get('StdMag_V'),'X_V':best_m_v.get('x'),'Y_V':best_m_v.get('y')});
                if e['StdMag_B'] is not None and e['StdMag_V'] is not None: e['B-V']=e['StdMag_B']-e['StdMag_V']
                if e['InstrMag_V'] is not None: matched_prev.append({'x':s_b['x'],'y':s_b['y'],'id':e['ID']});
                if best_m_v and pop_idx!=-1: avail_v.pop(pop_idx)
                results.append(e)
        results.sort(key=lambda s:s.get('StdMag_B') if s.get('StdMag_B') is not None else (s.get('InstrMag_B') if s.get('InstrMag_B') is not None else float('inf')))
        status_messages.append(f"Cross-matched. {len(matched_prev)} pairs.")
        df_data=[[round(r.get(col),3) if isinstance(r.get(col),float) else r.get(col) for col in display_columns] for r in results]
        final_results_df_for_ui=pd.DataFrame(df_data,columns=display_columns); status_messages.append("Table prepared.")
        if corrected_b_data is not None:
            plt.figure(figsize=(8,8));norm=ImageNormalize(corrected_b_data,interval=ZScaleInterval());plt.imshow(corrected_b_data,cmap='gray',origin='lower',norm=norm);plt.colorbar(label="Pixel Value")
            plt.title(f"B-Prev: {os.path.basename(raw_b_path)}");plt.xlabel("X");plt.ylabel("Y")
            if roi_x is not None: plt.gca().add_patch(plt.Circle((roi_x,roi_y),roi_r,color='blue',fill=False,ls='--',label='ROI'))
            added_labels=set()
            for star in matched_prev: lbl='Matched Source' if 'Matched Source' not in added_labels else None; plt.gca().add_patch(plt.Circle((star['x'],star['y']),10,color='lime',fill=False,alpha=0.7,label=lbl)); plt.text(star['x']+12,star['y']+12,str(star['id']),color='lime',fontsize=9); added_labels.add('Matched Source')
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles: plt.legend(handles, labels)
            preview_image_path_for_ui=os.path.join(previews_dir,f"tab4_b_preview_{time.strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(preview_image_path_for_ui);plt.close();status_messages.append(f"Preview: {os.path.basename(preview_image_path_for_ui)}")
        if results:
            obj_name=b_header.get('OBJECT','UnknownObject').strip().replace(' ','_');csv_fn=f"{obj_name}_phot_results_{time.strftime('%Y%m%d_%H%M%S')}.csv";csv_file_path_for_ui=os.path.join(csv_output_dir,csv_fn)
            final_results_df_for_ui.to_csv(csv_file_path_for_ui,index=False,float_format='%.3f');status_messages.append(f"CSV: {os.path.basename(csv_file_path_for_ui)}")
        status_messages.append("Tab 4 Analysis Completed.")
    except Exception as e_main: status_messages.append(f"CRITICAL ERROR Tab 4: {str(e_main)}"); final_results_df_for_ui=pd.DataFrame(columns=display_columns); preview_image_path_for_ui=None; csv_file_path_for_ui=None
    finally:
        if temp_dir_obj: temp_dir_obj.cleanup()
    return "\\n".join(status_messages),final_results_df_for_ui,preview_image_path_for_ui,csv_file_path_for_ui

# Handler for Tab 5: H-R Diagram
def handle_tab5_hr_diagram(csv_file_obj, object_name_str): # Removed request: gr.Request
    if not csv_file_obj: return "Error: No CSV file uploaded.", None
    status_messages = ["Processing H-R Diagram..."]; hr_diagram_image_path = None
    try:
        csv_file_path = csv_file_obj.name; df = pd.read_csv(csv_file_path)
        v_mag_col_options = ['StdMag_V', 'StdMag V']; bv_color_col_options = ['B-V', 'B_V']
        v_mag_col = next((col for col in v_mag_col_options if col in df.columns), None)
        bv_color_col = next((col for col in bv_color_col_options if col in df.columns), None)
        if not v_mag_col or not bv_color_col: raise ValueError(f"CSV must contain V-mag (e.g., 'StdMag_V') and B-V color (e.g., 'B-V'). Found: {list(df.columns)}")
        df_cleaned = df[[v_mag_col, bv_color_col]].dropna()
        v_mags = df_cleaned[v_mag_col].tolist(); bv_colors = df_cleaned[bv_color_col].tolist()
        if not v_mags or not bv_colors: raise ValueError("No valid data pairs for V-mag and B-V color in CSV.")
        status_messages.append(f"Extracted {len(v_mags)} valid data points.")
        plot_title = "H-R Diagram"; obj_name_from_csv = None
        if 'OBJECT_NAME' in df.columns and df['OBJECT_NAME'].notna().any() and str(df['OBJECT_NAME'].dropna().iloc[0]).strip():
             obj_name_from_csv = str(df['OBJECT_NAME'].dropna().iloc[0]).strip()
        final_object_name = object_name_str.strip() if object_name_str and object_name_str.strip() else obj_name_from_csv
        if final_object_name: plot_title = f"{final_object_name} H-R Diagram"

        # Define output directory for H-R diagrams (user-facing)
        hr_diagram_output_dir = os.path.join("calibrated_lights_output", "hr_diagrams")
        os.makedirs(hr_diagram_output_dir, exist_ok=True)

        safe_title_part = "".join(c if c.isalnum() or c in ['_','-'] else '' for c in (final_object_name if final_object_name else "HR_Diagram")).replace(" ","_")
        hr_diagram_image_path_val = os.path.join(hr_diagram_output_dir, f"{safe_title_part if safe_title_part else 'HR_Diagram'}_{time.strftime('%Y%m%d_%H%M%S')}.png")

        if plot_hr_diagram(
            magnitudes=v_mags,
            colors=bv_colors,
            colors_data=bv_colors,
            output_image_path=hr_diagram_image_path_val,
            title=plot_title
        ):
            status_messages.append(f"H-R Diagram generated: {os.path.basename(hr_diagram_image_path_val)}"); hr_diagram_image_path = hr_diagram_image_path_val
        else: status_messages.append("Error: plot_hr_diagram function failed to generate H-R plot."); hr_diagram_image_path = None
    except Exception as e: status_messages.append(f"Error during H-R diagram generation: {str(e)}"); hr_diagram_image_path = None
    return "\\n".join(status_messages), hr_diagram_image_path

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
                download_master_bias = gr.File(label="Download Master BIAS", interactive=False, visible=False, elem_id="tab1_dl_mbias")
                download_master_darks_display = gr.Files(label="Download Generated/Uploaded Master DARKs", interactive=False, visible=False, elem_id="tab1_dl_mdarks_files")
                download_master_flats_display = gr.Files(label="Download Generated/Uploaded Prelim. Master FLATs", interactive=False, visible=False, elem_id="tab1_dl_mflats_files")

            generate_master_bias_button.click(fn=handle_generate_master_bias, inputs=[bias_uploads, master_bias_path_state], outputs=[tab1_status_display, master_bias_path_state, download_master_bias])
            generate_master_dark_button.click(fn=handle_generate_master_dark, inputs=[dark_uploads, master_bias_path_state, master_dark_paths_state], outputs=[tab1_status_display, master_dark_paths_state, download_master_darks_display])
            generate_master_flat_button.click(fn=handle_generate_master_flat, inputs=[flat_uploads, master_flat_paths_state], outputs=[tab1_status_display, master_flat_paths_state, download_master_flats_display])

            upload_master_bias.upload(fn=handle_upload_master_bias, inputs=[upload_master_bias, master_bias_path_state], outputs=[tab1_status_display, master_bias_path_state, download_master_bias])
            upload_master_darks.upload(fn=handle_upload_master_darks, inputs=[upload_master_darks, master_dark_paths_state], outputs=[tab1_status_display, master_dark_paths_state, download_master_darks_display])
            upload_master_flats.upload(fn=handle_upload_master_flats, inputs=[upload_master_flats, master_flat_paths_state], outputs=[tab1_status_display, master_flat_paths_state, download_master_flats_display])

        with gr.TabItem("LIGHT Frame Correction (Tab 2)"):
            tab2_calibrated_files_summary_display = gr.Textbox(label="Calibrated Files List", interactive=False, lines=5, visible=False, elem_id="tab2_calibrated_list_txt")
            with gr.Row():
                with gr.Column(scale=1):
                    light_frame_uploads = gr.Files(label="Upload Raw LIGHT Frames (FITS)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab2_light_uploads")
                    stretch_options_dropdown = gr.Dropdown(label="Preview Stretch Mode", choices=["ZScale", "MinMax", "Percentile 99.5%", "Percentile 99%", "Percentile 98%", "Percentile 95%"], value="ZScale", elem_id="tab2_stretch_dropdown")
                    calibrate_lights_button = gr.Button("Calibrate Uploaded LIGHT Frames", elem_id="tab2_calibrate_btn")
                with gr.Column(scale=2):
                    tab2_status_display = gr.Textbox(label="Calibration Status", interactive=False, lines=10, elem_id="tab2_status_disp")
            gr.Markdown("### Calibrated Image Preview (First Image)")
            calibrated_light_preview = gr.Image(label="Calibrated LIGHT Frame Preview (PNG)", type="filepath", interactive=False, elem_id="tab2_preview_img", height=400, visible=True) # Set visible=True initially or handle via output
            gr.Markdown("### Download Calibrated LIGHT Frame (First Image)")
            download_calibrated_light = gr.File(label="Download Calibrated LIGHT Frame (FITS)", interactive=False, visible=False, elem_id="tab2_download_fits")
            # gr.Markdown("### List of All Calibrated Files") # This seems redundant if summary display is used.
            calibrate_lights_button.click(
                fn=handle_calibrate_lights,
                inputs=[light_frame_uploads, master_bias_path_state, master_dark_paths_state, master_flat_paths_state, stretch_options_dropdown],
                outputs=[tab2_status_display, calibrated_light_preview, download_calibrated_light, tab2_calibrated_files_summary_display]
            )

        with gr.TabItem("Extinction Coefficient (Tab 3)"):
            gr.Markdown("## Calculate Atmospheric Extinction Coefficient (k) from FITS Files")
            gr.Markdown("Upload a series of LIGHT frames of the same star taken at different airmasses. The application will calibrate them, perform photometry on the brightest star, and then calculate 'k' and 'm0'.")

            tab3_fits_uploads = gr.Files(label="Upload LIGHT Frames for Extinction Analysis (FITS)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab3_fits_uploads_extinction")
            calculate_button_tab3 = gr.Button("Calculate Extinction from FITS", elem_id="tab3_calc_ext_fits_btn")
            with gr.Row():
                k_output = gr.Textbox(label="Extinction Coefficient (k)", interactive=False, elem_id="tab3_k_output")
                k_err_output = gr.Textbox(label="Uncertainty in k (k_err)", interactive=False, elem_id="tab3_k_err_output")
                m0_output = gr.Textbox(label="Magnitude at Zero Airmass (m0)", interactive=False, elem_id="tab3_m0_output")
            tab3_status_display = gr.Textbox(label="Processing Status / Errors", interactive=False, lines=8, elem_id="tab3_status_disp_extinction")
            gr.Markdown("### Results Plot & Data Table")
            with gr.Row():
                tab3_plot_display = gr.Plot(label="Airmass vs. Magnitude Plot", visible=True, elem_id="tab3_extinction_plot") # Set visible=True or handle via output
            with gr.Row():
                tab3_results_table = gr.DataFrame(label="Photometry & Airmass Data", visible=True, headers=["Filename", "Airmass", "Instrumental_Magnitude", "Raw_Flux", "Net_Flux", "X_cen", "Y_cen", "Skipped_Reason"], wrap=True, elem_id="tab3_extinction_table")

            calculate_button_tab3.click(
                fn=handle_tab3_extinction_from_fits,
                inputs=[
                    tab3_fits_uploads,
                    master_bias_path_state,
                    master_dark_paths_state,
                    master_flat_paths_state
                ],
                outputs=[
                    tab3_status_display,
                    k_output,
                    k_err_output,
                    m0_output,
                    tab3_plot_display,
                    tab3_results_table
                ]
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
                    gr.Markdown("### Photometry Settings")
                    tab4_fwhm_input = gr.Number(label="FWHM (pixels)", value=4.0, minimum=1.0, step=0.1, elem_id="tab4_fwhm", info="Full Width at Half Maximum for source detection/photometry.")
                    tab4_aperture_radius_input = gr.Number(label="Aperture Radius (pixels)", value=5.0, minimum=1.0, step=0.1, elem_id="tab4_ap_radius")
                    tab4_sky_inner_input = gr.Number(label="Sky Annulus Inner Radius (pixels)", value=8.0, minimum=1.0, step=0.1, elem_id="tab4_sky_inner")
                    tab4_sky_outer_input = gr.Number(label="Sky Annulus Outer Radius (pixels)", value=12.0, minimum=1.0, step=0.1, elem_id="tab4_sky_outer")
                    gr.Markdown("### Atmospheric Extinction Coefficients")
                    tab4_k_b_input = gr.Textbox(label="Extinction Coefficient k(B)", value="0.22", elem_id="tab4_k_b_val", info="Atmospheric extinction k-value for B-filter observations.")
                    tab4_k_v_input = gr.Textbox(label="Extinction Coefficient k(V)", value="0.12", elem_id="tab4_k_v_val", info="Atmospheric extinction k-value for V-filter observations.")
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
                    tab4_results_table = gr.DataFrame(label="Photometry Results", headers=["ID", "X_B", "Y_B", "RA_deg", "Dec_deg", "InstrMag_B", "InstrMag_V", "StdMag_B", "StdMag_V", "B-V"], interactive=False, wrap=True, elem_id="tab4_results_df")
                    tab4_csv_download = gr.File(label="Download Results as CSV", interactive=False, visible=False, elem_id="tab4_csv_dl")
                with gr.Column(scale=1):
                    gr.Markdown("### Preview (B-filter)")
                    tab4_preview_image = gr.Image(label="B-filter Preview with Detections/ROI", type="filepath", interactive=False, height=400, visible=True, elem_id="tab4_preview_img_b") # Set visible=True
            tab4_status_display = gr.Textbox(label="Status / Errors", lines=5, interactive=False, elem_id="tab4_status_text")
            tab4_run_button.click(
                fn=handle_tab4_photometry,
                inputs=[
                    tab4_b_frame_upload, tab4_v_frame_upload,
                    tab4_std_star_fits_upload, tab4_std_b_mag_input, tab4_std_v_mag_input,
                    tab4_roi_input,
                    tab4_fwhm_input, tab4_aperture_radius_input,
                    tab4_sky_inner_input, tab4_sky_outer_input,
                    tab4_k_b_input, tab4_k_v_input, # Pass new k-value inputs
                    master_bias_path_state, master_dark_paths_state, master_flat_paths_state
                ],
                outputs=[tab4_status_display, tab4_results_table, tab4_preview_image, tab4_csv_download]
            )

        with gr.TabItem("H-R Diagram (Tab 5)"):
            gr.Markdown("## Tab 5: H-R Diagram (Color-Magnitude Diagram)")
            gr.Markdown("Upload the CSV file generated by Tab 4 (which should contain 'StdMag_V' and 'B-V' columns). Optionally, provide the object name if not in CSV or for a custom title.")
            with gr.Row():
                with gr.Column(scale=1):
                    tab5_csv_upload = gr.File(label="Upload Photometry CSV File", file_types=['.csv'], type="filepath", elem_id="tab5_csv_upload")
                    tab5_object_name_input = gr.Textbox(label="Object Name (for Diagram Title)", placeholder="e.g., M45, NGC1234, or from CSV if column exists", elem_id="tab5_obj_name")
                    tab5_plot_hr_button = gr.Button("Generate H-R Diagram", variant="primary", elem_id="tab5_plot_btn")
                with gr.Column(scale=2):
                    tab5_hr_diagram_display = gr.Image(label="H-R Diagram", type="filepath", interactive=False, height=500, elem_id="tab5_hr_img", visible=True) # Set visible=True
                    tab5_status_display = gr.Textbox(label="Status / Errors", lines=3, interactive=False, elem_id="tab5_hr_status")
            tab5_plot_hr_button.click(
                fn=handle_tab5_hr_diagram,
                inputs=[tab5_csv_upload, tab5_object_name_input],
                outputs=[tab5_status_display, tab5_hr_diagram_display]
            )

if __name__ == '__main__':
    try: import scipy # Check for a common dependency to indicate environment readiness
    except ImportError: print("WARNING: Key dependency (scipy) not found. Some app features might fail.")
    astro_app.launch()
