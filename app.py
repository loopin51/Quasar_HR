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
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize # For PNG preview
from astropy.wcs import WCS
from astropy.table import Table # DAOStarFinder might return this
from astropy.coordinates import SkyCoord # For cross-matching
import astropy.units as u # For SkyCoord units
from photutils.detection import DAOStarFinder

from utils import load_fits_data, get_fits_header, save_fits_data
from tab1_functions import create_master_frame
from tab2_functions import correct_light_frame
from tab3_functions import calculate_extinction_coefficient
from tab4_functions import perform_photometry
from tab5_functions import plot_hr_diagram
from scipy.stats import linregress # For Tab 3

# --- Helper for cleanup in handlers ---
def _try_remove(file_path):
    if file_path and os.path.exists(file_path):
        try: os.remove(file_path)
        except Exception as e: print(f"Warning: Could not remove temp file {file_path}: {e}")

def _try_rmdir_if_empty(dir_path):
    if dir_path and os.path.exists(dir_path) and not os.listdir(dir_path):
        try: os.rmdir(dir_path)
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
    print(f"Calling create_master_frame for BIAS with files: {uploaded_file_paths} to output: {output_master_bias_path}")
    try:
        success = create_master_frame(file_paths=uploaded_file_paths, output_path=output_master_bias_path, method="median", frame_type="BIAS")
    except Exception as e_create: return f"Exception during Master BIAS generation: {e_create}", current_master_bias_path_state, gr.File(visible=False)
    if success:
        print(f"Master BIAS generated successfully: {output_master_bias_path}")
        return f"Master BIAS generated: {output_master_bias_path}", output_master_bias_path, gr.File(value=output_master_bias_path, label=f"Download Master BIAS ({os.path.basename(output_master_bias_path)})", visible=True, interactive=True)
    else:
        _try_remove(output_master_bias_path)
        return "Failed to generate Master BIAS. Check console/logs.", current_master_bias_path_state, gr.File(visible=False)

# --- Handler for Tab 1: Master DARKs ---
def handle_generate_master_dark(dark_uploads_list, master_bias_path, current_master_dark_paths_state):
    print(f"Master BIAS path received in handle_generate_master_dark: {master_bias_path}")
    if not dark_uploads_list: return "No DARK files uploaded.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)
    if not master_bias_path or not os.path.exists(master_bias_path): return "Master BIAS not available or path invalid.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)
    master_bias_data = load_fits_data(master_bias_path)
    if master_bias_data is None: return f"Failed to load Master BIAS data from {master_bias_path}.", current_master_dark_paths_state or {}, gr.Textbox(visible=False)
    output_dir = "masters_output"; temp_subtracted_dark_dir = os.path.join(output_dir, "temp_subtracted_darks")
    os.makedirs(temp_subtracted_dark_dir, exist_ok=True); print(f"Ensured temp directory exists: {temp_subtracted_dark_dir}")
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
        print(f"Processing DARKs for exposure: {exptime_str} ({len(file_list)} files)"); bias_subtracted_dark_paths = []
        for i, raw_dark_path_iter in enumerate(file_list):
            raw_dark_data = load_fits_data(raw_dark_path_iter)
            if raw_dark_data is None: status_messages.append(f"Warning: Failed to load raw DARK {raw_dark_path_iter}."); continue
            if raw_dark_data.shape != master_bias_data.shape: status_messages.append(f"Warning: Shape mismatch DARK {raw_dark_path_iter} vs BIAS."); continue
            subtracted_data = raw_dark_data - master_bias_data
            temp_dark_filename = f"bias_subtracted_dark_{exptime_str}_{i}.fits"; temp_path = os.path.join(temp_subtracted_dark_dir, temp_dark_filename)
            original_header = get_fits_header(raw_dark_path_iter)
            if original_header: original_header.add_history(f"BIAS subtracted using {os.path.basename(master_bias_path)}")
            if save_fits_data(temp_path, subtracted_data, header=original_header): bias_subtracted_dark_paths.append(temp_path)
            else: status_messages.append(f"Warning: Failed to save BIAS-subtracted DARK {temp_path}.")
        if not bias_subtracted_dark_paths: status_messages.append(f"No valid BIAS-subtracted DARKs for exp {exptime_str}."); continue
        output_master_dark_path = os.path.join(output_dir, f"master_dark_{exptime_str}.fits")
        success = create_master_frame(file_paths=bias_subtracted_dark_paths, output_path=output_master_dark_path, method="median", frame_type=f"DARK_{exptime_str}")
        if success: new_master_dark_paths[exptime_str] = output_master_dark_path; status_messages.append(f"Master DARK {exptime_str} gen: {output_master_dark_path}")
        else: status_messages.append(f"Failed Master DARK gen for {exptime_str}."); _try_remove(output_master_dark_path)
        for temp_f_path in bias_subtracted_dark_paths: _try_remove(temp_f_path)
    _try_rmdir_if_empty(temp_subtracted_dark_dir)
    final_status = "\\n".join(status_messages) if status_messages else "Processing completed."
    if not new_master_dark_paths and not status_messages: final_status += "\\nNo Master DARKs generated." # If no messages and no paths, means no valid groups.
    elif not new_master_dark_paths: final_status += "\\nNo Master DARKs were successfully generated." # If messages but no paths, means all groups failed.
    dark_paths_display_text = "Generated Master DARKs:\\n" + "\\n".join([f"{exp}: {p}" for exp, p in new_master_dark_paths.items()])
    return final_status, new_master_dark_paths, gr.Textbox(value=dark_paths_display_text, label="Generated Master DARK Paths", visible=True, interactive=False)

# --- Handler for Tab 1: Master FLATs ---
def handle_generate_master_flat(flat_uploads_list, current_master_flat_paths_state):
    if not flat_uploads_list: return "No FLAT files uploaded.", current_master_flat_paths_state or {}, gr.Textbox(visible=False)
    output_dir = "masters_output"; os.makedirs(output_dir, exist_ok=True); print(f"Ensured output directory exists: {output_dir}")
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
        print(f"Processing Prelim Master FLATs for filter: {filter_name_key} ({len(file_list)} files)")
        output_master_flat_path = os.path.join(output_dir, f"prelim_master_flat_{filter_name_key}.fits")
        try:
            success = create_master_frame(file_paths=file_list, output_path=output_master_flat_path, method="median", frame_type=f"PRELIM_FLAT_{filter_name_key.upper()}")
        except Exception as e_create_flat: error_msg = f"Exception for Prelim FLAT {filter_name_key}: {e_create_flat}"; print(error_msg); status_messages.append(error_msg); success = False
        if success: new_master_flat_paths[filter_name_key] = output_master_flat_path; status_messages.append(f"Prelim Master FLAT '{filter_name_key}' gen: {output_master_flat_path}")
        else: status_messages.append(f"Failed Prelim Master FLAT for '{filter_name_key}'."); _try_remove(output_master_flat_path)
    final_status = "\\n".join(status_messages) if status_messages else "Flat processing completed."
    if not new_master_flat_paths and not status_messages: final_status = "No Prelim Master FLATs generated or no valid flats."
    elif not new_master_flat_paths: final_status += "\\nNo Prelim Master FLATs successfully generated."
    flat_paths_display_text = "Generated Prelim. Master FLATs:\\n" + "\\n".join([f"{filt}: {p}" for filt, p in new_master_flat_paths.items()])
    return final_status, new_master_flat_paths, gr.Textbox(value=flat_paths_display_text, label="Generated Prelim. Master FLAT Paths", visible=True, interactive=False)

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
            base_name_part = os.path.basename(file_obj.name); safe_base_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name_part)
            destination_filename = f"uploaded_master_dark_{exp_time_key}_{safe_base_name}"; destination_path = os.path.join(output_dir, destination_filename)
            shutil.copy(file_obj.name, destination_path); new_master_dark_paths[exp_time_key] = destination_path
            status_messages.append(f"Uploaded DARK '{base_name_part}' (key: {exp_time_key}): {destination_path}"); print(f"Uploaded DARK '{base_name_part}' -> {destination_path} (key {exp_time_key})")
        except Exception as e: error_msg = f"Error copying DARK {file_obj.name}: {e}"; status_messages.append(error_msg); print(error_msg)
    final_status = "\\n".join(status_messages) if status_messages else "No files processed."
    if not new_master_dark_paths and status_messages: final_status = "\\n".join(status_messages)
    elif not new_master_dark_paths and not status_messages: final_status = "No dark files processed."
    elif not status_messages and new_master_dark_paths: final_status = "All darks uploaded."
    dark_paths_display_text = "Uploaded/Updated Master DARKs:\\n" + "\\n".join([f"{exp}: {p}" for exp, p in new_master_dark_paths.items()])
    return final_status, new_master_dark_paths, gr.Textbox(value=dark_paths_display_text, label="Uploaded Master DARK Paths", visible=True, interactive=False)

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
            status_messages.append(f"Uploaded FLAT '{base_name_part}' (key: {filter_key}): {destination_path}"); print(f"Uploaded FLAT '{base_name_part}' -> {destination_path} (key {filter_key})")
        except Exception as e: error_msg = f"Error copying FLAT {file_obj.name}: {e}"; status_messages.append(error_msg); print(error_msg)
    final_status = "\\n".join(status_messages) if status_messages else "No files processed."
    if not new_master_flat_paths and status_messages: final_status = "\\n".join(status_messages)
    elif not new_master_flat_paths and not status_messages: final_status = "No flat files processed."
    elif not status_messages and new_master_flat_paths: final_status = "All flats uploaded."
    flat_paths_display_text = "Uploaded/Updated Master FLATs:\\n" + "\\n".join([f"{filt}: {p}" for filt, p in new_master_flat_paths.items()])
    return final_status, new_master_flat_paths, gr.Textbox(value=flat_paths_display_text, label="Uploaded Master FLAT Paths", visible=True, interactive=False)

# --- Handler for Tab 2: LIGHT Frame Correction ---
def handle_calibrate_lights(light_uploads_list, mbias_path, mdark_paths_dict, mflat_paths_dict, request: gr.Request):
    if not light_uploads_list: return "No LIGHT frames uploaded for calibration.", None, None
    status_messages = [f"Received {len(light_uploads_list)} LIGHT frames for calibration."]
    status_messages.append(f"Master BIAS path from state: {mbias_path}")
    status_messages.append(f"Master DARK paths from state: {mdark_paths_dict}")
    status_messages.append(f"Prelim. Master FLAT paths from state: {mflat_paths_dict}")
    status_messages.append("\\n(Full calibration logic not yet implemented in this handler.)")
    return "\\n".join(status_messages), None, None

# --- Handler for Tab 3: Extinction Coefficient ---
def handle_extinction_calculation(airmass_str: str, magnitude_str: str) -> tuple[str, str, str, str]:
    try:
        if not airmass_str.strip() or not magnitude_str.strip(): return "", "", "", "Error: Airmass and Magnitude fields cannot be empty."
        airmasses = [float(x.strip()) for x in airmass_str.split(',')]
        magnitudes = [float(x.strip()) for x in magnitude_str.split(',')]
        if len(airmasses) != len(magnitudes): return "", "", "", "Error: Mismatched list lengths."
        if len(airmasses) < 2: return "", "", "", "Error: At least two data points needed."
        # from scipy.stats import linregress # Already imported globally now
        regression_result = linregress(airmasses, magnitudes)
        k = -regression_result.slope; m0 = regression_result.intercept; k_err = regression_result.stderr
        return f"{k:.4f}", f"{k_err:.4f}", f"{m0:.4f}", ""
    except ValueError as ve: return "", "", "", f"Error: Invalid input. Ensure comma-separated numbers. Details: {ve}"
    except Exception as e: return "", "", "", f"An unexpected error occurred: {e}"

# --- Handler for Tab 4: Photometry ---
def handle_tab4_photometry(
    tab4_b_file_obj, tab4_v_file_obj,
    tab4_std_star_file_obj, tab4_std_b_mag_str, tab4_std_v_mag_str,
    tab4_roi_str, tab4_k_value_str,
    master_bias_path_state, master_dark_paths_state, master_flat_paths_state,
    request: gr.Request
):
    status_messages = ["Starting Tab 4 Photometry Analysis..."]
    final_results_df_for_ui = None; preview_image_path_for_ui = None; csv_file_path_for_ui = None
    display_columns = ['ID', 'X_B', 'Y_B', 'RA_deg', 'Dec_deg', 'InstrMag_B', 'InstrMag_V', 'StdMag_B', 'StdMag_V', 'B-V']
    if master_dark_paths_state is None: master_dark_paths_state = {}
    if master_flat_paths_state is None: master_flat_paths_state = {}
    try:
        if not tab4_b_file_obj or not tab4_v_file_obj: raise ValueError("Both B and V frames must be uploaded.")
        raw_b_path = tab4_b_file_obj.name; raw_v_path = tab4_v_file_obj.name
        status_messages.append(f"B: {os.path.basename(raw_b_path)}, V: {os.path.basename(raw_v_path)}")
        b_data_raw = load_fits_data(raw_b_path); b_header = get_fits_header(raw_b_path)
        v_data_raw = load_fits_data(raw_v_path); v_header = get_fits_header(raw_v_path)
        if b_data_raw is None or b_header is None: raise ValueError(f"Could not load B-frame/header: {raw_b_path}")
        if v_data_raw is None or v_header is None: raise ValueError(f"Could not load V-frame/header: {raw_v_path}")
        status_messages.append("B,V LIGHTs loaded.")
        if not master_bias_path_state or not os.path.exists(master_bias_path_state): raise ValueError("Master BIAS missing/invalid.")
        master_bias_data = load_fits_data(master_bias_path_state)
        if master_bias_data is None: raise ValueError(f"Failed to load Master BIAS: {master_bias_path_state}")
        status_messages.append(f"Master BIAS loaded.")
        temp_dir = os.path.join("masters_output", "temp_final_flats_tab4"); calibrated_dir = os.path.join("calibrated_lights_output", "tab4_corrected")
        os.makedirs(temp_dir, exist_ok=True); os.makedirs(calibrated_dir, exist_ok=True)

        def _correct_science_frame(label, raw_path, header, data_raw_arg_unused):
            status_messages.append(f"Correcting {label}-frame..."); exptime_val = header.get('EXPTIME', header.get('EXPOSURE')); filter_name = header.get('FILTER', header.get('FILTER1', header.get('FILTNAME')))
            if exptime_val is None: raise ValueError(f"{label}-frame missing EXPTIME.");
            if filter_name is None: raise ValueError(f"{label}-frame missing FILTER.")
            exptime_float = float(exptime_val); filter_key = str(filter_name).strip().replace(" ", "_")
            dark_exp_key = str(exptime_float).replace('.', 'p') + "s"; frame_dark_path = master_dark_paths_state.get(dark_exp_key); frame_dark_data = None
            if frame_dark_path and os.path.exists(frame_dark_path):
                frame_dark_data = load_fits_data(frame_dark_path)
                if frame_dark_data is None: status_messages.append(f"Warn: Failed to load {label}-DARK {frame_dark_path}.")
                else: status_messages.append(f"Using {label}-DARK {frame_dark_path} for {label}-flat.")
            else: status_messages.append(f"Warn: {label}-DARK for {exptime_float}s (key {dark_exp_key}) not in {list(master_dark_paths_state.keys()) if master_dark_paths_state else 'empty'}. No dark for {label}-flat.")
            prelim_flat_p = master_flat_paths_state.get(filter_key); final_flat_p = None
            if prelim_flat_p and os.path.exists(prelim_flat_p):
                prelim_flat_d = load_fits_data(prelim_flat_p)
                if prelim_flat_d is not None and master_bias_data is not None:
                    if prelim_flat_d.shape!=master_bias_data.shape: raise ValueError(f"{label}-PrelimFlat shape mismatch BIAS")
                    f_tmp1 = prelim_flat_d - master_bias_data; f_tmp2 = f_tmp1
                    if frame_dark_data is not None:
                        if f_tmp1.shape!=frame_dark_data.shape: raise ValueError(f"{label}-Flat(bias-sub) shape mismatch {label}-Dark")
                        f_tmp2 = f_tmp1 - frame_dark_data
                    med_f_tmp2 = np.median(f_tmp2);
                    if med_f_tmp2==0: raise ValueError(f"Median of processed {label}-FLAT is zero.")
                    final_flat_d = f_tmp2 / med_f_tmp2; final_flat_p = os.path.join(temp_dir, f"final_{label}_flat_{filter_key}.fits")
                    tmp_flat_h = get_fits_header(prelim_flat_p) if get_fits_header(prelim_flat_p) else fits.Header()
                    tmp_flat_h['HISTORY'] = f'Final {label}-flat for Tab4'; save_fits_data(final_flat_p, final_flat_d, header=tmp_flat_h)
                    status_messages.append(f"Temp final {label}-FLAT: {final_flat_p}")
                else: status_messages.append(f"Warn: Failed load Prelim {label}-FLAT or BIAS. {label}-frame not flat-fielded.")
            else: status_messages.append(f"Warn: Prelim {label}-FLAT for '{filter_key}' not in {list(master_flat_paths_state.keys()) if master_flat_paths_state else 'empty'}. {label}-frame not flat-fielded.")
            base_n = os.path.splitext(os.path.basename(raw_path))[0]; corrected_p = os.path.join(calibrated_dir, f"{base_n}_cal_{label}.fits")
            actual_dark_for_light = frame_dark_path if (frame_dark_path and os.path.exists(frame_dark_path)) else None
            if not correct_light_frame(raw_path, corrected_p, master_bias_path_state, actual_dark_for_light, final_flat_p):
                raise ValueError(f"Failed to correct {label}-frame.")
            status_messages.append(f"{label}-frame corrected: {corrected_p}"); _try_remove(final_flat_p)
            return corrected_p

        corrected_b_path = _correct_science_frame("B", raw_b_path, b_header, b_data_raw)
        corrected_v_path = _correct_science_frame("V", raw_v_path, v_header, v_data_raw)
        corrected_b_data = load_fits_data(corrected_b_path) if corrected_b_path else None
        corrected_v_data = load_fits_data(corrected_v_path) if corrected_v_path else None
        if corrected_b_data is None or corrected_v_data is None: raise ValueError("Failed to load corrected B or V frame.")

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
                corr_std_p = _correct_science_frame("STD",std_path,std_hdr_raw,load_fits_data(std_path))
                if not corr_std_p: raise ValueError("Std star correction failed.")
                corr_std_d = load_fits_data(corr_std_p);
                if corr_std_d is None: raise ValueError("Failed load corrected std star data.")
                mean_s,med_s,std_s=sigma_clipped_stats(corr_std_d,sigma=3.0); fwhm_s=5.0; dao_s=DAOStarFinder(fwhm=fwhm_s,threshold=5.*std_s)
                srcs_s_tbl=dao_s(corr_std_d-med_s);
                if not srcs_s_tbl or len(srcs_s_tbl)==0: raise ValueError("No sources in std star img.")
                srcs_s_tbl.sort('flux',reverse=True); std_x,std_y=srcs_s_tbl[0]['xcentroid'],srcs_s_tbl[0]['ycentroid']
                ap_rad_s=fwhm_s*1.5; std_phot_res=perform_photometry(corr_std_d,[(std_x,std_y)],ap_rad_s,ap_rad_s+3,ap_rad_s+8)
                if not std_phot_res or 'instrumental_mag' not in std_phot_res[0] or std_phot_res[0]['instrumental_mag'] is None: raise ValueError("Photometry failed for std star.")
                instr_mag_s=std_phot_res[0]['instrumental_mag']; k_val=float(tab4_k_value_str.strip()) if tab4_k_value_str.strip() else 0.15
                std_airmass=float(std_hdr_raw.get('AIRMASS',1.0))
                if std_filt_key.startswith('B'): m0_B=std_b_known-instr_mag_s+(k_val*std_airmass); status_messages.append(f"Calib m0_B={m0_B:.3f}")
                elif std_filt_key.startswith('V'): m0_V=std_v_known-instr_mag_s+(k_val*std_airmass); status_messages.append(f"Calib m0_V={m0_V:.3f}")
                else: status_messages.append(f"Warn: Std star filt '{std_filt_key}' not B/V.")
                _try_remove(corr_std_p)
            except Exception as e_std: status_messages.append(f"Std star error: {e_std}. Using default m0s.")

        srcs_b_tbl=None;
        if corrected_b_data is not None: _,med_b,std_b=sigma_clipped_stats(corrected_b_data,sigma=3.0);daofind_b=DAOStarFinder(fwhm=4.0,threshold=5.*std_b);srcs_b_tbl=daofind_b(corrected_b_data-med_b); status_messages.append(f"B-srcs: {len(srcs_b_tbl) if srcs_b_tbl else 0}")
        srcs_v_tbl=None;
        if corrected_v_data is not None: _,med_v,std_v=sigma_clipped_stats(corrected_v_data,sigma=3.0);daofind_v=DAOStarFinder(fwhm=4.0,threshold=5.*std_v);srcs_v_tbl=daofind_v(corrected_v_data-med_v); status_messages.append(f"V-srcs: {len(srcs_v_tbl) if srcs_v_tbl else 0}")

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

        ap_r,sky_i,sky_o = 5.0,8.0,12.0
        if srcs_b_tbl and len(srcs_b_tbl)>0: coords_b=np.array([(s['xcentroid'],s['ycentroid']) for s in srcs_b_tbl]); phot_b=perform_photometry(corrected_b_data,coords_b,ap_r,sky_i,sky_o)
        if srcs_v_tbl and len(srcs_v_tbl)>0: coords_v=np.array([(s['xcentroid'],s['ycentroid']) for s in srcs_v_tbl]); phot_v=perform_photometry(corrected_v_data,coords_v,ap_r,sky_i,sky_o)

        k_val=float(tab4_k_value_str.strip()) if tab4_k_value_str.strip() else 0.15
        air_b=float(b_header.get('AIRMASS',1.0)); air_v=float(v_header.get('AIRMASS',1.0))
        for r_d in phot_b:
            if r_d.get('instrumental_mag') is not None: r_d['StdMag_B']=r_d['instrumental_mag']+m0_B-(k_val*air_b)
            if b_header: try: w=WCS(b_header);_=[r_d.update({'ra_deg':rd.ra.deg,'dec_deg':rd.dec.deg}) for rd in [w.pixel_to_world(r_d['x'],r_d['y'])] if w.is_celestial] except:pass
        for r_d in phot_v:
            if r_d.get('instrumental_mag') is not None: r_d['StdMag_V']=r_d['instrumental_mag']+m0_V-(k_val*air_v)
            if v_header: try: w=WCS(v_header);_=[r_d.update({'ra_deg':rd.ra.deg,'dec_deg':rd.dec.deg}) for rd in [w.pixel_to_world(r_d['x'],r_d['y'])] if w.is_celestial] except:pass
        status_messages.append("Std Mags Calc.")

        results,matched_prev = [],[]; match_r_px=3.0
        use_sky = all(p.get('ra_deg') is not None for p in phot_b) and all(p.get('ra_deg') is not None for p in phot_v)
        if use_sky:
            status_messages.append("RA/Dec cross-match."); sky_b=SkyCoord([p['ra_deg'] for p in phot_b]*u.deg,[p['dec_deg'] for p in phot_b]*u.deg); sky_v=SkyCoord([p['ra_deg'] for p in phot_v]*u.deg,[p['dec_deg'] for p in phot_v]*u.deg)
            idx,d2d,_=sky_b.match_to_catalog_sky(sky_v); p_scale=abs(b_header.get('CDELT1',b_header.get('CD1_1',0.5))*3600); tol=match_r_px*p_scale*u.arcsec; v_matched_indices=set() # Added CD1_1 as fallback
            for i,s_b in enumerate(phot_b):
                e={'ID':i+1,'X_B':s_b.get('x'),'Y_B':s_b.get('y'),'RA_deg':s_b.get('ra_deg'),'Dec_deg':s_b.get('dec_deg'),'InstrMag_B':s_b.get('instrumental_mag'),'StdMag_B':s_b.get('StdMag_B'),'InstrMag_V':None,'StdMag_V':None,'B-V':None,'X_V':None,'Y_V':None}
                if d2d[i]<=tol:
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
            if roi_x is not None: plt.gca().add_patch(plt.Circle((roi_x,roi_y),roi_r,color='blue',fill=False,ls='--',label='ROI')) # Corrected label to lbl
            added_labels=set()
            for star in matched_prev: lbl='Matched Source' if 'Matched Source' not in added_labels else None; plt.gca().add_patch(plt.Circle((star['x'],star['y']),10,color='lime',fill=False,alpha=0.7,label=lbl)); plt.text(star['x']+12,star['y']+12,str(star['id']),color='lime',fontsize=9); added_labels.add('Matched Source')
            # Only add legend if there are actual labels
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles: plt.legend(handles, labels)

            prev_dir=os.path.join(calibrated_dir,"previews");os.makedirs(prev_dir,exist_ok=True);preview_image_path_for_ui=os.path.join(prev_dir,"tab4_b_preview.png")
            plt.savefig(preview_image_path_for_ui);plt.close();status_messages.append(f"Preview: {preview_image_path_for_ui}")
        if results:
            obj_name=b_header.get('OBJECT','UnknownObject').strip().replace(' ','_');csv_fn=f"{obj_name}_phot_results.csv";csv_p_temp=os.path.join(calibrated_dir,csv_fn)
            final_results_df_for_ui.to_csv(csv_p_temp,index=False,float_format='%.3f');csv_file_path_for_ui=csv_p_temp;status_messages.append(f"CSV: {csv_file_path_for_ui}")
        status_messages.append("Tab 4 Analysis Completed.")
    except Exception as e_main: status_messages.append(f"CRITICAL ERROR Tab 4: {str(e_main)}"); final_results_df_for_ui=pd.DataFrame(columns=display_columns); preview_image_path_for_ui=None; csv_file_path_for_ui=None
    return "\\n".join(status_messages),final_results_df_for_ui,preview_image_path_for_ui,csv_file_path_for_ui

# Handler for Tab 5: H-R Diagram
def handle_tab5_hr_diagram(csv_file_obj, object_name_str, request: gr.Request): # Added request
    if not csv_file_obj:
        return "Error: No CSV file uploaded.", None

    status_messages = ["Processing H-R Diagram..."]
    hr_diagram_image_path = None # Initialize

    try:
        csv_file_path = csv_file_obj.name
        df = pd.read_csv(csv_file_path)

        v_mag_col_options = ['StdMag_V', 'StdMag V']
        bv_color_col_options = ['B-V', 'B_V']

        v_mag_col = next((col for col in v_mag_col_options if col in df.columns), None)
        bv_color_col = next((col for col in bv_color_col_options if col in df.columns), None)

        if not v_mag_col or not bv_color_col:
            raise ValueError(f"CSV must contain a V-magnitude column (e.g., 'StdMag_V') and a B-V color column (e.g., 'B-V'). Found: {list(df.columns)}")

        df_cleaned = df[[v_mag_col, bv_color_col]].dropna()

        v_mags = df_cleaned[v_mag_col].tolist()
        bv_colors = df_cleaned[bv_color_col].tolist()

        if not v_mags or not bv_colors:
            raise ValueError("No valid (non-NaN) data pairs for V-magnitude and B-V color found in CSV for plotting.")

        status_messages.append(f"Extracted {len(v_mags)} valid data points for H-R diagram.")

        plot_title = "H-R Diagram"
        obj_name_from_csv = None
        # Check if 'OBJECT_NAME' column exists and has at least one non-empty, non-NaN value
        if 'OBJECT_NAME' in df.columns and df['OBJECT_NAME'].notna().any() and str(df['OBJECT_NAME'].dropna().iloc[0]).strip():
             obj_name_from_csv = str(df['OBJECT_NAME'].dropna().iloc[0]).strip()

        final_object_name = object_name_str.strip() if object_name_str and object_name_str.strip() else obj_name_from_csv
        if final_object_name:
            plot_title = f"{final_object_name} H-R Diagram"

        preview_dir = os.path.join("calibrated_lights_output", "previews")
        os.makedirs(preview_dir, exist_ok=True)

        safe_title_part = "".join(c if c.isalnum() or c in ['_','-'] else '' for c in (final_object_name if final_object_name else "HR_Diagram")).replace(" ","_")
        hr_diagram_image_path_val = os.path.join(preview_dir, f"{safe_title_part if safe_title_part else 'HR_Diagram'}.png")

        if plot_hr_diagram(magnitudes=v_mags, colors=bv_colors, output_image_path=hr_diagram_image_path_val, title=plot_title):
            status_messages.append(f"H-R Diagram generated: {hr_diagram_image_path_val}")
            hr_diagram_image_path = hr_diagram_image_path_val
        else:
            status_messages.append("Error: plot_hr_diagram function failed to generate H-R plot.")
            hr_diagram_image_path = None

    except Exception as e:
        status_messages.append(f"Error during H-R diagram generation: {str(e)}")
        hr_diagram_image_path = None

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
                download_master_darks_display = gr.Textbox(label="Generated Master DARK Paths", interactive=False, visible=False, lines=3, elem_id="tab1_dl_mdarks_txt")
                download_master_flats_display = gr.Textbox(label="Generated Prelim. Master FLAT Paths", interactive=False, visible=False, lines=3, elem_id="tab1_dl_mflats_txt")

            generate_master_bias_button.click(fn=handle_generate_master_bias, inputs=[bias_uploads, master_bias_path_state], outputs=[tab1_status_display, master_bias_path_state, download_master_bias])
            generate_master_dark_button.click(fn=handle_generate_master_dark, inputs=[dark_uploads, master_bias_path_state, master_dark_paths_state], outputs=[tab1_status_display, master_dark_paths_state, download_master_darks_display])
            generate_master_flat_button.click(fn=handle_generate_master_flat, inputs=[flat_uploads, master_flat_paths_state], outputs=[tab1_status_display, master_flat_paths_state, download_master_flats_display])

            upload_master_bias.upload(fn=handle_upload_master_bias, inputs=[upload_master_bias, master_bias_path_state], outputs=[tab1_status_display, master_bias_path_state, download_master_bias])
            upload_master_darks.upload(fn=handle_upload_master_darks, inputs=[upload_master_darks, master_dark_paths_state], outputs=[tab1_status_display, master_dark_paths_state, download_master_darks_display])
            upload_master_flats.upload(fn=handle_upload_master_flats, inputs=[upload_master_flats, master_flat_paths_state], outputs=[tab1_status_display, master_flat_paths_state, download_master_flats_display])

        with gr.TabItem("LIGHT Frame Correction (Tab 2)"):
            gr.Markdown("## Calibrate Raw LIGHT Frames")

            with gr.Row():
                with gr.Column(scale=1):
                    light_frame_uploads = gr.Files(label="Upload Raw LIGHT Frames (FITS)", file_types=['.fits', '.fit'], type="filepath", elem_id="tab2_light_uploads")
                    calibrate_lights_button = gr.Button("Calibrate Uploaded LIGHT Frames", elem_id="tab2_calibrate_btn")

                with gr.Column(scale=2):
                    tab2_status_display = gr.Textbox(label="Calibration Status", interactive=False, lines=10, elem_id="tab2_status_disp")

            gr.Markdown("### Calibrated Image Preview")
            calibrated_light_preview = gr.Image(label="Calibrated LIGHT Frame Preview (PNG)", type="filepath", interactive=False, elem_id="tab2_preview_img", height=400, visible=False)

            gr.Markdown("### Download Calibrated LIGHT Frame")
            download_calibrated_light = gr.File(label="Download Calibrated LIGHT Frame (FITS)", interactive=False, visible=False, elem_id="tab2_download_fits")

            calibrate_lights_button.click(
                fn=handle_calibrate_lights,
                inputs=[light_frame_uploads, master_bias_path_state, master_dark_paths_state, master_flat_paths_state],
                outputs=[tab2_status_display, calibrated_light_preview, download_calibrated_light],
                request=True # Added request to match handler
            )

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
                    tab4_k_value_input = gr.Textbox(label="Extinction Coefficient (k)", value="0.15", elem_id="tab4_k_val")
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
                    tab4_results_table = gr.DataFrame(label="Photometry Results", headers=["ID", "X", "Y", "RA", "Dec", "InstrMag_B", "InstrMag_V", "StdMag_B", "StdMag_V", "B-V"], interactive=False, wrap=True, max_rows=10, overflow_row_behaviour='paginate', elem_id="tab4_results_df")
                    tab4_csv_download = gr.File(label="Download Results as CSV", interactive=False, visible=False, elem_id="tab4_csv_dl")
                with gr.Column(scale=1):
                    gr.Markdown("### Preview (B-filter)")
                    tab4_preview_image = gr.Image(label="B-filter Preview with Detections/ROI", type="filepath", interactive=False, height=400, visible=False, elem_id="tab4_preview_img_b")
            tab4_status_display = gr.Textbox(label="Status / Errors", lines=5, interactive=False, elem_id="tab4_status_text")
            tab4_run_button.click(
                fn=handle_tab4_photometry,
                inputs=[tab4_b_frame_upload, tab4_v_frame_upload, tab4_std_star_fits_upload, tab4_std_b_mag_input, tab4_std_v_mag_input, tab4_roi_input, tab4_k_value_input, master_bias_path_state, master_dark_paths_state, master_flat_paths_state],
                outputs=[tab4_status_display, tab4_results_table, tab4_preview_image, tab4_csv_download], # Corrected output order for preview
                request=True # Added request
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
                    tab5_hr_diagram_display = gr.Image(label="H-R Diagram", type="filepath", interactive=False, height=500, elem_id="tab5_hr_img", visible=False)
                    tab5_status_display = gr.Textbox(label="Status / Errors", lines=3, interactive=False, elem_id="tab5_hr_status")
            tab5_plot_hr_button.click(
                fn=handle_tab5_hr_diagram,
                inputs=[tab5_csv_upload, tab5_object_name_input],
                outputs=[tab5_status_display, tab5_hr_diagram_display],
                request=True
            )

# --- Handler for Tab 5: H-R Diagram ---
def handle_tab5_hr_diagram(csv_file_obj, object_name_str, request: gr.Request):
    if not csv_file_obj:
        return "Error: No CSV file uploaded.", None
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
        preview_dir = os.path.join("calibrated_lights_output", "previews"); os.makedirs(preview_dir, exist_ok=True)
        safe_title_part = "".join(c if c.isalnum() or c in ['_','-'] else '' for c in (final_object_name if final_object_name else "HR_Diagram")).replace(" ","_")
        hr_diagram_image_path_val = os.path.join(preview_dir, f"{safe_title_part if safe_title_part else 'HR_Diagram'}.png")
        if plot_hr_diagram(magnitudes=v_mags, colors=bv_colors, output_image_path=hr_diagram_image_path_val, title=plot_title):
            status_messages.append(f"H-R Diagram generated: {hr_diagram_image_path_val}"); hr_diagram_image_path = hr_diagram_image_path_val
        else: status_messages.append("Error: plot_hr_diagram function failed."); hr_diagram_image_path = None
    except Exception as e: status_messages.append(f"Error in H-R diagram: {str(e)}"); hr_diagram_image_path = None
    return "\\n".join(status_messages), hr_diagram_image_path

if __name__ == '__main__':
    try: import scipy # Check for a common dependency to indicate environment readiness
    except ImportError: print("WARNING: Key dependency (scipy) not found. Some app features might fail.")
    astro_app.launch()
