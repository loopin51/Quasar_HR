# Python script for subtask - Tab 4 Handler Part 1
import re
import os
import numpy as np # Not used directly in this script, but good to have for context
from astropy.io import fits # Not used directly, but for context

app_py_path = "app.py"

# Logic for the first part of Tab 4 handler
tab4_handler_part1_code = """
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
        return "\\n".join(status_messages), None, None, None # status, df, preview_img, csv_dl

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
        return "\\n".join(status_messages), None, None, None
    if v_data_check is None or v_header is None:
        status_messages.append(f"Error: Could not load V-filter frame or its header from {raw_v_path}.")
        return "\\n".join(status_messages), None, None, None
    status_messages.append("B and V LIGHT frames loaded successfully.")

    if not master_bias_path_state or not os.path.exists(master_bias_path_state):
        status_messages.append("Error: Master BIAS not found or path invalid. Please prepare/upload in Tab 1.")
        return "\\n".join(status_messages), None, None, None
    master_bias_data = load_fits_data(master_bias_path_state)
    if master_bias_data is None:
        status_messages.append(f"Error: Failed to load Master BIAS from {master_bias_path_state}.")
        return "\\n".join(status_messages), None, None, None
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
        return "\\n".join(status_messages), None, None, None

    status_messages.append("\\nPhotometry Part 1 (Calibration) finished. Corrected frames (if successful):")
    status_messages.append(f"Corrected B: {corrected_b_path if corrected_b_path else 'N/A'}")
    status_messages.append(f"Corrected V: {corrected_v_path if corrected_v_path else 'N/A'}")
    status_messages.append("\\nSource detection, photometry, and catalog matching not yet implemented in this part.")

    # For now, return None for DataFrame, preview image, and CSV download path
    # Preview could show the corrected B-frame if successful
    preview_image_path = corrected_b_path # or None if it failed

    return "\\n".join(status_messages), None, preview_image_path, None
""" # End of tab4_handler_part1_code string

try:
    with open(app_py_path, "r") as f:
        content = f.read()

    dummy_handler_name = "handle_run_photometry_analysis_dummy"
    actual_handler_name = "handle_tab4_photometry" # New name for the actual handler

    # Add necessary imports if missing
    # These are the imports that the new handler logic might need available in app.py's global scope
    # if they are not already imported by other functions or globally.
    imports_to_ensure = {
        "perform_photometry": "from tab4_functions import perform_photometry",
        "DAOStarFinder": "from photutils.detection import DAOStarFinder", # For Part 2
        "sigma_clipped_stats_ap": "from astropy.stats import sigma_clipped_stats", # For Part 2, aliased to avoid clash if other form used
        "np": "import numpy as np", # For Part 1 & 2
        "fits_header": "from astropy.io import fits" # Specifically for fits.Header() in Part 1
    }

    # Find the block of imports at the start of the file
    import_block_match = re.match(r"(^(?:import .*\n|from .* import .*\n)+)", content, re.MULTILINE)
    existing_imports_text = ""
    if import_block_match:
        existing_imports_text = import_block_match.group(1)

    new_imports_to_add_list = []
    for key, imp_statement in imports_to_ensure.items():
        # Check if the specific module/function or a wildcard import of its parent is present
        module_name = imp_statement.split(" import ")[0].split(" ")[1] # e.g. "numpy" from "import numpy as np"
        if f"from {module_name} import *" not in existing_imports_text and \
           imp_statement not in existing_imports_text and \
           (key != "np" or "import numpy" not in existing_imports_text) and \
           (key != "fits_header" or "from astropy.io import fits" not in existing_imports_text) and \
           (key != "sigma_clipped_stats_ap" or "from astropy.stats import sigma_clipped_stats" not in existing_imports_text):

            # More specific check for aliased imports like 'import numpy as np'
            if key == "np" and re.search(r"import\s+numpy\s+as\s+\w+", existing_imports_text): continue
            if key == "fits_header" and re.search(r"from\s+astropy\.io\s+import\s+fits", existing_imports_text): continue # Already covered by general astropy.io.fits
            if key == "sigma_clipped_stats_ap" and re.search(r"from\s+astropy\.stats\s+import\s+sigma_clipped_stats", existing_imports_text): continue


            print(f"Adding import: {imp_statement}")
            new_imports_to_add_list.append(imp_statement)

    if new_imports_to_add_list:
        # Add new imports after the existing import block
        if import_block_match:
            content = content.replace(existing_imports_text, existing_imports_text + "\n".join(new_imports_to_add_list) + "\n")
        else: # No existing import block, prepend to file (less ideal)
            content = "\n".join(new_imports_to_add_list) + "\n\n" + content
        print("Updated imports for Tab 4 handler.")


    # Replace the dummy handler name in the click event first
    # Make sure this replacement is robust and only targets the Tab 4 button click
    # Assuming tab4_run_button is unique enough for this context
    old_click_line_pattern = re.compile(r"(^\s*tab4_run_button\.click\(\s*fn=)" + re.escape(dummy_handler_name) + r"(,\s*inputs=.*,\s*outputs=.*\)\s*$)", re.MULTILINE)
    new_click_line = r"\1" + actual_handler_name + r"\2" # Uses backreferences to \1 and \2

    content, num_click_replacements = old_click_line_pattern.subn(new_click_line, content)
    if num_click_replacements > 0:
        print(f"Updated click handler in Tab 4 to use new function name: {actual_handler_name}")
    else:
        print(f"Could not find click handler for '{dummy_handler_name}' to update, or it was already updated.")

    # Now replace the dummy function definition with the new one
    # Regex to find the dummy function definition block
    dummy_func_pattern = re.compile(r"^\s*def " + re.escape(dummy_handler_name) + r"\(.*?\):\s*\n(?:^\s+.*\n|^\s*#.*?\n|^\s*\n)*", re.MULTILINE)

    new_code_block_for_sub = tab4_handler_part1_code.strip() + "\n\n"

    search_result_dummy = dummy_func_pattern.search(content)
    if search_result_dummy:
        updated_content = dummy_func_pattern.sub(new_code_block_for_sub, content, count=1)
        print(f"Replaced dummy handler '{dummy_handler_name}' with initial logic for '{actual_handler_name}'.")
    else:
        print(f"Could not find dummy handler '{dummy_handler_name}' to replace. Checking if actual handler '{actual_handler_name}' exists to replace its content.")
        actual_func_pattern = re.compile(r"^\s*def " + re.escape(actual_handler_name) + r"\(.*?\):\s*\n(?:^\s+.*\n|^\s*#.*?\n|^\s*\n)*", re.MULTILINE)
        search_result_actual = actual_func_pattern.search(content)
        if search_result_actual:
            updated_content = actual_func_pattern.sub(new_code_block_for_sub, content, count=1)
            print(f"Found existing '{actual_handler_name}' and replaced its content with initial logic.")
        else:
            # If neither found, try to insert it before 'with gr.Blocks()'
            print(f"Neither '{dummy_handler_name}' nor '{actual_handler_name}' found for replacement. Attempting fresh insert.")
            main_block_match = re.search(r"^\s*with gr\.Blocks\(\) as astro_app:", content, re.MULTILINE)
            if main_block_match:
                insert_pos = main_block_match.start()
                updated_content = content[:insert_pos] + new_code_block_for_sub + content[insert_pos:]
                print(f"Inserted '{actual_handler_name}' logic before gr.Blocks().")
            else:
                updated_content = content # No change if gr.Blocks not found either
                print("Could not find gr.Blocks() to insert handler. Handler not added/updated.")


    if content != updated_content:
        with open(app_py_path, "w") as f:
            f.write(updated_content)
        print("app.py updated for Tab 4 handler - Part 1.")
    else:
        print("No textual changes made to app.py for Tab 4 Handler Part 1 (check script logic if this is unexpected).")

    # Verification
    with open(app_py_path, "r") as f:
        final_content = f.read()
    if "status_messages.append(\"\\nPhotometry Part 1 (Calibration) finished." in final_content and \
       f"def {actual_handler_name}(" in final_content:
        print("Verification successful: Tab 4 handler Part 1 logic seems to be in place.")
    else:
        print("Verification Failed: Expected logic for Tab 4 handler Part 1 not found in the final content.")
        if f"def {actual_handler_name}(" not in final_content:
            print(f"Reason: Function 'def {actual_handler_name}(' not found.")
        if "status_messages.append(\"\\nPhotometry Part 1 (Calibration) finished." not in final_content:
            print("Reason: Specific status message for part 1 completion not found.")


except FileNotFoundError:
    print(f"Error: {app_py_path} not found.")
except Exception as e:
    print(f"An error occurred: {e}")
