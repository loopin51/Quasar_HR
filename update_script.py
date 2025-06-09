# MEGA SUBTASK SCRIPT to integrate Tab 4 (all parts) and Tab 5 (UI & Handler)
import re
import os
import shutil # For copying files

app_py_path = "app.py"

# --- Imports to ensure at the top of app.py ---
imports_block_for_ensure = """import gradio as gr
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
from tab5_functions import plot_hr_diagram"""

# --- Full Handler Code for Tab 4 (handle_tab4_photometry) ---
tab4_full_handler_code = """
# Handler for Tab 4: Detailed Photometry (Combined Parts 1, 2, 3)
def handle_tab4_photometry(
    tab4_b_file_obj, tab4_v_file_obj,
    tab4_std_star_file_obj, tab4_std_b_mag_str, tab4_std_v_mag_str,
    tab4_roi_str, tab4_k_value_str,
    master_bias_path_state, master_dark_paths_state, master_flat_paths_state,
    request: gr.Request
):
    status_messages = ["Starting Tab 4 Photometry Analysis..."]
    final_results_df_for_ui = None
    preview_image_path_for_ui = None
    csv_file_path_for_ui = None
    display_columns = ['ID', 'X_B', 'Y_B', 'RA_deg', 'Dec_deg', 'InstrMag_B', 'InstrMag_V', 'StdMag_B', 'StdMag_V', 'B-V'] # Define here for except block

    # Ensure master_dark_paths_state and master_flat_paths_state are dictionaries
    if master_dark_paths_state is None: master_dark_paths_state = {}
    if master_flat_paths_state is None: master_flat_paths_state = {}


    try:
        # === PART 1: Load & Correct Files, Standard Star Photometry, Detect Sources ===
        if not tab4_b_file_obj or not tab4_v_file_obj:
            status_messages.append("Error: Both B-filter and V-filter LIGHT frames must be uploaded.")
            raise ValueError("Missing B or V frame.")

        raw_b_path = tab4_b_file_obj.name
        raw_v_path = tab4_v_file_obj.name
        status_messages.append(f"B-frame: {os.path.basename(raw_b_path)}, V-frame: {os.path.basename(raw_v_path)}")

        b_data_raw = load_fits_data(raw_b_path)
        b_header = get_fits_header(raw_b_path)
        v_data_raw = load_fits_data(raw_v_path)
        v_header = get_fits_header(raw_v_path)

        if b_data_raw is None or b_header is None: raise ValueError(f"Could not load B-frame/header from {raw_b_path}")
        if v_data_raw is None or v_header is None: raise ValueError(f"Could not load V-frame/header from {raw_v_path}")
        status_messages.append("B and V LIGHT frames loaded.")

        if not master_bias_path_state or not os.path.exists(master_bias_path_state):
            raise ValueError("Master BIAS not found or path invalid. Prepare/upload in Tab 1.")
        master_bias_data = load_fits_data(master_bias_path_state)
        if master_bias_data is None: raise ValueError(f"Failed to load Master BIAS from {master_bias_path_state}.")
        status_messages.append(f"Master BIAS loaded.")

        temp_dir = os.path.join("masters_output", "temp_final_flats_tab4")
        calibrated_dir = os.path.join("calibrated_lights_output", "tab4_corrected")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(calibrated_dir, exist_ok=True)

        def _correct_science_frame(frame_label, raw_path, frame_header, frame_data_raw_unused_arg): # Added unused arg to match call
            status_messages.append(f"Processing {frame_label}-frame correction...")
            exptime_val = frame_header.get('EXPTIME', frame_header.get('EXPOSURE'))
            filter_name = frame_header.get('FILTER', frame_header.get('FILTER1', frame_header.get('FILTNAME')))
            if exptime_val is None: raise ValueError(f"{frame_label}-frame missing EXPTIME/EXPOSURE.")
            if filter_name is None: raise ValueError(f"{frame_label}-frame missing FILTER name.")

            exptime_float = float(exptime_val)
            filter_key = str(filter_name).strip().replace(" ", "_")

            dark_exp_key = str(exptime_float).replace('.', 'p') + "s"
            frame_master_dark_path = master_dark_paths_state.get(dark_exp_key)
            frame_master_dark_data = None
            if frame_master_dark_path and os.path.exists(frame_master_dark_path):
                frame_master_dark_data = load_fits_data(frame_master_dark_path)
                if frame_master_dark_data is None: status_messages.append(f"Warning: Failed to load {frame_label}-DARK from {frame_master_dark_path}.")
                else: status_messages.append(f"Using {frame_label}-DARK {frame_master_dark_path} for {frame_label}-flat processing.")
            else: status_messages.append(f"Warning: {frame_label}-DARK for exp {exptime_float}s (key {dark_exp_key}) not found in {list(master_dark_paths_state.keys()) if master_dark_paths_state else 'empty state'}. No dark for {frame_label}-flat.")

            prelim_flat_path = master_flat_paths_state.get(filter_key)
            path_to_final_flat = None
            if prelim_flat_path and os.path.exists(prelim_flat_path):
                prelim_flat_data = load_fits_data(prelim_flat_path)
                if prelim_flat_data is not None and master_bias_data is not None:
                    if prelim_flat_data.shape != master_bias_data.shape: raise ValueError(f"{frame_label}-PrelimFlat shape {prelim_flat_data.shape} mismatch MasterBias {master_bias_data.shape}")
                    flat_temp1 = prelim_flat_data - master_bias_data
                    flat_temp2 = flat_temp1
                    if frame_master_dark_data is not None:
                        if flat_temp1.shape != frame_master_dark_data.shape: raise ValueError(f"{frame_label}-Flat(bias-sub) shape {flat_temp1.shape} mismatch {frame_label}-Dark {frame_master_dark_data.shape}")
                        flat_temp2 = flat_temp1 - frame_master_dark_data

                    median_flat_temp2 = np.median(flat_temp2)
                    if median_flat_temp2 == 0: raise ValueError(f"Median of processed {frame_label}-FLAT is zero.")
                    final_flat_data = flat_temp2 / median_flat_temp2
                    path_to_final_flat = os.path.join(temp_dir, f"final_{frame_label}_flat_{filter_key}.fits")
                    temp_flat_header = get_fits_header(prelim_flat_path) if get_fits_header(prelim_flat_path) else fits.Header()
                    temp_flat_header['HISTORY'] = f'Generated final {frame_label}-flat for Tab 4'
                    save_fits_data(path_to_final_flat, final_flat_data, header=temp_flat_header)
                    status_messages.append(f"Generated temporary final {frame_label}-FLAT: {path_to_final_flat}")
                else: status_messages.append(f"Warning: Failed to load Prelim {frame_label}-FLAT from {prelim_flat_path} or Master BIAS missing. {frame_label}-frame not flat-fielded.")
            else: status_messages.append(f"Warning: Prelim {frame_label}-FLAT for filter '{filter_key}' not found in {list(master_flat_paths_state.keys()) if master_flat_paths_state else 'empty state'}. {frame_label}-frame not flat-fielded.")

            base_name = os.path.splitext(os.path.basename(raw_path))[0]
            corrected_frame_path = os.path.join(calibrated_dir, f"{base_name}_cal_{frame_label}.fits")
            actual_master_dark_for_light = frame_master_dark_path if (frame_master_dark_path and os.path.exists(frame_master_dark_path)) else None
            if not correct_light_frame(raw_path, corrected_frame_path, master_bias_path_state, actual_master_dark_for_light, path_to_final_flat):
                raise ValueError(f"Failed to correct {frame_label}-frame using correct_light_frame.")
            status_messages.append(f"{frame_label}-frame corrected: {corrected_frame_path}")
            if path_to_final_flat and os.path.exists(path_to_final_flat):
                try: os.remove(path_to_final_flat)
                except Exception as e_rem_final_flat: status_messages.append(f"Warning: Could not remove temp final flat {path_to_final_flat}: {e_rem_final_flat}")
            return corrected_frame_path

        corrected_b_path = _correct_science_frame("B", raw_b_path, b_header, b_data_raw)
        corrected_v_path = _correct_science_frame("V", raw_v_path, v_header, v_data_raw)

        corrected_b_data = load_fits_data(corrected_b_path) if corrected_b_path else None
        corrected_v_data = load_fits_data(corrected_v_path) if corrected_v_path else None
        if corrected_b_data is None or corrected_v_data is None:
            raise ValueError("Failed to load one or both corrected science frames after calibration attempt.")

        m0_eff_B_local = 25.0
        m0_eff_V_local = 25.0
        status_messages.append(f"Initial default m0_eff_B={m0_eff_B_local:.3f}, m0_eff_V={m0_eff_V_local:.3f}.")

        if tab4_std_star_file_obj and tab4_std_b_mag_str.strip() and tab4_std_v_mag_str.strip():
            status_messages.append("Attempting Standard Star Processing...")
            try:
                std_star_path = tab4_std_star_file_obj.name
                std_b_known = float(tab4_std_b_mag_str.strip())
                std_v_known = float(tab4_std_v_mag_str.strip())
                std_star_header_raw = get_fits_header(std_star_path)
                if std_star_header_raw is None: raise ValueError("Cannot read std star FITS header.")

                std_filter_name = std_star_header_raw.get('FILTER', std_star_header_raw.get('FILTER1', std_star_header_raw.get('FILTNAME')))
                if std_filter_name is None: raise ValueError("Std Star FITS header missing FILTER info.")
                std_filter_name_key = str(std_filter_name).strip().upper() # For B or V check

                corrected_std_star_path = _correct_science_frame("STD", std_star_path, std_star_header_raw, load_fits_data(std_star_path))
                if not corrected_std_star_path: raise ValueError("Standard star frame correction failed.")

                corrected_std_data = load_fits_data(corrected_std_star_path)
                if corrected_std_data is None: raise ValueError("Failed to load corrected standard star data.")

                mean_std, median_std, std_dev_std = sigma_clipped_stats(corrected_std_data, sigma=3.0)
                fwhm_std_guess = 5.0
                daofind_std = DAOStarFinder(fwhm=fwhm_std_guess, threshold=5.*std_dev_std)
                sources_std_table = daofind_std(corrected_std_data - median_std)
                if not sources_std_table or len(sources_std_table) == 0: raise ValueError("No sources detected in standard star image.")

                sources_std_table.sort('flux', reverse=True) # Brightest first
                std_star_x, std_star_y = sources_std_table[0]['xcentroid'], sources_std_table[0]['ycentroid']

                aperture_rad_std = fwhm_std_guess * 1.5
                std_phot_results = perform_photometry(corrected_std_data, [(std_star_x, std_star_y)], aperture_rad_std, sky_annulus_inner_radius=aperture_rad_std + 3, sky_annulus_outer_radius=aperture_rad_std + 8)

                if not std_phot_results or 'instrumental_mag' not in std_phot_results[0] or std_phot_results[0]['instrumental_mag'] is None:
                    raise ValueError("Photometry failed for standard star or instrumental_mag is None.")

                instr_mag_std = std_phot_results[0]['instrumental_mag']
                k_val_float = float(tab4_k_value_str.strip()) if tab4_k_value_str.strip() else 0.15
                std_airmass_float = float(std_star_header_raw.get('AIRMASS', 1.0)) # Default airmass 1.0 if not in header

                if std_filter_name_key.startswith('B'):
                    m0_eff_B_local = std_b_known - instr_mag_std + (k_val_float * std_airmass_float)
                    status_messages.append(f"Calibrated m0_eff_B = {m0_eff_B_local:.3f} using standard star.")
                elif std_filter_name_key.startswith('V'):
                    m0_eff_V_local = std_v_known - instr_mag_std + (k_val_float * std_airmass_float)
                    status_messages.append(f"Calibrated m0_eff_V = {m0_eff_V_local:.3f} using standard star.")
                else:
                    status_messages.append(f"Warning: Standard star filter '{std_filter_name_key}' not B or V. Cannot calibrate m0_eff for this filter.")

                if os.path.exists(corrected_std_star_path): os.remove(corrected_std_star_path)
            except Exception as e_std_proc:
                status_messages.append(f"Standard star processing error: {e_std_proc}. Using default m0_eff values.")

        sources_b_table = None
        if corrected_b_data is not None:
            mean_b, median_b, std_b = sigma_clipped_stats(corrected_b_data, sigma=3.0)
            fwhm_b_guess = 4.0
            daofind_b = DAOStarFinder(fwhm=fwhm_b_guess, threshold=5.*std_b)
            sources_b_table = daofind_b(corrected_b_data - median_b)
            if sources_b_table: status_messages.append(f"Detected {len(sources_b_table)} sources in B-frame.")
            else: status_messages.append("No sources detected in B-frame.")

        sources_v_table = None
        if corrected_v_data is not None:
            mean_v, median_v, std_v = sigma_clipped_stats(corrected_v_data, sigma=3.0)
            fwhm_v_guess = 4.0
            daofind_v = DAOStarFinder(fwhm=fwhm_v_guess, threshold=5.*std_v)
            sources_v_table = daofind_v(corrected_v_data - median_v)
            if sources_v_table: status_messages.append(f"Detected {len(sources_v_table)} sources in V-frame.")
            else: status_messages.append("No sources detected in V-frame.")

        # === PART 2: ROI Filter, Instrumental Photometry, WCS, Standard Mags ===
        processed_sources_b_phot = []
        processed_sources_v_phot = []

        roi_x, roi_y, roi_radius = None, None, None
        if tab4_roi_str and tab4_roi_str.strip():
            try:
                parts = [float(p.strip()) for p in tab4_roi_str.split(',')]
                if len(parts) == 3:
                    roi_x, roi_y, roi_radius = parts[0], parts[1], parts[2]
                    if sources_b_table:
                        dist_b = np.sqrt((sources_b_table['xcentroid'] - roi_x)**2 + (sources_b_table['ycentroid'] - roi_y)**2)
                        sources_b_table = sources_b_table[dist_b <= roi_radius]
                    if sources_v_table:
                        dist_v = np.sqrt((sources_v_table['xcentroid'] - roi_x)**2 + (sources_v_table['ycentroid'] - roi_y)**2)
                        sources_v_table = sources_v_table[dist_v <= roi_radius]
                    status_messages.append(f"Applied ROI: {len(sources_b_table) if sources_b_table else 0} B sources, {len(sources_v_table) if sources_v_table else 0} V sources remaining.")
                else: raise ValueError("ROI string needs 3 comma-separated numbers: x,y,radius")
            except Exception as e_roi: status_messages.append(f"Warning: ROI format error or application failed: {e_roi}. Processing all sources.")

        phot_ap_rad = 5.0; phot_sky_in = 8.0; phot_sky_out = 12.0
        if sources_b_table and len(sources_b_table) > 0:
            coords_b = np.array([(s['xcentroid'], s['ycentroid']) for s in sources_b_table])
            processed_sources_b_phot = perform_photometry(corrected_b_data, coords_b, phot_ap_rad, phot_sky_in, phot_sky_out)
        if sources_v_table and len(sources_v_table) > 0:
            coords_v = np.array([(s['xcentroid'], s['ycentroid']) for s in sources_v_table])
            processed_sources_v_phot = perform_photometry(corrected_v_data, coords_v, phot_ap_rad, phot_sky_in, phot_sky_out)

        k_val_float = float(tab4_k_value_str.strip()) if tab4_k_value_str.strip() else 0.15
        airmass_b_float = float(b_header.get('AIRMASS', 1.0))
        airmass_v_float = float(v_header.get('AIRMASS', 1.0))

        for r_dict in processed_sources_b_phot:
            if r_dict.get('instrumental_mag') is not None:
                r_dict['StdMag_B'] = r_dict['instrumental_mag'] + m0_eff_B_local - (k_val_float * airmass_b_float)
            if b_header:
                try:
                    wcs_b = WCS(b_header);
                    if wcs_b.is_celestial: ra_dec = wcs_b.pixel_to_world(r_dict['x'], r_dict['y']); r_dict['ra_deg']=ra_dec.ra.deg; r_dict['dec_deg']=ra_dec.dec.deg
                except: pass
        for r_dict in processed_sources_v_phot:
            if r_dict.get('instrumental_mag') is not None:
                r_dict['StdMag_V'] = r_dict['instrumental_mag'] + m0_eff_V_local - (k_val_float * airmass_v_float)
            if v_header:
                try:
                    wcs_v = WCS(v_header);
                    if wcs_v.is_celestial: ra_dec = wcs_v.pixel_to_world(r_dict['x'], r_dict['y']); r_dict['ra_deg']=ra_dec.ra.deg; r_dict['dec_deg']=ra_dec.dec.deg
                except: pass
        status_messages.append("Instrumental & Standard Magnitudes Calculated.")

        # === PART 3: Cross-match, B-V, Table, Preview, CSV ===
        final_results_list = []
        matched_stars_for_preview = []
        match_radius_pixels = 3.0

        use_sky_matching = all(p.get('ra_deg') is not None for p in processed_sources_b_phot) and \
                           all(p.get('ra_deg') is not None for p in processed_sources_v_phot)

        if use_sky_matching:
            status_messages.append("Attempting Sky Coordinate (RA/Dec) cross-match.")
            coords_b_sky = SkyCoord([p['ra_deg'] for p in processed_sources_b_phot]*u.deg, [p['dec_deg'] for p in processed_sources_b_phot]*u.deg)
            coords_v_sky = SkyCoord([p['ra_deg'] for p in processed_sources_v_phot]*u.deg, [p['dec_deg'] for p in processed_sources_v_phot]*u.deg)
            idx_v, d2d_v, _ = coords_b_sky.match_to_catalog_sky(coords_v_sky)

            plate_scale_arcsec_per_px_b = abs(b_header.get('CDELT1', b_header.get('SECPIX1', 0.5)) * 3600) # Use abs for scale
            match_tolerance_arcsec = match_radius_pixels * plate_scale_arcsec_per_px_b * u.arcsec

            processed_v_phot_matched_indices = set() # To avoid matching a V source multiple times

            for i_b, src_b_dict in enumerate(processed_sources_b_phot):
                entry = {'ID': i_b + 1, 'X_B': src_b_dict.get('x'), 'Y_B': src_b_dict.get('y'),
                         'RA_deg': src_b_dict.get('ra_deg'), 'Dec_deg': src_b_dict.get('dec_deg'),
                         'InstrMag_B': src_b_dict.get('instrumental_mag'), 'StdMag_B': src_b_dict.get('StdMag_B'),
                         'InstrMag_V': None, 'StdMag_V': None, 'B-V': None, 'X_V': None, 'Y_V': None}
                if d2d_v[i_b] <= match_tolerance_arcsec:
                    matched_v_idx = idx_v[i_b]
                    if matched_v_idx not in processed_v_phot_matched_indices:
                        src_v_dict = processed_sources_v_phot[matched_v_idx]
                        entry.update({'InstrMag_V': src_v_dict.get('instrumental_mag'), 'StdMag_V': src_v_dict.get('StdMag_V'),
                                      'X_V': src_v_dict.get('x'), 'Y_V': src_v_dict.get('y')})
                        if entry['StdMag_B'] is not None and entry['StdMag_V'] is not None:
                            entry['B-V'] = entry['StdMag_B'] - entry['StdMag_V']
                        matched_stars_for_preview.append({'x': src_b_dict['x'], 'y': src_b_dict['y'], 'id': entry['ID']})
                        processed_v_phot_matched_indices.add(matched_v_idx)
                final_results_list.append(entry)
        else:
            status_messages.append("Attempting Pixel Coordinate (X/Y) cross-match (WCS might be missing/inconsistent).")
            available_v_sources_dicts_px = [dict(s) for s in processed_sources_v_phot]
            for i_b, src_b_dict in enumerate(processed_sources_b_phot):
                src_b_dict['id'] = src_b_dict.get('id', i_b + 1)
                best_match_v_dict_px = None; min_dist_px = float('inf'); idx_to_pop_v_px = -1
                for i_v_px, src_v_dict_avail_px in enumerate(available_v_sources_dicts_px):
                    dist_px = np.sqrt((src_b_dict['x'] - src_v_dict_avail_px['x'])**2 + (src_b_dict['y'] - src_v_dict_avail_px['y'])**2)
                    if dist_px < min_dist_px and dist_px <= match_radius_pixels:
                        min_dist_px = dist_px; best_match_v_dict_px = src_v_dict_avail_px; idx_to_pop_v_px = i_v_px
                entry = {'ID': src_b_dict['id'], 'X_B': src_b_dict.get('x'), 'Y_B': src_b_dict.get('y'),
                         'RA_deg': src_b_dict.get('ra_deg'), 'Dec_deg': src_b_dict.get('dec_deg'),
                         'InstrMag_B': src_b_dict.get('instrumental_mag'), 'StdMag_B': src_b_dict.get('StdMag_B'),
                         'InstrMag_V': None, 'StdMag_V': None, 'B-V': None, 'X_V': None, 'Y_V': None}
                if best_match_v_dict_px:
                    entry.update({'InstrMag_V': best_match_v_dict_px.get('instrumental_mag'), 'StdMag_V': best_match_v_dict_px.get('StdMag_V'),
                                  'X_V': best_match_v_dict_px.get('x'), 'Y_V': best_match_v_dict_px.get('y')})
                    if entry['StdMag_B'] is not None and entry['StdMag_V'] is not None:
                        entry['B-V'] = entry['StdMag_B'] - entry['StdMag_V']
                    matched_stars_for_preview.append({'x': src_b_dict['x'], 'y': src_b_dict['y'], 'id': entry['ID']})
                    if idx_to_pop_v_px != -1: available_v_sources_dicts_px.pop(idx_to_pop_v_px)
                final_results_list.append(entry)

        final_results_list.sort(key=lambda s: s.get('StdMag_B') if s.get('StdMag_B') is not None else (s.get('InstrMag_B') if s.get('InstrMag_B') is not None else float('inf')))
        status_messages.append(f"Cross-matched B & V sources. Found {len(matched_stars_for_preview)} matches.")


        df_data_for_ui = [[round(row.get(col),3) if isinstance(row.get(col), float) else row.get(col) for col in display_columns] for row in final_results_list]
        final_results_df_for_ui = pd.DataFrame(df_data_for_ui, columns=display_columns)
        status_messages.append("Results table prepared.")

        if corrected_b_data is not None:
            plt.figure(figsize=(8, 8))
            norm = ImageNormalize(corrected_b_data, interval=ZScaleInterval())
            plt.imshow(corrected_b_data, cmap='gray', origin='lower', norm=norm); plt.colorbar(label="Pixel Value")
            plt.title(f"B-filter Preview: {os.path.basename(raw_b_path)}"); plt.xlabel("X pixel"); plt.ylabel("Y pixel")
            if roi_x is not None and roi_y is not None and roi_radius is not None:
                plt.gca().add_patch(plt.Circle((roi_x, roi_y), roi_radius, color='blue', fill=False, ls='--', label='ROI'))

            temp_labels_added = set()
            for star_info in matched_stars_for_preview:
                label = 'Matched Source' if 'Matched Source' not in temp_labels_added else None
                plt.gca().add_patch(plt.Circle((star_info['x'], star_info['y']), 10, color='lime', fill=False, alpha=0.7, label=label))
                if label: temp_labels_added.add('Matched Source')
                plt.text(star_info['x'] + 12, star_info['y'] + 12, str(star_info['id']), color='lime', fontsize=9)

            if plt.gca().get_legend_handles_labels()[1]: # Check if there are any labels
                 plt.legend()

            preview_img_dir = os.path.join(calibrated_dir, "previews")
            os.makedirs(preview_img_dir, exist_ok=True)
            preview_image_path_for_ui = os.path.join(preview_img_dir, "tab4_b_preview.png")
            plt.savefig(preview_image_path_for_ui); plt.close()
            status_messages.append(f"Preview image saved: {preview_image_path_for_ui}")

        if final_results_list:
            object_name_from_header = b_header.get('OBJECT', 'UnknownObject').strip().replace(' ', '_')
            csv_filename = f"{object_name_from_header}_phot_results.csv"
            csv_path_temp = os.path.join(calibrated_dir, csv_filename)
            final_results_df_for_ui.to_csv(csv_path_temp, index=False, float_format='%.3f')
            csv_file_path_for_ui = csv_path_temp
            status_messages.append(f"Results CSV ready for download: {csv_file_path_for_ui}")

        status_messages.append("Tab 4 Analysis Completed.")

    except Exception as e_tab4_main:
        status_messages.append(f"CRITICAL ERROR in Tab 4: {str(e_tab4_main)}")
        final_results_df_for_ui = pd.DataFrame(columns=display_columns)
        preview_image_path_for_ui = None
        csv_file_path_for_ui = None

    return "\\n".join(status_messages), final_results_df_for_ui, preview_image_path_for_ui, csv_file_path_for_ui
"""

# --- Tab 5 UI Code (ensure click handler is part of it) ---
tab5_ui_code = """
gr.Markdown("## Tab 5: H-R Diagram (Color-Magnitude Diagram)")
gr.Markdown("Upload the CSV file generated by Tab 4 (which should contain 'StdMag_V' and 'B-V' columns). Optionally, provide the object name if not in CSV or for a custom title.")

with gr.Row():
    with gr.Column(scale=1):
        tab5_csv_upload = gr.File(label="Upload Photometry CSV File", file_types=['.csv'], type="filepath", elem_id="tab5_csv_upload")
        tab5_object_name_input = gr.Textbox(label="Object Name (for Diagram Title)", placeholder="e.g., M45, NGC1234, or from CSV if column exists", elem_id="tab5_obj_name")
        tab5_plot_hr_button = gr.Button("Generate H-R Diagram", variant="primary", elem_id="tab5_plot_btn")

    with gr.Column(scale=2):
        tab5_hr_diagram_display = gr.Image(label="H-R Diagram", type="filepath", interactive=False, height=500, elem_id="tab5_hr_img", visible=False) # Initially hidden
        tab5_status_display = gr.Textbox(label="Status / Errors", lines=3, interactive=False, elem_id="tab5_hr_status")

# Connect button to handler for Tab 5
tab5_plot_hr_button.click(
    fn=handle_tab5_hr_diagram,
    inputs=[tab5_csv_upload, tab5_object_name_input],
    outputs=[tab5_status_display, tab5_hr_diagram_display],
    request=True # Added request for handle_tab5_hr_diagram
)
"""

# --- Tab 5 Handler Code ---
tab5_handler_code = """
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
        if 'OBJECT_NAME' in df.columns and pd.notna(df['OBJECT_NAME'].iloc[0]) and str(df['OBJECT_NAME'].iloc[0]).strip():
             obj_name_from_csv = str(df['OBJECT_NAME'].iloc[0]).strip()

        final_object_name = object_name_str.strip() if object_name_str and object_name_str.strip() else obj_name_from_csv
        if final_object_name:
            plot_title = f"{final_object_name} H-R Diagram"

        preview_dir = os.path.join("calibrated_lights_output", "previews")
        os.makedirs(preview_dir, exist_ok=True)

        safe_title_part = "".join(c if c.isalnum() or c in ['_','-'] else '' for c in final_object_name if final_object_name else "HR_Diagram")
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
"""

# --- Main script logic to modify app.py ---
try:
    with open(app_py_path, "r") as f:
        content = f.read()
    original_content = content

    # 1. Ensure all imports are present
    existing_imports_match = re.match(r"(^(?:import .*\n|from .* import .*\n|#.*\n|\s*\n)*)", content, re.MULTILINE)
    existing_imports_block_str = existing_imports_match.group(1) if existing_imports_match else ""
    non_import_content = content[len(existing_imports_block_str):]

    current_imports_set = set(line.strip() for line in existing_imports_block_str.splitlines() if line.strip() and not line.strip().startswith("#"))
    all_needed_imports_set = set(line.strip() for line in imports_block_for_ensure.splitlines() if line.strip() and not line.strip().startswith("#"))

    new_imports_to_add = sorted(list(all_needed_imports_set - current_imports_set))

    if new_imports_to_add:
        # Add new imports to the existing block or at the start of the file
        updated_imports_section = existing_imports_block_str.strip() + "\n" + "\n".join(new_imports_to_add) + "\n"
        content = updated_imports_section.strip() + "\n\n" + non_import_content
        print(f"Added missing imports:\n{new_imports_to_add}")
    else:
        print("All necessary imports seem to be present.")


    # 2. Tab 4: Replace dummy/partial handler with full logic
    dummy_handler_name_t4 = "handle_run_photometry_analysis_dummy"
    actual_handler_name_t4 = "handle_tab4_photometry"

    content = content.replace(f"fn={dummy_handler_name_t4}", f"fn={actual_handler_name_t4}") # Ensure button calls correct name
    print(f"Ensured Tab 4 button click uses: fn={actual_handler_name_t4}")

    # Define a more robust regex for function replacement that handles docstrings and varying newlines
    def replace_function_body(current_content, func_name_to_replace, new_func_code_str):
        # This regex tries to find 'def func_name(signature):' then captures everything up to the next 'def ' or 'with gr.Blocks()' or end of file.
        # It's complex to make this perfectly robust for all code structures via regex.
        # Simpler: find start of function, then find where it ends (next function or major block).
        # This assumes functions are defined at a certain indentation level (e.g. global).
        pattern_str = r"(^\s*def " + re.escape(func_name_to_replace) + r"\(.*?\):\s*\n(?:(?:^\s+.*\n|^\s*#.*?\n|^\s*\"{3}.*?\"{3}\s*\n|^\s*'{3}.*?'{3}\s*\n|^\s*\n)+)?)"
        match_func_start = re.search(pattern_str, current_content, re.MULTILINE | re.DOTALL)

        if match_func_start:
            start_of_func_def = match_func_start.group(1) # The 'def func_name(...):' and any docstring
            end_of_header_pos = match_func_start.end()

            # Find where the function body ends. Look for next 'def ', 'class ', or 'with gr.Blocks()' at same or lesser indent, or EOF
            # This is still tricky. A simpler but potentially overreaching approach for this script:
            # Assume dummy functions are relatively simple and followed by another function or major block.

            # Let's use a simpler regex that matches the dummy structure more directly if it's known
            # Or, for replacement, just find the start and replace up to a known end pattern of the dummy.
            # Given the script already has specific regexes for dummy handlers, let's stick to that for replacement.
            # The func_def_pattern_t4_dummy/actual from previous steps were designed for this.

            # Using a simplified approach: replace based on name, assuming it's a global function.
            # The full new code includes "def handle_tab4_photometry(...):" so it replaces the old one entirely.
            # If the old function doesn't exist, the new one will be inserted.

            # Try to remove old definition first, then add new one.
            # Pattern to match the whole function block (simplified)
            old_func_block_pattern = re.compile(r"^\s*def " + re.escape(func_name_to_replace) + r"\(.*?\):\s*(\n\s+.*)*(\n|$)", re.MULTILINE)
            current_content = old_func_block_pattern.sub("", current_content) # Remove if exists

            # Now add the new function code. Try to place it before 'with gr.Blocks()'.
            new_code_block_to_insert = new_func_code_str.strip() + "\n\n"
            match_blocks_insert = re.search(r"^\s*with gr\.Blocks\(\) as astro_app:", current_content, re.MULTILINE)
            if match_blocks_insert:
                current_content = current_content[:match_blocks_insert.start()] + new_code_block_to_insert + current_content[match_blocks_insert.start():]
            else: # Fallback: append to end (less ideal)
                current_content += "\n" + new_code_block_to_insert
            print(f"Replaced/Inserted handler logic for '{func_name_to_replace}' with new full logic.")
            return current_content
        else:
            print(f"Could not find function start for '{func_name_to_replace}' to replace/insert robustly.")
            return current_content # Return unchanged

    # Replace Tab 4 handler
    content = replace_function_body(content, dummy_handler_name_t4, tab4_full_handler_code)
    content = replace_function_body(content, actual_handler_name_t4, tab4_full_handler_code) # In case it was already renamed


    # 3. Tab 5: Replace placeholder UI (including its click handler now)
    placeholder_t5_regex = r"^(\s*)gr\.Markdown\(\"Placeholder for Tab 5: H-R Diagram Plotting\"\)\s*$"
    match_t5_placeholder = re.search(placeholder_t5_regex, content, re.MULTILINE)
    if match_t5_placeholder:
        indentation_t5 = match_t5_placeholder.group(1)
        indented_tab5_ui_lines = [(indentation_t5 + ui_line).rstrip() for ui_line in tab5_ui_code.splitlines()]
        indented_tab5_ui_code = "\n".join(indented_tab5_ui_lines)
        content = content.replace(match_t5_placeholder.group(0), indented_tab5_ui_code)
        print("Replaced Tab 5 UI placeholder with new UI including click handler.")
    else:
        if "tab5_plot_hr_button.click(" not in content:
            print("Tab 5 UI placeholder not found, and click handler not present. Tab 5 UI might be missing or script needs adjustment.")
        else:
            print("Tab 5 UI placeholder not found, but click handler present. Assuming Tab 5 UI already updated.")


    # 4. Tab 5: Add handler function definition (if not already there due to replace_function_body being too greedy)
    if "def handle_tab5_hr_diagram(" not in content:
        content = replace_function_body(content, "handle_tab5_hr_diagram", tab5_handler_code) # Use same robust replacement
        print("Added/Replaced Tab 5 handler function 'handle_tab5_hr_diagram'.")
    else:
        # If it exists, ensure it's the correct one (by replacing it)
        content = replace_function_body(content, "handle_tab5_hr_diagram", tab5_handler_code)
        print("Tab 5 handler 'handle_tab5_hr_diagram' already exists, content updated to ensure latest version.")


    # 5. Write changes if any
    if original_content != content:
        with open(app_py_path, "w") as f:
            f.write(content)
        print("app.py MODIFIED with Tab 4 (full) & Tab 5 (UI & handler) logic.")
    else:
        print("app.py was NOT modified by the script (no textual changes detected vs original).")


    # Verification (basic)
    with open(app_py_path, "r") as f:
        final_content = f.read()
    tab4_ok = "def handle_tab4_photometry(" in final_content and \
              "status_messages.append(\"\\nPhotometry Part 1 (Calibration) finished." in final_content and \
              "status_messages.append(\"Results CSV ready for download:" in final_content
    tab5_ok = "def handle_tab5_hr_diagram(" in final_content and \
              "tab5_plot_hr_button.click(" in final_content and \
              "gr.Image(label=\\\"H-R Diagram\\\"" in final_content # Escaped quote for literal match

    if tab4_ok and tab5_ok:
        print("High-level verification PASSED: Tab 4 full handler & Tab 5 UI/handler seem present.")
    else:
        print("High-level verification FAILED for Tab 4/5 integration:")
        if not tab4_ok: print("  - Tab 4 elements missing or incomplete.")
        if not tab5_ok: print("  - Tab 5 elements missing or incomplete.")

except FileNotFoundError:
    print(f"Error: {app_py_path} not found.")
except Exception as e_mega:
    print(f"MEGA SUBTASK SCRIPT ERROR: {e_mega}")

# Final confirmation message for subtask runner
print("Mega subtask script execution finished.")
