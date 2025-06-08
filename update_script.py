# Python script for the subtask - Tab 2 UI Replacement
import re
import os

app_py_path = "app.py"

# The UI code block for Tab 2, including the button click definition
# Note: The state variables (master_bias_path_state, etc.) are defined in Tab 1's scope
# but should be accessible here as they are part of the same gr.Blocks() instance.
tab2_ui_replacement_code = """gr.Markdown("## Calibrate Raw LIGHT Frames")

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
"""

try:
    if not os.path.exists(app_py_path):
        print(f"Error: {app_py_path} not found. Cannot update Tab 2 UI.")
    else:
        with open(app_py_path, "r") as f:
            content = f.read()

        placeholder_line = 'gr.Markdown("Placeholder for Tab 2: LIGHT Frame Correction")'
        # Regex to find the placeholder line and capture its indentation
        placeholder_regex = r"^(\s*)" + re.escape(placeholder_line) + r"\s*$"

        match = re.search(placeholder_regex, content, re.MULTILINE)

        if match:
            indentation = match.group(1)

            # Prepare the new UI code with correct indentation
            indented_tab2_ui_lines = []
            for ui_line in tab2_ui_replacement_code.splitlines():
                indented_tab2_ui_lines.append(indentation + ui_line)
            indented_tab2_ui_code = "\n".join(indented_tab2_ui_lines)

            # Replace the placeholder line with the new indented UI block
            updated_content = content.replace(match.group(0), indented_tab2_ui_code, 1)

            with open(app_py_path, "w") as f:
                f.write(updated_content)
            print(f"Found placeholder and replaced it with Tab 2 UI. Indentation used: '{indentation}'")

            # Verification step
            with open(app_py_path, "r") as f:
                final_content = f.read()
            if "calibrate_lights_button.click(" in final_content and "light_frame_uploads = gr.Files(" in final_content:
                print("Verification successful: Tab 2 UI and click handler seem to be in place.")
            else:
                print("Verification Warning: Placeholder was replaced, but expected UI elements for Tab 2 not fully verified in final content.")
        else:
            print(f"Placeholder for Tab 2 ('{placeholder_line}') not found. No changes made to UI.")
            # Verification for already updated
            if "calibrate_lights_button.click(" in content and "light_frame_uploads = gr.Files(" in content:
                 print("Verification successful (already updated): Tab 2 UI and click handler found.")
            else:
                 print("Verification failed: Tab 2 UI elements not found, and placeholder was also not found.")


except FileNotFoundError: # Should be caught by os.path.exists now
    print(f"Error: {app_py_path} not found.")
except Exception as e:
    print(f"An error occurred: {e}")
