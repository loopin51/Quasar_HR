import re
import os # <-- ADDED IMPORT OS

# --- Helper function to manage app.py content ---
def update_app_py(content: str) -> str:
    # 1. Add import for tab1_functions if not present
    if "from tab1_functions import" not in content and "from tab1_functions import *" not in content :
        content = content.replace(
            "import gradio as gr",
            "import gradio as gr\nfrom tab1_functions import * # Placeholder for specific imports"
        )
        print("Added import for tab1_functions.")

    # 2. Define dummy handler functions globally
    # These will be inserted before 'with gr.Blocks() as astro_app:'
    dummy_handlers_code = '''
# Dummy handlers for Tab 1 UI (will be replaced with actual logic)
def handle_generate_master_bias(files):
    if not files: return "No BIAS files uploaded. Cannot generate Master BIAS."
    # Actual call to tab1_functions.create_master_frame will go here
    # For now, using a generic name from the import *
    # result = create_master_frame(file_paths=[f.name for f in files], output_path="temp_master_bias.fits", method="mean", frame_type="BIAS")
    # return f"Master BIAS generation attempted. Result: {result}"
    return f"Attempting to generate Master BIAS from {len(files)} file(s)... (UI Handler - Not fully implemented yet)"

def handle_generate_master_dark(files):
    if not files: return "No DARK files uploaded. Cannot generate Master DARKs."
    return f"Attempting to generate Master DARKs from {len(files)} file(s)... (UI Handler - Not fully implemented yet)"

def handle_generate_master_flat(files):
    if not files: return "No FLAT files uploaded. Cannot generate Prelim. Master FLATs."
    return f"Attempting to generate Prelim. Master FLATs from {len(files)} file(s)... (UI Handler - Not fully implemented yet)"

def handle_upload_master_bias(file_obj):
    if not file_obj: return "No Master BIAS file provided for upload."
    # Session state or copy file to a known location would happen here.
    return f"Master BIAS '{file_obj.name}' received for upload. (UI Handler - Not fully implemented yet)"

def handle_upload_master_darks(files):
    if not files: return "No Master DARK files provided for upload."
    # Session state or copy files to a known location would happen here.
    return f"Master DARKs received for upload: {[f.name for f in files]} (UI Handler - Not fully implemented yet)"

def handle_upload_master_flats(files):
    if not files: return "No Master FLAT files provided for upload."
    # Session state or copy files to a known location would happen here.
    return f"Master FLATs received for upload: {[f.name for f in files]} (UI Handler - Not fully implemented yet)"

'''
    if "def handle_generate_master_bias(" not in content: # Check if handlers are already there
        # Find the line 'with gr.Blocks() as astro_app:' and insert before it
        match_blocks = re.search(r"^\s*with gr\.Blocks\(\) as astro_app:", content, re.MULTILINE)
        if match_blocks:
            insert_pos = match_blocks.start()
            content = content[:insert_pos] + dummy_handlers_code + "\n" + content[insert_pos:]
            print("Added dummy handler functions for Tab 1 globally.")
        else:
            # Fallback: append before if __name__ == '__main__':
            match_main = re.search(r"^\s*if __name__ == '__main__':", content, re.MULTILINE)
            if match_main:
                insert_pos = match_main.start()
                content = content[:insert_pos] + dummy_handlers_code + "\n" + content[insert_pos:]
                print("Added dummy handler functions for Tab 1 before main block.")
            else: # Or just append
                content += "\n" + dummy_handlers_code
                print("Appended dummy handler functions for Tab 1 (fallback).")


    # 3. Define Tab 1 UI content
    tab1_ui_py_code = '''gr.Markdown("## Create Master Calibration Frames or Upload Existing Ones")

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
                download_master_bias_placeholder = gr.Markdown("Master BIAS download placeholder", elem_id="tab1_dl_mbias_ph") # Placeholder
                download_master_dark_placeholder = gr.Markdown("Master DARKs download placeholder", elem_id="tab1_dl_mdark_ph") # Placeholder
                download_master_flat_placeholder = gr.Markdown("Master FLATs download placeholder", elem_id="tab1_dl_mflat_ph") # Placeholder

            # Connect buttons to handlers
            generate_master_bias_button.click(fn=handle_generate_master_bias, inputs=[bias_uploads], outputs=[tab1_status_display])
            generate_master_dark_button.click(fn=handle_generate_master_dark, inputs=[dark_uploads], outputs=[tab1_status_display])
            generate_master_flat_button.click(fn=handle_generate_master_flat, inputs=[flat_uploads], outputs=[tab1_status_display])

            upload_master_bias.upload(fn=handle_upload_master_bias, inputs=[upload_master_bias], outputs=[tab1_status_display])
            upload_master_darks.upload(fn=handle_upload_master_darks, inputs=[upload_master_darks], outputs=[tab1_status_display])
            upload_master_flats.upload(fn=handle_upload_master_flats, inputs=[upload_master_flats], outputs=[tab1_status_display])
'''

    # 4. Replace placeholder with Tab 1 UI
    placeholder_regex = r'^\s*gr\.Markdown\("Placeholder for Tab 1: Master Frame Generation"\)'

    # Find the placeholder and its indentation
    match_placeholder = re.search(placeholder_regex, content, re.MULTILINE)
    if match_placeholder:
        placeholder_line_full = match_placeholder.group(0)
        indentation = re.match(r"^\s*", placeholder_line_full).group(0)

        lines = tab1_ui_py_code.split('\n')
        # Ensure first line gets the indent, and subsequent lines too if they are not empty
        indented_tab1_ui_code = indentation + lines[0] + "\n"
        indented_tab1_ui_code += "\n".join([indentation + line if line.strip() else line for line in lines[1:]])

        content = content.replace(placeholder_line_full, indented_tab1_ui_code.rstrip()) # rstrip to remove trailing newlines from the block itself
        print("Replaced Tab 1 placeholder with new UI components.")
    else:
        print("Tab 1 placeholder not found. UI not updated.")
        # This case should ideally not happen if app.py is as expected.

    return content

# --- Main script logic for subtask ---
try:
    app_py_path = "app.py" # Assuming app.py is in the current directory for the script
    # First, ensure app.py exists, if not, create a basic shell.
    # This might be needed if previous steps failed to create app.py robustly.
    if not os.path.exists(app_py_path):
        print(f"{app_py_path} not found. Creating a basic shell for it.")
        with open(app_py_path, "w") as f:
            f.write("""import gradio as gr

# Placeholder for tab3_functions import if script adds it before this point
# from tab3_functions import calculate_extinction_coefficient

# Dummy handlers will be inserted here by the script

with gr.Blocks() as astro_app:
    gr.Markdown("# Astro App")

    with gr.Tabs():
        with gr.TabItem("Master Frame Generation (Tab 1)"):
            gr.Markdown("Placeholder for Tab 1: Master Frame Generation")

        with gr.TabItem("LIGHT Frame Correction (Tab 2)"):
            gr.Markdown("Placeholder for Tab 2: LIGHT Frame Correction")

        with gr.TabItem("Extinction Coefficient (Tab 3)"):
            gr.Markdown("Placeholder for Tab 3: Extinction Coefficient") # Basic placeholder

        with gr.TabItem("Detailed Photometry (Tab 4)"):
            gr.Markdown("Placeholder for Tab 4: Detailed Photometry and Catalog Analysis")

        with gr.TabItem("H-R Diagram (Tab 5)"):
            gr.Markdown("Placeholder for Tab 5: H-R Diagram Plotting")

if __name__ == '__main__':
    astro_app.launch()
""")
        print(f"Basic {app_py_path} created.")


    with open(app_py_path, "r") as f:
        original_content = f.read()

    updated_content = update_app_py(original_content)

    if original_content != updated_content:
        with open(app_py_path, "w") as f:
            f.write(updated_content)
        print("app.py successfully updated for Tab 1 UI.")
    else:
        print("No changes made to app.py (either already updated or issues encountered).")

except FileNotFoundError: # Should be handled by the os.path.exists check now
    print(f"Error: {app_py_path} not found and could not be created.")
except Exception as e:
    print(f"An error occurred: {e}")

# Verify by printing the content of app.py
try:
    with open("app.py", "r") as f:
        print("\n--- Content of app.py after update: ---")
        print(f.read())
        print("--- End of app.py content ---")
except FileNotFoundError:
    print(f"Error: Could not read {app_py_path} to verify content.")
