# Main application file for the Gradio Astro App
import gradio as gr
from tab1_functions import * # Placeholder for specific imports
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



with gr.Blocks() as astro_app:
    gr.Markdown("# Astro App")

    with gr.Tabs():
        with gr.TabItem("Master Frame Generation (Tab 1)"):
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
            # TODO: Add UI components for Tab 1

        with gr.TabItem("LIGHT Frame Correction (Tab 2)"):
            gr.Markdown("Placeholder for Tab 2: LIGHT Frame Correction")
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

if __name__ == '__main__':
    # Ensure necessary packages are installed
    try:
        import scipy
    except ImportError:
        print("SCIPY NOT INSTALLED. PLEASE INSTALL IT: pip install scipy")
        # In a real scenario, you might try to install it here or provide clearer instructions.

    astro_app.launch()
