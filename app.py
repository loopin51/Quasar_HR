# Main application file for the Gradio Astro App
import gradio as gr
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

with gr.Blocks() as astro_app:
    gr.Markdown("# Astro App")

    with gr.Tabs():
        with gr.TabItem("Master Frame Generation (Tab 1)"):
            gr.Markdown("Placeholder for Tab 1: Master Frame Generation")
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
