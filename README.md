# Astro App: Astronomical Image Processing Suite

## Overview

Astro App is a comprehensive tool designed for amateur and professional astronomers to perform essential astronomical image processing tasks. It provides a user-friendly interface built with Gradio, allowing users to calibrate images, perform photometry, and generate scientific plots like H-R diagrams.

Key functionalities include:
-   **Master Calibration Frame Generation:** Create or upload master bias, dark, and flat frames.
-   **Light Frame Calibration:** Correct raw astronomical images using master calibration frames.
-   **Atmospheric Extinction Calculation:** Determine the extinction coefficient (k) and magnitude at zero airmass (m0).
-   **Detailed Photometry:** Perform source detection and aperture photometry on calibrated images, with options for standard star calibration.
-   **H-R Diagram Plotting:** Generate Hertzsprung-Russell diagrams from photometry data.

## Setup and Installation

Follow these steps to set up your environment and run Astro App:

1.  **Prerequisites:**
    *   Python 3.10 or newer is recommended.
    *   Git (for cloning the repository).

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd astro-app # Or your repository's directory name
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    Install all necessary Python libraries using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Once the setup is complete, you can run Astro App using:

```bash
python app.py
```

This will typically start a local web server, and you can access the application by opening the provided URL (e.g., `http://127.0.0.1:7860`) in your web browser.

## Functionality by Tab

The application is organized into several tabs, each dedicated to a specific task:

### Tab 1: Master Frame Generation

This tab allows you to create or manage your master calibration frames (Bias, Darks, Flats).
-   **Generate Master Frames:**
    -   **Master BIAS:** Upload multiple raw bias frames. The app will combine them (median combine) to create a Master BIAS frame.
    -   **Master DARKs:** Upload multiple raw dark frames. These are automatically grouped by exposure time. For each group, a Master DARK is generated (median combine) after subtracting the Master BIAS (if available).
    -   **Prelim. Master FLATs:** Upload multiple raw flat frames. These are automatically grouped by filter name. For each group, a preliminary Master FLAT is generated (median combine). These prelim flats are then further processed in Tab 4 for final flat fielding.
-   **Upload Existing Master Frames:**
    -   You can also upload your own pre-processed Master BIAS, Master DARKs (one per exposure time, identified by 'EXPTIME' in header), and preliminary Master FLATs (one per filter, identified by 'FILTER' in header).
-   Generated or uploaded master frames are stored and become available for use in other tabs.

### Tab 2: LIGHT Frame Correction

This tab is for calibrating your raw science (LIGHT) images.
-   **Upload LIGHT Frames:** Upload one or more raw LIGHT frames.
-   **Calibration Process:** The app uses the Master BIAS, appropriate Master DARK (matched by exposure time from Tab 1), and appropriate preliminary Master FLAT (matched by filter from Tab 1, which is then fully calibrated on-the-fly) to correct each LIGHT frame.
-   **Preview & Stretch Options:** A PNG preview of the *first* successfully calibrated LIGHT frame is displayed. You can choose different stretch modes (e.g., ZScale, MinMax, Percentile) for the preview image.
-   **Download:** You can download the first calibrated FITS file. A list of all successfully calibrated files is also provided.

### Tab 3: Extinction Coefficient

Calculate the atmospheric extinction coefficient (k) for your observing site and conditions.
-   **Input Data:** Enter comma-separated airmass values and their corresponding instrumental magnitudes for a standard star observed at different airmasses.
-   **Calculation:** The app performs a linear regression on the provided data (`m = m0 - kX`).
-   **Outputs:**
    -   `k`: The calculated extinction coefficient.
    -   `k_err`: The uncertainty in k.
    -   `m0`: The magnitude at zero airmass (instrumental).
    -   **Plot:** A plot of airmass vs. instrumental magnitude, showing the data points and the fitted regression line.
    -   **Table:** A table displaying the input airmass and magnitude data points.

### Tab 4: Detailed Photometry

Perform detailed photometry on your calibrated B and V filter science images.
-   **Upload Frames:** Upload your calibrated B-filter and V-filter LIGHT frames.
-   **Master Frames:** This tab uses the master calibration frames established in Tab 1 to perform on-the-fly calibration of the B and V frames if they are raw, or can work with pre-calibrated images.
-   **Settings:**
    -   **ROI (Region of Interest):** Optionally define a circular region (center_x, center_y, radius_pixels) to restrict photometry to a specific area.
    -   **FWHM (pixels):** Set the Full Width at Half Maximum for source detection (used by DAOStarFinder).
    -   **Aperture Settings:** Define the aperture radius, sky annulus inner radius, and sky annulus outer radius (in pixels) for photometry.
    -   **Extinction Coefficient (k):** Input the k-value (e.g., from Tab 3) for magnitude calculations.
-   **Optional Standard Star Calibration:**
    -   Upload a FITS file of a standard star (observed in B or V).
    -   Provide its known standard B and V magnitudes.
    -   If provided, this will be used to calculate the zero-point magnitudes (m0_B, m0_V) for calibration. Otherwise, default zero-points are used.
-   **Outputs:**
    -   **Results Table:** A table listing detected sources with their coordinates, instrumental magnitudes, standard magnitudes (if m0 and k are applied), and B-V color.
    -   **Preview Image:** A PNG preview of the B-filter image with detected and matched sources and the ROI marked.
    -   **CSV Download:** A downloadable CSV file containing the full photometry results table.

### Tab 5: H-R Diagram

Generate a Hertzsprung-Russell (H-R) diagram or Color-Magnitude Diagram (CMD).
-   **Upload CSV:** Upload the CSV file generated from Tab 4. This file should contain columns for V-magnitude (e.g., 'StdMag_V') and B-V color (e.g., 'B-V').
-   **Object Name:** Optionally provide an object name for the diagram's title (e.g., cluster name).
-   **Output:** An H-R diagram image (PNG) plotting V magnitude against B-V color.

## Output Directories

Generated files are stored in the following directories within the application's root folder:
-   `masters_output/`: Stores generated Master BIAS, Master DARKs, and preliminary Master FLATs from Tab 1. Also stores uploaded master frames and temporary files used during master generation.
-   `calibrated_lights_output/`: Stores calibrated LIGHT frames from Tab 2 and Tab 4, as well as photometry results (CSVs, PNG previews including H-R diagrams and extinction plots) from Tabs 3, 4, and 5.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.