# Functions for Tab 5: H-R Diagram Plotting
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving files without GUI
import matplotlib.pyplot as plt
import os

def plot_hr_diagram(
    magnitudes: list[float] | np.ndarray,
    colors: list[float] | np.ndarray,
    output_image_path: str,
    title: str = "H-R Diagram",
    x_label: str = "Color Index (e.g., B-V)",
    y_label: str = "Absolute Magnitude",
    point_color: str = 'blue',
    point_size: int = 10
) -> bool:
    """
    Generates and saves an H-R diagram.

    Args:
        magnitudes: A list or NumPy array of magnitudes (apparent or absolute).
                    Brighter objects have smaller magnitude values.
        colors: A list or NumPy array of color indices (e.g., B-V).
                Should be the same length as magnitudes.
        output_image_path: Path to save the generated H-R diagram image (e.g., "hr_diagram.png").
        title: Optional title for the plot.
        x_label: Optional label for the x-axis (color index).
        y_label: Optional label for the y-axis (magnitude).
        point_color: Optional color for the scatter plot points.
        point_size: Optional size for the scatter plot points.

    Returns:
        True if the plot was successfully generated and saved, False otherwise.
    """
    # Initial check for None or completely empty sequences if not numpy arrays yet
    if magnitudes is None or colors is None or (not isinstance(magnitudes, np.ndarray) and not magnitudes) or \
       (not isinstance(colors, np.ndarray) and not colors):
        # This handles cases like magnitudes=[] or colors=None
        # If they are already numpy arrays, this check might be problematic, so np.asarray first is better.
        pass # Will be caught by size check after conversion

    try:
        magnitudes_arr = np.asarray(magnitudes, dtype=float)
        colors_arr = np.asarray(colors, dtype=float)
    except ValueError as e:
        print(f"Error: Could not convert magnitudes or colors to numeric arrays: {e}")
        return False

    if magnitudes_arr.size == 0 or colors_arr.size == 0:
        print("Error: Magnitudes or colors data is empty.")
        return False

    # For H-R diagrams, inputs are expected to be 1D arrays (list of magnitudes, list of colors)
    # Using .shape for comparison is fine, but primarily concerned with having equal number of points.
    if magnitudes_arr.ndim != 1 or colors_arr.ndim != 1 or magnitudes_arr.shape[0] != colors_arr.shape[0]:
        print(f"Error: Magnitudes (shape {magnitudes_arr.shape}) and colors (shape {colors_arr.shape}) must be 1D arrays of the same length.")
        return False

    fig, ax = plt.subplots(figsize=(8, 6)) # Create a new figure and axes

    ax.scatter(colors_arr, magnitudes_arr, s=point_size, c=point_color, alpha=0.7, edgecolors='none')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Invert the y-axis (magnitude) so brighter stars are at the top
    ax.invert_yaxis()

    # Optional: Add a grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            # Close the plot to free memory before returning
            plt.close(fig)
            return False

    try:
        plt.savefig(output_image_path)
        print(f"H-R diagram saved to {output_image_path}")
        # Close the plot to free memory
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error saving H-R diagram to {output_image_path}: {e}")
        # Close the plot to free memory
        plt.close(fig)
        return False
