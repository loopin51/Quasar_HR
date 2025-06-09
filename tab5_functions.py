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
    y_label: str = "Magnitude", # Changed from Absolute Magnitude to be more general
    colors_data: list[float] | np.ndarray | None = None, # New parameter for B-V values for colormap
    colormap: str = 'RdYlBu_r', # Default colormap
    point_size: int = 20 # Increased default point size for 'x' markers
) -> bool:
    """
    Generates and saves an H-R diagram.

    Args:
        magnitudes: A list or NumPy array of magnitudes.
        colors: A list or NumPy array of color indices for the x-axis (e.g., B-V).
        output_image_path: Path to save the generated H-R diagram image.
        title: Optional title for the plot.
        x_label: Optional label for the x-axis.
        y_label: Optional label for the y-axis.
        colors_data: Optional list/array of numerical values (e.g., B-V again, or other property)
                     to map to the colormap for scatter points. If None, points are single color.
        colormap: Optional name of the matplotlib colormap to use if colors_data is provided.
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

    scatter_plot = ax.scatter(
        colors_arr,
        magnitudes_arr,
        s=point_size,
        c=colors_data if colors_data is not None else 'blue', # Use colors_data if provided
        cmap=colormap if colors_data is not None else None,
        marker='x', # Changed marker to 'x'
        alpha=0.7,
        edgecolors='none' if colors_data is not None else 'grey' # Add edgecolors for single-color case
    )

    if colors_data is not None:
        cbar = fig.colorbar(scatter_plot, ax=ax, label="B-V Color Index") # Add colorbar

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
