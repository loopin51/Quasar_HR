# Functions for Tab 3: Atmospheric Extinction Coefficient Calculation
from scipy.stats import linregress

def calculate_extinction_coefficient(airmasses: list[float], magnitudes: list[float]) -> tuple[float, float]:
    """
    Calculates the atmospheric extinction coefficient and its uncertainty.

    The relationship between observed magnitude (m), magnitude at zero airmass (m0),
    extinction coefficient (k), and airmass (X) is given by:
    m = m0 - kX

    This function performs a linear regression on airmasses (X) and magnitudes (m)
    to find the slope (-k) and its standard error.

    Args:
        airmasses: A list of airmass values.
        magnitudes: A list of corresponding observed magnitude values.

    Returns:
        A tuple containing:
            - k (float): The calculated extinction coefficient.
            - k_err (float): The uncertainty in the extinction coefficient.
    """
    if not airmasses or not magnitudes:
        raise ValueError("Input lists cannot be empty.")
    if len(airmasses) != len(magnitudes):
        raise ValueError("Airmasses and magnitudes lists must have the same length.")
    if len(airmasses) < 2: # linregress requires at least 2 points
        raise ValueError("At least two data points are required for linear regression.")

    # Perform linear regression: y = slope*x + intercept
    # Here, magnitudes (m) is y, airmasses (X) is x.
    # So, m = slope*X + intercept.
    # Comparing with m = m0 - kX, we have:
    # slope = -k  => k = -slope
    # intercept = m0
    regression_result = linregress(airmasses, magnitudes)

    k = -regression_result.slope
    k_err = regression_result.stderr

    return k, k_err
