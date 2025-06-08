# Tests for Tab 3 functions
import unittest
import numpy as np
from tab3_functions import calculate_extinction_coefficient # Assuming tab3_functions.py is in the PYTHONPATH

class TestCalculateExtinctionCoefficient(unittest.TestCase):

    def test_ideal_conditions(self):
        # m = m0 - kX
        # Let m0 = 10, k = 0.2
        # X = [1, 2, 3, 4, 5]
        # m = [10 - 0.2*1, 10 - 0.2*2, 10 - 0.2*3, 10 - 0.2*4, 10 - 0.2*5]
        # m = [9.8, 9.6, 9.4, 9.2, 9.0]
        airmasses = [1.0, 2.0, 3.0, 4.0, 5.0]
        magnitudes = [9.8, 9.6, 9.4, 9.2, 9.0]

        k, k_err = calculate_extinction_coefficient(airmasses, magnitudes)
        self.assertAlmostEqual(k, 0.2, places=5)
        self.assertTrue(k_err >= 0) # Uncertainty should be non-negative

    def test_noisy_data(self):
        # m = m0 - kX
        # Let m0 = 15, k = 0.15
        # X = [1.1, 1.5, 2.0, 2.5, 3.0]
        # m_ideal = [15 - 0.15*X_i for X_i in X]
        # m_ideal = [14.835, 14.775, 14.7, 14.625, 14.55]
        # Add some noise
        airmasses = [1.1, 1.5, 2.0, 2.5, 3.0]
        magnitudes_ideal = np.array([14.835, 14.775, 14.7, 14.625, 14.55])
        noise = np.array([0.01, -0.02, 0.005, -0.005, 0.015])
        magnitudes_noisy = list(magnitudes_ideal + noise)

        k, k_err = calculate_extinction_coefficient(airmasses, magnitudes_noisy)
        self.assertAlmostEqual(k, 0.15, delta=0.05) # Allow some deviation due to noise
        self.assertTrue(k_err >= 0)

    def test_insufficient_data(self):
        with self.assertRaisesRegex(ValueError, "At least two data points are required for linear regression."):
            calculate_extinction_coefficient([1.0], [10.0])

        # Also test with zero points, though the "empty lists" check might catch this first.
        with self.assertRaisesRegex(ValueError, "Input lists cannot be empty."):
            calculate_extinction_coefficient([], [])


    def test_empty_lists(self):
        with self.assertRaisesRegex(ValueError, "Input lists cannot be empty."):
            calculate_extinction_coefficient([], [])
        with self.assertRaisesRegex(ValueError, "Input lists cannot be empty."):
            calculate_extinction_coefficient([1.0, 2.0], [])
        with self.assertRaisesRegex(ValueError, "Input lists cannot be empty."):
            calculate_extinction_coefficient([], [10.0, 9.8])

    def test_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "Airmasses and magnitudes lists must have the same length."):
            calculate_extinction_coefficient([1.0, 2.0, 3.0], [10.0, 9.8])
        with self.assertRaisesRegex(ValueError, "Airmasses and magnitudes lists must have the same length."):
            calculate_extinction_coefficient([1.0, 2.0], [10.0, 9.8, 9.5])

if __name__ == '__main__':
    unittest.main()
