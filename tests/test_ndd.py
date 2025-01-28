import pytest
import random
#from toploc.C.ndd import compute_newton_coefficients, evaluate_polynomial
from toploc.ndd import compute_newton_coefficients, evaluate_polynomial

@pytest.mark.parametrize("x, y", [
    ([0], [42]),
    ([12, 11, 15], [0, 0, 0]),
    ([1, 3, 5, 2], [1, 5, 4, 1]),
    ([1, 8, 3], [1, 2, 3]),
    ([100, 55, 2], [1, 0, 1]),
])
def test_newton_interpolation_specific(x: list[int], y: list[int]):
    """Test Newton interpolation with specific known values"""
    x = [1, 3, 5, 2]
    y = [1, 5, 4, 1]
    
    # Compute interpolation coefficients
    coeffs = compute_newton_coefficients(x, y)
    
    # Verify interpolation at each point
    for xi, yi in zip(x, y):
        result = evaluate_polynomial(coeffs, xi)
        assert result == yi, f"Interpolation failed at x={xi}, expected {yi} but got {result}"

def test_newton_interpolation_random():
    """Test Newton interpolation with random points"""
    # Generate random unique x values
    x_values = random.sample(range(0, 65497), random.randint(5, 1000))
    
    # Generate random y values
    y_values = [random.randint(0, 2 ** 15) for _ in range(len(x_values))]
    
    # Compute interpolation coefficients
    coeffs = compute_newton_coefficients(x_values, y_values)
    
    # Verify interpolation at each point
    for xi, yi in zip(x_values, y_values):
        result = evaluate_polynomial(coeffs, xi)
        assert result == yi, f"Interpolation failed at x={xi}, expected {yi} but got {result} {x_values, y_values, coeffs}"

def test_error_conditions():
    """Test error handling"""
    # Test with empty lists
    with pytest.raises(Exception):
        compute_newton_coefficients([], [])
    
    # Test with mismatched lengths
    with pytest.raises(Exception):
        compute_newton_coefficients([1, 2], [1])
    