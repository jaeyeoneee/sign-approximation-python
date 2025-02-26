import numpy as np
import matplotlib.pyplot as plt
from Remez import remez_algorithm, chebyshev_eval, sign_function

# Define the domain and degree of the polynomials
domain1, domain2 = 0.096, 1
degree1, degree2 = 13, 13

# Remez to find the coefficients of the composite polynomial
first_coeffs, first_err = remez_algorithm(domain1, domain2, degree1)
print("First remez_algorithm call returns:")
print("first coeffs:", first_coeffs)
print("first err:", first_err)

second_coeffs, second_err = remez_algorithm(1-first_err, 1+first_err, degree2)  # Note: removed domain2 parameter
print("\nSecond remez_algorithm call returns:")
print("second coeffs:", second_coeffs)
print(", ".join(f"{x:.10f}" for x in second_coeffs))
print("second err:", second_err)

# Evaluate the polynomials
x_vals = np.linspace(-1, 1, 1000)
y_true = sign_function(x_vals)

try:
    y_approx = [chebyshev_eval(second_coeffs, chebyshev_eval(first_coeffs, x, degree1, 1), degree2, 1) for x in x_vals]
except Exception as e:
    print(f"\nError during polynomial evaluation:")
    print(f"Error type: {type(e)}")
    print(f"Error message: {str(e)}")

# Plot results
plt.figure(figsize=(10, 6))
print(x_vals)
print(y_true)
plt.plot(x_vals, y_true, label="Sign Function", linestyle="dashed")
try:
    plt.plot(x_vals, y_approx, label="Composite Polynomial [13, 13]")
except Exception as e:
    print(f"\nError during plotting:")
    print(f"Error message: {str(e)}")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Comparison of Composite Polynomial and Direct Polynomial Approximation")
plt.grid()
plt.show()