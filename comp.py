import numpy as np
import matplotlib.pyplot as plt
from Remez import remez_algorithm, chebyshev_eval, sign_function

# Define the domain
domain1, domain2 = 0.3333333333333333, 1
degree1, degree2 = 7, 7

x_vals = np.linspace(-1, 1, 1000)

# Debug prints to see what remez_algorithm returns
coeffs_7, err = remez_algorithm(domain1, domain2, degree1)
print("First remez_algorithm call returns:")
print("coeffs_7:", coeffs_7)
print("err:", err)

coeffs_7_7, err_comp = remez_algorithm(1-err, 1+err, degree2)  # Note: removed domain2 parameter
print("\nSecond remez_algorithm call returns:")
print("coeffs_7_7:", coeffs_7_7)

# 비교
coeffs_15, err_15 = remez_algorithm(domain1, domain2, 15)

# Evaluate the polynomials
try:
    approx_7_7 = [chebyshev_eval(coeffs_7_7, chebyshev_eval(coeffs_7, x, degree1, 1), degree2, 1) for x in x_vals]
except Exception as e:
    print(f"\nError during polynomial evaluation:")
    print(f"Error type: {type(e)}")
    print(f"Error message: {str(e)}")
    print(f"Shape of coeffs_7: {np.array(coeffs_7).shape}")
    print(f"Shape of coeffs_7_7: {np.array(coeffs_7_7).shape}")

# True sign function
y_true = sign_function(x_vals)
y_15 = [chebyshev_eval(coeffs_15, x, 15, 1) for x in x_vals]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_true, label="Sign Function", linestyle="dashed")
plt.plot(x_vals, y_15, label="Chebyshev Approximation [15]")
try:
    plt.plot(x_vals, approx_7_7, label="Composite Polynomial [7, 7]")
except Exception as e:
    print(f"\nError during plotting:")
    print(f"Error message: {str(e)}")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Comparison of Composite Polynomial and Direct Polynomial Approximation")
plt.grid()
plt.show()

print(err_comp, err_15)