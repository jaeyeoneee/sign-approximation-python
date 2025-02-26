import numpy as np
import matplotlib.pyplot as plt

# C++에서 가져온 13차 다항식 계수
coeffs_7 = [0.0, 9.838210, 0.0, -128.56747, 0.0, 806.872804, 0.0, -2451.208474, 
            0.0, 3799.4958, 0.0, -2896.1413, 0.0, 860.87843]

coeffs_7_7 = [0.0, 2.977760, 0.0, -6.095633, 0.0, 9.26823, 0.0, -8.86180, 
              0.0, 5.1409, 0.0, -1.657773, 0.0, 0.22827]

# 다항식 평가 함수
def evaluate_polynomial(coeffs, x):
    return np.polyval(list(reversed(coeffs)), x)

# x 값 범위 설정 (-1 ~ 1)
x_vals = np.linspace(-1, 1, 1000)
y_vals_7 = evaluate_polynomial(coeffs_7, x_vals)
y_vals_7_7 = evaluate_polynomial(coeffs_7_7, y_vals_7)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals_7, label="Polynomial 7", color="red")
plt.plot(x_vals, y_vals_7_7, label="Polynomial 7_7", color="green")

plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Plot of Given 13th Degree Polynomials")
plt.legend()
plt.grid()
plt.show()
