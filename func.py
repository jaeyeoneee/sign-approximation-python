import os
from Remez import remez_algorithm

# dep와 mult의 값은 openFHE 코드 중 CustomEvalPoly로 계산한 값을 사용
def dep(deg: int) -> int:
    """홀수 차수 deg(1,3,5,...)에 대한 깊이(depth)를 반환."""
    d = [0, 3, 4, 4, 5, 5, 5, 5, 
         6, 6, 6, 6, 6, 6, 6, 6]  # deg: 1,3,5,7,...,31
    return d[(deg - 1) // 2]


def mult(deg: int) -> int:
    """홀수 차수 deg(1,3,5,...)에 대한 곱셈 연산 수를 반환."""
    m = [0, 2, 4, 6, 8, 10, 12, 14,
         16, 18, 20, 22, 24, 26, 28, 30]  # deg: 1,3,5,7,...,63
    return m[(deg - 1) // 2]


import numpy as np

def add_zero_to_poly(coeffs, degree):
    result = np.zeros(degree+1)
    l = (degree+1)//2
    for i in range(l):
        result[2*i+1] = coeffs[i]
    
    return result


def save_results(optimal_degs, target_value):
    degree_str = "_".join(map(str, optimal_degs))
    filename = f"sign_{degree_str}.txt"
    filepath = os.path.join("result", filename)
    
    os.makedirs("result", exist_ok=True)
    
    with open(filepath, "w") as f:
        #  다항식 차수 개수 저장
        f.write(f"{len(optimal_degs)}\n")
        
        # first error
        chev_coeffs, err = remez_algorithm(1- target_value, 1, optimal_degs[0])
        print(chev_coeffs)
        power_coeffs = np.polynomial.chebyshev.cheb2poly(add_zero_to_poly(chev_coeffs, optimal_degs[0]))
        print(power_coeffs)
        coeffs_str = " ".join(map(str, power_coeffs))
        f.write(f"{coeffs_str}\n")
        
        for i in range(len(optimal_degs) - 1):
            chev_coeffs, err = remez_algorithm(1 - err, 1 + err, optimal_degs[i + 1])
            print(chev_coeffs)
            power_coeffs = np.polynomial.chebyshev.cheb2poly(add_zero_to_poly(chev_coeffs, optimal_degs[i+1]))
            coeffs_str = " ".join(map(str, power_coeffs))
            f.write(f"{coeffs_str}\n")
        
        f.write(f"{target_value}\n")
    
    print(f"Results saved to {filepath}")
        
