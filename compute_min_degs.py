import os
import numpy as np
from IME import IME
import time
from functools import lru_cache
from multiprocessing import Pool
import multiprocessing
from Remez import remez_algorithm

# Memoize IME function calls
@lru_cache(maxsize=1000)
def cached_ime(target_err, deg):
    return IME(target_err, (0, 1), deg, 1e-2, 100, 1, 100, 20)


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


def process_degree(args):
    m, n, deg, target_err, h_prev = args
    if mult(deg) > m or dep(deg) > n:
        return None
    try:
        temp = cached_ime(h_prev, deg)
        return (deg, temp)
    except:
        return None


def save_results(optimal_degs, target_value):
    degree_str = "_".join(map(str, optimal_degs))
    filename = f"sign_{degree_str}.txt"
    filepath = os.path.join("result", filename)
    
    os.makedirs("result", exist_ok=True)
    
    with open(filepath, "w") as f:
        #  다항식 차수 개수 저장
        f.write(f"{len(optimal_degs)}\n")
        
        # first error
        print(optimal_degs[0])
        coeffs, err = remez_algorithm(1- target_value, 1, optimal_degs[0])
        print(coeffs)
        coeffs_str = " ".join(map(str, coeffs))
        f.write(f"{coeffs_str}\n")
        
        for i in range(len(optimal_degs) - 1):
            print(optimal_degs[i + 1])
            coeffs, err = remez_algorithm(1 - err, 1 + err, optimal_degs[i + 1])
            coeffs_str = " ".join(map(str, coeffs))
            f.write(f"{coeffs_str}\n")
        
        f.write(f"{target_value}\n")
    
    print(f"Results saved to {filepath}")
        
def compute_min_multdepth(alpha, epsilon):
    maxdeg = 15
    m_max, n_max = 30, 20
    target = (1 - epsilon) / (1 + epsilon)
    
    print(f"------------------------------------")
    print(f"alpha: {alpha}")
    print(f"epsilon: {epsilon}")

    # Initialize tables
    h_table = np.zeros((m_max + 1, n_max + 1))
    G_table = [[[] for _ in range(n_max + 1)] for _ in range(m_max + 1)]

    # Base cases
    for m in range(m_max + 1):
        for n in range(n_max + 1):
            if m <= 1 or n <= 1:
                h_table[m][n] = 2 ** (1 - alpha)
                
    print(2**(1-alpha))

    # Set up multiprocessing
    num_cores = multiprocessing.cpu_count()
    print(num_cores)
    pool = Pool(processes=num_cores)

    # Main computation with parallel processing
    for m in range(2, m_max + 1):
        for n in range(2, n_max + 1):
            print(f"Processing m={m}, n={n}")
            
            start_time_inner = time.time()
            
            # Generate all possible degrees to test
            degrees = range(1, (maxdeg // 2) + 1)
            args = [(m, n, 2*k+1, h_table[m - mult(2*k+1)][n - dep(2*k+1)], 
                    h_table[m - mult(2*k+1)][n - dep(2*k+1)]) 
                   for k in degrees 
                   if mult(2*k+1) <= m and dep(2*k+1) <= n]
            
            if not args:
                continue

            # Process degrees in parallel
            results = pool.map(process_degree, args)
            valid_results = [r for r in results if r is not None]
            
            if valid_results:
                best_deg, max_val = max(valid_results, key=lambda x: x[1])
                h_table[m][n] = max_val
                G_table[m][n] = [best_deg] + G_table[m - mult(best_deg)][n - dep(best_deg)]
                
                # Early termination if we found a solution
                if h_table[m][n] >= target:
                    break
            
            end_time_inner = time.time()
            print(f"Time for m={m}, n={n}: {end_time_inner - start_time_inner} seconds")

    pool.close()
    pool.join()

    # Find minimum depth
    mindep = next((n for n in range(n_max + 1) if h_table[m_max][n] >= target), None)
    if mindep is None:
        print("Failure: Could not find minimum depth.")
        return None, None, None

    # Find minimum multiplications and optimal degrees
    minmult = next((m for m in range(m_max + 1) if h_table[m][mindep] >= target), None)
    if minmult is None:
        print("Failure: Could not find minimum multiplications.")
        return None, None, None

    optimal_degs = G_table[minmult][mindep]

    return mindep, minmult, optimal_degs


if __name__ == "__main__":
    # alpha = 12
    # epsilon = 0.05
    # print(epsilon)
    # print((1 - epsilon) / (1 + epsilon))

    # start_time = time.time()
    # mindep, minmult, optimal_degs = compute_min_multdepth(alpha, epsilon)
    # end_time = time.time()

    # print(f"\nExecution Time: {end_time - start_time} seconds")
    # print(f"\nResults:")
    # print(f"Minimum Depth: {mindep}")
    # print(f"Minimum Multiplications: {minmult}")
    # print(f"Optimal Degrees: {optimal_degs}")
    # print(f"Target Value: {(1 - epsilon) / (1 + epsilon)}")
    
    # file save test
    save_results([13, 13], 0.9047619047619047)