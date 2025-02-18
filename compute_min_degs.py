import numpy as np
from IME import IME
import time
from functools import lru_cache
from multiprocessing import Pool
import multiprocessing

# Memoize IME function calls
@lru_cache(maxsize=1000)
def cached_ime(target_err, deg):
    return IME(target_err, (0, 1), deg, 1e-2, 100, 1, 100, 20)

def dep(deg: int) -> int:
    """홀수 차수 deg(1,3,5,...)에 대한 깊이(depth)를 반환."""
    d = [0, 2, 3, 3, 4, 4, 4, 4, 
         5, 5, 5, 5, 5, 5, 5, 5, 
         6, 6, 6, 6, 6, 6, 6, 6, 
         6, 6, 6, 6, 6, 6, 6, 6]  # deg: 1,3,5,7,...,63
    return d[(deg - 1) // 2]

def mult(deg: int) -> int:
    """홀수 차수 deg(1,3,5,...)에 대한 곱셈 연산 수를 반환."""
    m = [0, 2, 3, 5, 5, 6, 7, 8, 
         8, 8, 9, 9, 10, 10, 11, 12, 
         11, 11, 11, 11, 12, 12, 13, 13, 
         14, 14, 14, 14, 15, 15, 16, 17]  # deg: 1,3,5,7,...,63
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

def compute_min_multdepth(alpha, epsilon):
    maxdeg = 15
    m_max, n_max = 10, 10 
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

    # Set up multiprocessing
    num_cores = multiprocessing.cpu_count()
    print(num_cores)
    pool = Pool(processes=num_cores)

    # Main computation with parallel processing
    for m in range(2, m_max + 1):
        for n in range(2, n_max + 1):
            print(f"Processing m={m}, n={n}")
            
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
    alpha = 12
    epsilon = 0.2
    print(epsilon)
    print((1 - epsilon) / (1 + epsilon))

    start_time = time.time()
    mindep, minmult, optimal_degs = compute_min_multdepth(alpha, epsilon)
    end_time = time.time()

    print(f"\nExecution Time: {end_time - start_time} seconds")
    print(f"\nResults:")
    print(f"Minimum Depth: {mindep}")
    print(f"Minimum Multiplications: {minmult}")
    print(f"Optimal Degrees: {optimal_degs}")
    print(f"Target Value: {(1 - epsilon) / (1 + epsilon)}")