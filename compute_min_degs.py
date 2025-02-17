import numpy as np
from IME import IME

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


def compute_min_multdepth(alpha, epsilon):
    """
    Compute the minimum multiplication and depth for given alpha and epsilon.

    Parameters:
        alpha (float): Precision parameter.
        epsilon (float): Error threshold.

    Returns:
        mindep (int): Minimum depth.
        minmult (int): Minimum multiplications.
        optimal_degs (list): Optimal polynomial degrees.
    """
    
    # 최대 다항식 차수, 최대 연산 수, 최대 깊이 설정
    maxdeg = 31  # 최대 다항식 차수 (odd)
    m_max, n_max = 70, 40  # 최대 곱셈 연산 수, 최대 깊이
    target = (1 - epsilon) / (1 + epsilon)  # 첫 번째 도메인
    print(f"------------------------------------")
    print(f"alpha: {alpha}")
    print(f"epsilon: {epsilon}")

    # alpha에 대한 정수 a 찾기
    a = 0
    while a + 1 < alpha + 0.5:
        a += 1

    # h(m, n), G(m, n) 초기화
    h_table = np.zeros((m_max + 1, n_max + 1))
    G_table = [[[] for _ in range(n_max + 1)] for _ in range(m_max + 1)]

    # h(m, n), G(m, n) 계산
    for m in range(m_max + 1):
        for n in range(n_max + 1):
            G_table[m][n].clear()

            # m<=1 또는 n<=1인 경우
            if m <= 1 or n <= 1:
                h_table[m][n] = 2 ** (1 - alpha)
            else:
                best_k = None
                max_val = 0
                
                for k in range(1, (maxdeg // 2) + 1):  # k는 1부터 시작, 홀수 차수만 고려
                    deg = 2 * k + 1
                    if mult(deg) > m or dep(deg) > n or deg > maxdeg:
                        continue

                    temp = IME(h_table[m - mult(deg)][n - dep(deg)], (0, 1), deg, 1e-2, 100, 1, 100)

                    if temp > max_val:
                        best_k = k
                        max_val = temp

                if best_k is not None:
                    deg = 2 * best_k + 1
                    h_table[m][n] = max_val
                    G_table[m][n] = [deg] + G_table[m - mult(deg)][n - dep(deg)]

    # ComputeMinDep: 최소 깊이 찾기
    mindep = next((n for n in range(n_max + 1) if h_table[m_max][n] >= target), None)
    if mindep is None:
        print("Failure: Could not find minimum depth.")
        return None, None, None

    print(f"mindep: {mindep}\n")

    # ComputeMinMultDegs: 최소 곱셈 연산 및 최적 다항식 차수 찾기
    optimal_degs = None
    for dep_val in range(mindep, mindep + 10):
        print(f"Checking depth: {dep_val}")
        
        minmult = next((m for m in range(m_max + 1) if h_table[m][dep_val] >= target), None)
        if minmult is None:
            print("Failure: Could not find minimum multiplications.")
            continue
        
        print(f"minmult: {minmult}")
        print(f"depth: {dep_val}")
        print(f"h_table value: {h_table[minmult][dep_val]}")

        current_degs = G_table[minmult][dep_val]
        print("Optimal polynomial degrees:", current_degs, "\n")

        # 첫 번째 valid depth 값에서 최적의 다항식 차수 선택
        if dep_val == mindep:
            optimal_degs = current_degs

    return mindep, minmult, optimal_degs


# 사용 예시
alpha = 12
epsilon = 1e-3
mindep, minmult, optimal_degs = compute_min_multdepth(alpha, epsilon)

print(f"Final Results:")
print(f"Minimum Depth: {mindep}")
print(f"Minimum Multiplications: {minmult}")
print(f"Optimal Degrees: {optimal_degs}")
