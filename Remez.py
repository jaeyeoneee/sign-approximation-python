import numpy as np
import matplotlib.pyplot as plt


def sign_function(x):
    return np.where(x>0, 1, np.where(x<0, -1, 0))


def chebyshev_basis(n, x, w):
    rs = np.zeros(n+1)
    rs[0] = 1
    if n==0:
        return rs
    
    rs[1] = x/w
    if n == 1:
        return rs
    
    for k in range(2, n+1):
        rs[k] = 2*(x/w)*rs[k-1] - rs[k-2]
    
    return rs
    
    
def chebyshev_odd_basis(n, x, w):
    basis = []
    rs = chebyshev_basis(n, x, w)

    for i in range(n+1):
        if i % 2 == 1:
            basis.append(rs[i])

    return np.array(basis)



def chebyshev_eval(coeffs, x, degree, w):
    rs = chebyshev_odd_basis(degree, x, w)  
    result = 0
    for i in range(degree//2 + 1):
        result += coeffs[i] * rs[i]
    return result


def chebyshev_eval_error(coeffs, x, degree, w):
    return chebyshev_eval(coeffs, x, degree, w) - sign_function(x)


def find_extreme_points(coeffs, domain1, domain2, degree, w, precision = 100, extreme_precision = 1000):
    # extreme precision이 충분히 크지 않으면 극점을 정확히 찾지 못해 행렬 연산 과정에서 오류가 발생할 수 있음
    sc = (domain2 - domain1) / (degree * extreme_precision)
    
    x_prev1 = domain1
    x_prev2 = domain1 + sc
    x_curr = domain1 + 2 * sc
    
    y_prev1 = chebyshev_eval_error(coeffs, x_prev1, degree, w)
    y_prev2 = chebyshev_eval_error(coeffs, x_prev2, degree, w)
    extreme_point = []
    
    while x_curr < domain2:
        y_curr = chebyshev_eval_error(coeffs, x_curr, degree, w)
        
        if (y_prev2 > y_prev1 and y_prev2 > y_curr) or (y_prev2 < y_prev1 and y_prev2 < y_curr):
            left, right = x_prev1, x_curr
            is_max = y_prev2 > y_prev1
            
            for _ in range(precision):
                mid = (left + right) / 2
                delta = (right-left) / 4
                x_points = [mid - delta, mid, mid + delta]
                y_points = [chebyshev_eval_error(coeffs, x, degree, w) for x in x_points]

                argmax = np.argmax(y_points) if is_max else np.argmin(y_points)
                
                if argmax == 0:
                    right = mid
                elif argmax == 2:
                    left = mid
                else:
                    left, right = x_points[1], x_points[2]
                    
            x_extreme = (left+right) / 2
            extreme_point.append(x_extreme)
        
        x_prev1, x_prev2 = x_prev2, x_curr
        y_prev1, y_prev2 = y_prev2, y_curr
        x_curr += sc
        
    extreme_rs = np.sort(np.concatenate((np.array([domain1]), np.array(extreme_point), np.array([domain2]))))
    return extreme_rs


def remez_algorithm(domain1, domain2, degree, tol = 1e-2, max_iter = 100, w=1, extreme_precision = 100):

    if (degree % 2) == 0:
        raise ValueError("Degree must be odd for sign function approximation")
    
    v_num = degree//2 + 2

    # print("v_num: ", v_num) 
    
    # Step 1: Initial point generation
    x = np.linspace(domain1, domain2, v_num)
    
    # print("x: ", x)
    
    max_err = 0
    
    for i in range(max_iter):
        # print("------------")
        # print("iteration: ", i)
        # Step 2 :Solve linear system for coefficients and error
        A = np.zeros((v_num, v_num))
        b = sign_function(x)
    
        # print("b: ", b)
        
        for i in range(v_num):
            # print(v_num, i)
            basis_values = chebyshev_odd_basis(degree, x[i], w)
            A[i, :-1] = basis_values
            A[i, -1] = (-1)**(i+1)
        
        # print("A: ", A)
    
        sol = np.linalg.solve(A, b)

        # print("coeff and error: ", sol)
    
        # Step 3 4 : Find the new Extreme Points
        coeffs = sol[:-1]
    
        extreme_points = find_extreme_points(coeffs, domain1, domain2, degree, w, extreme_precision)
    
        # print("extreme_points: ", extreme_points)
        # print("extreme points size: ", len(extreme_points))
    
        max_err = np.max(np.array([np.abs(chebyshev_eval_error(coeffs, x, degree, w)) for x in extreme_points]))
        min_err = np.min(np.array([np.abs(chebyshev_eval_error(coeffs, x, degree, w)) for x in extreme_points]))
    
        if (max_err - min_err) / min_err < tol:
            break
        
        x = extreme_points
        
    return coeffs, max_err 
        
    
if __name__ == '__main__':
    # chebyshev_eval_error test
    domain1 = 0.1
    domain2 = 1.0
    degree = 23
    tol=1e-2
    max_iter=100
    w = 1
    extreme_precision = 1000
    
    coeffs, max_err = remez_algorithm(domain1, domain2, degree, tol, max_iter, w, extreme_precision)
    print("max_err: ", max_err)
    
    # Plot Chebyshev approximation
    x_vals = np.linspace(-1, 1, 1000)
    y_true = sign_function(x_vals)
    y_approx = [chebyshev_eval(coeffs, x, degree, w) for x in x_vals]
    
    plt.plot(x_vals, y_true, label="Sign Function", linestyle="dashed")
    plt.plot(x_vals, y_approx, label="Chebyshev Approximation")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Chebyshev Approximation of Sign Function")
    plt.grid()
    plt.show()