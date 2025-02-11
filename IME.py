from Remez import remez_algorithm


def ME(domain1, domain2, degree, tol, max_iter, w, extreme_precision):
    _, max_err = remez_algorithm(domain1, domain2, degree, tol, max_iter, w, extreme_precision)
    return max_err


def IME(target_err, search_range, degree, tol, max_iter, w, extreme_precision, IME_precision = 100, break_cond = 1e-5):
    left, right = search_range
    
    for _ in range(IME_precision):
        mid = (left + right) / 2
        ME_val = ME(1-mid, 1+mid, degree, tol, max_iter, w, extreme_precision)

        if ME_val < target_err:
            left = mid
        else:
            right = mid
            
        if (right - left) < break_cond:
            break
    
    return (left+right) / 2


if __name__ == "__main__":
    domain1 = 0.1
    domain2 = 1
    degree = 27
    tol = 1e-2
    max_iter = 100
    w = 1
    extreme_precision = 100
    IME_precision = 100
    
    # ME test
    print("ME: ", ME(domain1, domain2, degree, tol, max_iter, w, extreme_precision))
    
    # IME test
    target_err = 1e-3
    search_range = (0, 1)
    print("IME: ", IME(target_err, search_range, degree, tol, max_iter, w, extreme_precision, IME_precision))