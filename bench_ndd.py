import timeit
from toploc.C.ndd import compute_newton_coefficients as compute_newton_coefficients_cpp
from toploc.ndd import compute_newton_coefficients as compute_newton_coefficients_py

MOD_N = 65497

n = 10000
x = list(range(n))
y = [i % MOD_N for i in range(n)]

t_py = timeit.timeit(lambda: compute_newton_coefficients_py(x, y), number=1)
print(f"Python (n={n}): {t_py:.2f}s")

t_cpp = timeit.timeit(lambda: compute_newton_coefficients_cpp(x, y), number=1)
print(f"C++ (n={n}): {t_cpp:.2f}s")

