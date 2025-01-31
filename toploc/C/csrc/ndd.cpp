#include <torch/torch.h>

// TODO: Optimize the NDD memory usage
// TODO: Optimize the modular arithmetic operations to be minimum required to keep results positive

namespace py = pybind11;

constexpr int MOD_N = 65497;

inline int safeMod(long long v) {
    v = v % MOD_N;
    if (v < 0) v += MOD_N;
    return static_cast<int>(v);
}

// Current modInverse cause each iter q = a /m; then m - a %m; & a = 5. Basically a m is can siliently used in wrong roles
// More standart eea where we keep track of old_r and ols. So keep track of the current reamainder and coefficient.
// eea a^-1 (mod m)
int modInverse(int a, int m) {
    a = safeMod(a);
    if (m <= 1) {
        return 0;
    }
    
    // old_r, old_s track the "previous" remainder and coefficient
    // r, s track the "current" remainder and coefficient
    int old_r = a, r = m;
    int old_s = 1, s = 0;

    while (r != 0) {
        int q = old_r / r;     
        int temp_r = old_r - q*r;
        old_r = r;
        r = temp_r;

        int temp_s = old_s - q*s;
        old_s = s;
        s = temp_s;
    }

    // gcd(a, m) should always be 1 if a != 0 mod m. So better throw an error jic.
    if (old_r != 1) {
        throw std::runtime_error("No modular inverse: gcd(a, m) != 1.");
    }

    // Bezout coefficient for a.
    return safeMod(old_s);
}

// Helper function to calculate divided differences
std::vector<std::vector<int>> calculate_divided_differences(
    const std::vector<int>& x,
    const std::vector<int>& y
) {
    int n = x.size();
    std::vector<std::vector<int>> dd(n, std::vector<int>(n, 0));
    
    // Fill in the y-values in the first column
    for (int i = 0; i < n; i++) {
        dd[i][0] = y[i];
    }
    
    long long numer, denom, inv_denom;
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < n - j; i++) {
            numer = (dd[i + 1][j - 1] - dd[i][j - 1]) % MOD_N;
            denom = (x[i + j] - x[i]) % MOD_N;
            // denom = (denom + MOD_N) % MOD_N; should not be needed
            inv_denom = modInverse(denom, MOD_N);
            dd[i][j] = (numer * inv_denom) % MOD_N;
            dd[i][j] = (dd[i][j] + MOD_N) % MOD_N;
        }
    }
    
    return dd;
}
/*
 * using  n(n+1)/2 memory instead of n^2
 * U should be able to allocate dd[i] with size (n - i) rather than using an n x n matrix.
 * So u really only need to store the sub triangle doesn't work -> needs investigation.
 * The speedup and memory usage optimization is negliable. R1 wanted a 2D matrix. But I guess this should be even better.
 * Btw. compute_newton_coefficients R1 suggestion was to externalize mod. Hasn't had a huge impact on speed. Current impl already is very fast.
std::vector<std::vector<int>> calculate_divided_differences(
    const std::vector<int>& x,
    const std::vector<int>& y
) {
    int n = static_cast<int>(x.size());
    
    std::vector<std::vector<int>> dd(n);
    for (int i = 0; i < n; i++) {
        dd[i].resize(n - i);
    }

    for (int i = 0; i < n; i++) {
        long long val = (static_cast<long long>(y[i]) % MOD_N);
        if (val < 0) {
            val += MOD_N;
        }
        dd[i][0] = static_cast<int>(val);
    }

    // triangular structure
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < n - j; i++) {
            // numer = (dd[i+1][j-1] - dd[i][j-1]) mod MOD_N
            long long tmp_numer = static_cast<long long>(dd[i + 1][j - 1]) 
                                  - static_cast<long long>(dd[i][j - 1]);
            tmp_numer = tmp_numer % MOD_N;
            if (tmp_numer < 0) {
                tmp_numer += MOD_N;
            }
            long long numer = tmp_numer;

            long long xi_j = (static_cast<long long>(x[i + j]) % MOD_N);
            if (xi_j < 0) {
                xi_j += MOD_N;
            }
            long long xi = (static_cast<long long>(x[i]) % MOD_N);
            if (xi < 0) {
                xi += MOD_N;
            }
            long long tmp_denom = (xi_j - xi) % MOD_N;
            if (tmp_denom < 0) {
                tmp_denom += MOD_N;
            }
            long long denom = tmp_denom;

            int inv_denom = modInverse(static_cast<int>(denom), MOD_N);

            long long product = numer * inv_denom;
            product = product % MOD_N;
            if (product < 0) {
                product += MOD_N;
            }

            dd[i][j] = static_cast<int>(product);
        }
    }

    return dd;
}
*/

// core idea we can c = [c0,c1,...,c{n-1}] we can do the polynomial expansion in a single pass.
// INstead of mulitiplying the polynomials inside a nested loop repeatly incurring a factor of n we keep a "factor polynomial" that updates each time by multipllying it once by (x-x[i])
//   P(x) = c0+c1*x+c2*x^2+... +c{n-1}*x^{n-1}(mod MOD_N)
// with interpolation points (x[i],y[i]).
// so we have O(n^2) 
std::vector<int> compute_newton_coefficients(const std::vector<int>& x,
                                            const std::vector<int>& y)
{
    TORCH_CHECK(x.size() == y.size(),
                "Input vectors must have the same size");
    TORCH_CHECK(!x.empty(),
                "Input vectors must not be empty");

    const int n = static_cast<int>(x.size());

    //int* dd = (int*)alloca(n * sizeof(int)); // std::vector<int> dd(n);
    std::vector<int> dd(n);
    for (int i = 0; i < n; i++) {
        dd[i] = safeMod(y[i]);
    }

    // compute the divided differences in plaace
    //       dd[i]=(dd[i]-dd[i-1])/(x[i]-x[i-k])(mod MOD_N)
    for (int k = 1; k < n; k++) {
        for (int i = n - 1; i >= k; i--) {
            int numer = safeMod((long long)dd[i] - dd[i - 1]);

            // denominator = (x[i] - x[i-k]) mod
            long long denom = (long long)x[i] - (long long)x[i - k];
            int invDen = modInverse(safeMod(denom), MOD_N);

            dd[i] = safeMod((long long)numer * invDen);
        }
    }
    // after this dd[i] holds the i-th Newton coefficient
    // we expand the polynomial in standard form c(x)  we store the final polynomial in coeffs
   
    std::vector<int> coeffs(n, 0); 

    //int* factor = (int*)alloca(n * sizeof(int));
    std::vector<int> factor(n, 0);
    factor[0] = 1;
    for (int i = 1; i < n; i++) {
        factor[i] = 0;
    }

   
    for (int i = 0; i < n; i++) {
        const int dd_i = dd[i];

        for (int j = 0; j <= i; j++) {
            long long sumVal = (long long)coeffs[j]
                             + (long long)dd_i * factor[j];
            coeffs[j] = safeMod(sumVal);
        }

        // If we're not on the last iter we need update factor(x) *= (x - x[i]). factor is currently degree i becomes degree i+1
        if (i + 1 < n) {
            const int minusXi = safeMod(- (long long)x[i]);
            // In place update
            int prevVal = factor[0];
            factor[0] = safeMod((long long)prevVal * minusXi);
            for (int k = 1; k <= i + 1; k++) {
                int oldVal = factor[k];
                long long newVal = (long long)prevVal
                                 + (long long)oldVal * minusXi;
                factor[k] = safeMod(newVal);
                prevVal = oldVal;
            }
        }
    }

    return coeffs;
}

// Fix interpolation test case error
int evaluate_polynomial(const std::vector<int>& coefficients, int x)
{
    // Start with highest degree coefficient 
    long long result = coefficients.back();

    // Apply Horner's method 
    for (int i = static_cast<int>(coefficients.size()) - 2; i >= 0; i--) {
        result = (result * x + coefficients[i]) % MOD_N;
    }

    return static_cast<int>((result + MOD_N) % MOD_N);
}

// Python module definition
PYBIND11_MODULE(ndd, m) {
    m.doc() = "Newton's divided difference interpolation for polynomial congruences"; 

    m.def("compute_newton_coefficients", &compute_newton_coefficients, 
          "Compute coefficients of interpolating polynomial using Newton's divided difference method",
          py::arg("x"),
          py::arg("y"));
    
    m.def("evaluate_polynomial", &evaluate_polynomial,
          "Evaluate polynomial at given point using computed coefficients",
          py::arg("coefficients"),
          py::arg("x"));
}
