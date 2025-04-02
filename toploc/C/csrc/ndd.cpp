#include <torch/torch.h>

namespace py = pybind11;

constexpr int MOD_N = 65497;

/**
 * Helper function 
 * Safely reduce an integer into the range [0, MOD_N-1],
 */
inline int safeMod(long long v) {
    v = v % MOD_N;
    if (v < 0) {
        v += MOD_N;
    }
    return static_cast<int>(v);
}

/**
 * Compute the modular inverse of a (mod m) using
 * standart EEA. https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
 * Throws if gcd(a, m) != 1.
 */
int modInverse(int a, int m) {
    a = safeMod(a);
    if (m <= 1) {
        return 0;  // No meaning if m <= 1
    }

    int old_r = a, r = m;  // remainders
    int old_s = 1, s = 0;  // coefficients for Bezout's identity

    while (r != 0) {
        int q = old_r / r;

        int tmp_r = old_r - q * r;
        old_r = r;
        r = tmp_r;

        int tmp_s = old_s - q * s;
        old_s = s;
        s = tmp_s;
    }

    // gcd(a, m) must be 1 if old_r == 1
    if (old_r != 1) {
        throw std::runtime_error("No modular inverse: gcd(a, m) != 1.");
    }

    return safeMod(old_s);
}

/**
 * Compute Newton polynomial coefficients
 * using an O(n^2) single-pass expansion.
 *
 * 1) First, compute "in-place" Newton coefficients (dd array).
 * 2) Then, expand in a single pass using a rolling factor polynomial.
 */
std::vector<int> compute_newton_coefficients(const std::vector<int>& x,
                                            const std::vector<int>& y)
{
    TORCH_CHECK(x.size() == y.size(), "Input vectors must have the same size");
    TORCH_CHECK(!x.empty(), "Input vectors must not be empty");

    int n = static_cast<int>(x.size());

    // In-place Newton Divided Differences (1D array)
    std::vector<int> dd(n);
    for (int i = 0; i < n; i++) {
        dd[i] = safeMod(y[i]);
    }

    // dd[i] = (dd[i] - dd[i-1]) / (x[i] - x[i-k]) (mod MOD_N)
    // for k in [1..n-1], i in [n-1..k..down]
    for (int k = 1; k < n; k++) {
        for (int i = n - 1; i >= k; i--) {
            int numerator = safeMod((long long)dd[i] - dd[i - 1]);
            long long denom = (long long)x[i] - (long long)x[i - k];
            int invDen = modInverse(safeMod(denom), MOD_N);

            dd[i] = safeMod((long long)numerator * invDen);
        }
    }

    // Now dd[i] is the i-th Newton coefficient.
    // Single-Pass Expansion into Standard Form
    // We'll accumulate final polynomial coeffs in 'coeffs'
    std::vector<int> coeffs(n, 0);

    // factor[] will represent the polynomial product (x - x[0])...(x - x[i-1])
    std::vector<int> factor(n, 0);
    factor[0] = 1; // initially 1

    for (int i = 0; i < n; i++) {
        // Add dd[i] * factor(x) to coeffs
        int dd_i = dd[i];
        for (int j = 0; j <= i; j++) {
            long long sumVal = (long long)coeffs[j] + (long long)dd_i * factor[j];
            coeffs[j] = safeMod(sumVal);
        }

        // Update factor(x) by multiplying by (x - x[i]) if i < n-1
        if (i + 1 < n) {
            int minusXi = safeMod(-(long long)x[i]);
            int prevVal = factor[0];
            factor[0] = safeMod((long long)prevVal * minusXi);
            for (int k = 1; k <= i + 1; k++) {
                int oldVal = factor[k];
                long long newVal = (long long)prevVal + (long long)oldVal * minusXi;
                factor[k] = safeMod(newVal);
                prevVal = oldVal;
            }
        }
    }

    return coeffs;
}


/**
 * Evaluate a polynomial at x using Horner's method.
 * Coefficients are in ascending order c[0] + c[1]*x + ...
 */
int evaluate_polynomial(const std::vector<int>& coefficients, int x)
{
    long long result = coefficients.back(); // start with highest-degree coeff 
    for (int i = static_cast<int>(coefficients.size()) - 2; i >= 0; i--) {
        result = (result * x + coefficients[i]) % MOD_N;
    }
    return safeMod(result);
}

/**
 * Evaluate a polynomial at multiple points using Horner's method.
 * Coefficients are in ascending order c[0] + c[1]*x + ...
 */
std::vector<int> evaluate_polynomials(const std::vector<int>& coefficients, const std::vector<int>& x)
{
    std::vector<int> results(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        results[i] = evaluate_polynomial(coefficients, x[i]);
    }
    return results;
}

PYBIND11_MODULE(ndd, m) {
    m.doc() = "Newton's divided difference interpolation for polynomial congruences";

    m.def("compute_newton_coefficients",
          &compute_newton_coefficients,
          "Compute expanded polynomial coefficients using Newton interpolation",
          py::arg("x"),
          py::arg("y"));

    m.def("evaluate_polynomial",
          &evaluate_polynomial,
          "Evaluate the polynomial at point x using Horner's method",
          py::arg("coefficients"),
          py::arg("x"));

    m.def("evaluate_polynomials",
          &evaluate_polynomials,
          "Evaluate the polynomial at points x using Horner's method",
          py::arg("coefficients"),
          py::arg("x"));    
}