#include <torch/torch.h>

// TODO: Optimize the NDD memory usage
// TODO: Optimize the modular arithmetic operations to be minimum required to keep results positive

namespace py = pybind11;

constexpr int MOD_N = 65497;

int modInverse(int a, int m) {
    a = (a + m) % m;  // Ensure a is positive
    int m0 = m;
    int y = 0, x = 1;
    
    if (m == 1)
        return 0;
    
    while (a > 1) {
        // q is quotient
        int q = a / m;
        int t = m;
        
        // m is remainder now
        m = a % m;
        a = t;
        t = y;
        
        // Update y and x
        y = x - q * y;
        x = t;
    }
    
    return (x + m0) % m0;  // Ensure result is positive
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
            denom = (denom + MOD_N) % MOD_N;
            inv_denom = modInverse(denom, MOD_N);
            dd[i][j] = (numer * inv_denom) % MOD_N;
            dd[i][j] = (dd[i][j] + MOD_N) % MOD_N;
        }
    }
    
    return dd;
}

// Main function to compute polynomial coefficients
std::vector<int> compute_newton_coefficients(
    const std::vector<int>& x,
    const std::vector<int>& y
) {
    TORCH_CHECK(x.size() == y.size(), "Input vectors must have the same size");
    TORCH_CHECK(!x.empty(), "Input vectors must not be empty");

    int n = x.size();
    std::vector<std::vector<int>> dd = calculate_divided_differences(x, y);
    std::vector<int> coefficients(n, 0);
    
    // Convert divided differences to polynomial coefficients
    coefficients[0] = dd[0][0];
    
    for (int i = 1; i < n; i++) {
        // Compute term: dd[0][i] * (x - x[0])(x - x[1])...(x - x[i-1])
        std::vector<int> term(i + 1, 0);
        term[0] = dd[0][i];
        
        for (int j = 0; j < i; j++) {
            // Multiply by (x - x[j])
            std::vector<int> temp = term;
            for (int k = i; k > 0; k--) {
                term[k] = temp[k - 1];
            }
            term[0] = 0;
            
            for (int k = 0; k <= i; k++) {
                term[k] -= (temp[k] * x[j]) % MOD_N;
                term[k] = (term[k] % MOD_N + MOD_N) % MOD_N;
            }
        }
        
        // Add term to coefficients
        for (int j = 0; j <= i; j++) {
            coefficients[j] += term[j];
            coefficients[j] %= MOD_N;
        }
    }
    
    return coefficients;
}

int evaluate_polynomial(
    const std::vector<int>& coefficients,
    int x
) {
    // Start with highest degree coefficient
    int result = coefficients.back();
    
    // Apply Horner's method
    for (int i = coefficients.size() - 2; i >= 0; i--) {
        result = (result * x) % MOD_N + coefficients[i];
        result %= MOD_N;
    }

    result = (result + MOD_N) % MOD_N;
    
    return result;
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
