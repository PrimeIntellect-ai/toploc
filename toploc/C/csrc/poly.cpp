#include <torch/torch.h>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include "./ndd.cpp"

// Namespace alias for pybind11
namespace py = pybind11;

class ProofPoly {
public:
    std::vector<int> coeffs;
    int modulus;

    ProofPoly(const std::vector<int>& coeffs_, int modulus_)
        : coeffs(coeffs_), modulus(modulus_) {}

    int call(int x) const {
        return evaluate_polynomial(coeffs, x);
    }

    size_t length() const {
        return coeffs.size();
    }

    static ProofPoly from_bytes(const std::string& data) {
        if (data.size() < 2) {
            throw std::invalid_argument("Data too short");
        }
        int modulus = (static_cast<unsigned char>(data[0]) << 8) | static_cast<unsigned char>(data[1]);
        std::vector<int> coeffs;
        for (size_t i = 2; i + 1 < data.size(); i += 2) {
            int coeff = (static_cast<unsigned char>(data[i]) << 8) | static_cast<unsigned char>(data[i + 1]);
            coeffs.push_back(coeff);
        }
        return ProofPoly(coeffs, modulus);
    }

    py::bytes to_bytes() const {
        // Create with exact size and fill later
        std::string result(2 + 2 * coeffs.size(), '\0');
        
        // Fill in bytes directly
        result[0] = static_cast<char>((modulus >> 8) & 0xFF);
        result[1] = static_cast<char>(modulus & 0xFF);
        
        // Fill coefficient bytes
        for (size_t i = 0; i < coeffs.size(); ++i) {
            result[2 + i * 2] = static_cast<char>((coeffs[i] >> 8) & 0xFF);
            result[2 + i * 2 + 1] = static_cast<char>(coeffs[i] & 0xFF);
        }
        
        return py::bytes(result);
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "ProofPoly[" << modulus << "](";
        oss << "[";
        for (size_t i = 0; i < coeffs.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << coeffs[i];
        }
        oss << "])";
        return oss.str();
    }
};

PYBIND11_MODULE(poly, m) {
    py::class_<ProofPoly>(m, "ProofPoly")
        .def(py::init<const std::vector<int>&, int>())
        .def("__call__", &ProofPoly::call)
        .def("__len__", &ProofPoly::length)
        .def("to_bytes", &ProofPoly::to_bytes)
        .def_static("from_bytes", &ProofPoly::from_bytes)
        .def("__repr__", &ProofPoly::repr);
}
