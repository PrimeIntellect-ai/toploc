#include <torch/torch.h>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "./ndd.cpp"
#include "./utils.cpp"

#ifdef DEBUG
#define DEBUG_PRINT(x) std::cout << x << std::endl
#else
#define DEBUG_PRINT(x)
#endif

// Namespace alias for pybind11
namespace py = pybind11;

// Add base64 encoding/decoding functions
static const std::string base64_chars = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static std::string base64_encode(const std::string& input) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];
    const unsigned char* bytes_to_encode = reinterpret_cast<const unsigned char*>(input.c_str());
    size_t in_len = input.length();

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for(i = 0; i < 4; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; j < i + 1; j++)
            ret += base64_chars[char_array_4[j]];

        while((i++ < 3))
            ret += '=';
    }
    return ret;
}

static std::string base64_decode(const std::string& encoded_string) {
    size_t in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    while (in_len-- && (encoded_string[in_] != '=')) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; i < 3; i++)
                ret += char_array_3[i];
            i = 0;
        }
    }

    if (i) {
        for (j = 0; j < i; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

        for (j = 0; j < i - 1; j++)
            ret += char_array_3[j];
    }

    return ret;
}

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

    bool operator==(const ProofPoly& other) const {
        return coeffs == other.coeffs && modulus == other.modulus;
    }

    bool operator!=(const ProofPoly& other) const {
        return !(*this == other);
    }

    py::tuple to_tuple() const {
        return py::make_tuple(coeffs, modulus);
    }

    static ProofPoly from_tuple(const py::tuple& tuple) {
        return ProofPoly(tuple[0].cast<std::vector<int>>(), tuple[1].cast<int>());
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

    static ProofPoly null(size_t length) {
        return ProofPoly(std::vector<int>(length, 0), 0);
    }

    std::string to_base64() const {
        return base64_encode(to_bytes());
    }

    static ProofPoly from_base64(const std::string& base64_str) {
        return from_bytes(base64_decode(base64_str));
    }

    // TODO: Make this work with int64_t x
    static ProofPoly from_points(const std::vector<int>& x, const std::vector<int>& y) {
        if (x.size() != y.size()) {
            throw std::invalid_argument("x and y must have the same length");
        }
        
        // Find injective modulus
        int modulus = 0;
        for (int i = 65497; i > 0; i--) {
            std::vector<int> modded;
            bool is_injective = true;
            for (int val : x) {
                int mod_val = val % i;
                if (std::find(modded.begin(), modded.end(), mod_val) != modded.end()) {
                    is_injective = false;
                    break;
                }
                modded.push_back(mod_val);
            }
            if (is_injective) {
                modulus = i;
                break;
            }
        }
        
        if (modulus == 0) {
            throw std::runtime_error("No injective modulus found!");
        }

        // Apply modulus to x values
        std::vector<int> x_mod;
        for (int val : x) {
            x_mod.push_back(val % modulus);
        }

        // Compute Newton coefficients
        std::vector<int> coeffs = compute_newton_coefficients(x_mod, y);
        return ProofPoly(coeffs, modulus);
    }

    static ProofPoly from_points_tensor(const torch::Tensor& x, const torch::Tensor& y) {
        if (x.dim() != 1 || y.dim() != 1) {
            throw std::invalid_argument("x and y must be 1D tensors");
        }
        if (x.dtype() != torch::kInt32 && x.dtype() != torch::kLong) {
            throw std::invalid_argument("x must be an int32 or long tensor");
        }

        // TODO: Make this work with int64_t x
        std::vector<int> x_vec;
        if (x.dtype() == torch::kLong) {
            x_vec = std::vector<int>(x.const_data_ptr<int64_t>(), x.const_data_ptr<int64_t>() + x.numel());
        } else if (x.dtype() == torch::kInt32 || x.dtype() == torch::kUInt32) {
            x_vec = std::vector<int>(x.const_data_ptr<int>(), x.const_data_ptr<int>() + x.numel());
        } else {
            throw std::invalid_argument("x must be of dtype [int32, uint32, long]");
        } 
        
        // We dont support float32 yet
        std::vector<int> y_vec;
        if (y.dtype() == torch::kBFloat16) {
            y_vec = std::vector<int>(
                reinterpret_cast<const uint16_t*>(y.const_data_ptr<c10::BFloat16>()),
                reinterpret_cast<const uint16_t*>(y.const_data_ptr<c10::BFloat16>() + y.numel())
            );
        } else if (y.dtype() == torch::kFloat16) {
            y_vec = std::vector<int>(
                reinterpret_cast<const uint16_t*>(y.const_data_ptr<c10::Half>()),
                reinterpret_cast<const uint16_t*>(y.const_data_ptr<c10::Half>() + y.numel())
            );
        } else if (y.dtype() == torch::kInt32) {
            y_vec = std::vector<int>(
                y.const_data_ptr<int32_t>(),
                y.const_data_ptr<int32_t>() + y.numel()
            );
        } else if (y.dtype() == torch::kUInt32) {
            y_vec = std::vector<int>(
                y.const_data_ptr<uint32_t>(),
                y.const_data_ptr<uint32_t>() + y.numel()
            );
        } else if (y.dtype() == torch::kLong) {
            y_vec = std::vector<int>(
                y.const_data_ptr<int64_t>(),
                y.const_data_ptr<int64_t>() + y.numel()
            );
        } else if (y.dtype() == torch::kFloat32) {
            throw std::invalid_argument("float32 not supported yet because interpolate has hardcode prime");
        } else {
            throw std::invalid_argument("y must be of dtype [float16, bfloat16, float32]");
        }

        return from_points(x_vec, y_vec);
    }
};

// NOTE (Jack): Attributes should always be a measure of error, increasing the further we are from the proof
// This way, acceptance is always below the threshold and rejection is always above
// e.g. exp_match is bad, exp_mismatch is good
class VerificationResult {
public:
    int exp_mismatches;
    double mant_err_mean;
    double mant_err_median;

    VerificationResult(int exp_mismatches_, double mant_err_mean_, double mant_err_median_)
        : exp_mismatches(exp_mismatches_), mant_err_mean(mant_err_mean_), mant_err_median(mant_err_median_) {}

    std::string repr() const {
        std::ostringstream oss;
        oss << "VerificationResult[exp_mismatches=" << exp_mismatches 
            << ", mant_err_mean=" << mant_err_mean 
            << ", mant_err_median=" << mant_err_median << "]";
        return oss.str();
    }

    bool operator==(const VerificationResult& other) const {
        return exp_mismatches == other.exp_mismatches &&
            mant_err_mean == other.mant_err_mean &&
            mant_err_median == other.mant_err_median;
    }

    bool operator!=(const VerificationResult& other) const {
        return !(*this == other);
    }

    py::tuple to_tuple() const {
        return py::make_tuple(exp_mismatches, mant_err_mean, mant_err_median);
    }

    static VerificationResult from_tuple(const py::tuple& tuple) {
        return VerificationResult(tuple[0].cast<int>(), tuple[1].cast<double>(), tuple[2].cast<double>());
    }
};

std::vector<VerificationResult> verify_proofs(
    const torch::Tensor& activations,
    const std::vector<ProofPoly>& proofs,
    int decode_batching_size,
    int topk
) {
    std::vector<VerificationResult> results;
    
    // TODO: Do batches in parallel
    // Process activations in batches
    for (size_t proof_idx = 0; proof_idx < proofs.size(); proof_idx++) {
        // Get corresponding activation batch
        int batch_start = proof_idx * decode_batching_size;
        int batch_end = std::min(batch_start + decode_batching_size, (int)activations.numel());
        DEBUG_PRINT("activations: " << activations.sizes());
        DEBUG_PRINT("batch_start: " << batch_start);
        DEBUG_PRINT("batch_end: " << batch_end);
        torch::Tensor chunk = activations.slice(0, batch_start, batch_end);
        
        DEBUG_PRINT("chunk: " << chunk.sizes());
        chunk = chunk.view({-1});
        DEBUG_PRINT("chunk: " << chunk.sizes());
        // Get top-k indices and values
        auto topk_result = chunk.abs().topk(topk);
        // Note: Up till here, the tensors could be on GPU
        torch::Tensor topk_indices = std::get<1>(topk_result);
        torch::Tensor topk_values = chunk.index_select(0, topk_indices).cpu();
        topk_indices = topk_indices.cpu();

        DEBUG_PRINT("topk_indices: " << topk_indices.sizes());
        DEBUG_PRINT("topk_values: " << topk_values.sizes());

        // Try going through the pointers
        // for (int64_t i : *topk_indices.const_data_ptr<int64_t>()) {
        //     std::cout << "topk_indices[" << i << "]: " << i << std::endl;
        // }

        // Evaluate polynomial at topk indices
        std::vector<int> indices_vec(
            topk_indices.const_data_ptr<int64_t>(), 
            topk_indices.const_data_ptr<int64_t>() + topk_indices.numel()
        );

        DEBUG_PRINT("indices_vec: " << indices_vec.size());
        DEBUG_PRINT("proofs[proof_idx].coeffs: " << proofs[proof_idx].coeffs.size());
        std::vector<int> y_values = evaluate_polynomials(proofs[proof_idx].coeffs, indices_vec);
        
        // Convert to tensors for comparison
        std::vector<uint16_t> proof_values(y_values.begin(), y_values.end());

        // Get exponents and mantissas
        auto [exps, mants] = get_fp_parts_vec(proof_values);
        auto [proof_exps, proof_mants] = get_fp_parts(topk_values);

        DEBUG_PRINT("exps: " << exps.size());
        DEBUG_PRINT("proof_exps: " << proof_exps.size());
        DEBUG_PRINT("mants: " << mants.size());
        DEBUG_PRINT("proof_mants: " << proof_mants.size());

        // Calculate mismatches and errors
        std::vector<bool> exp_mismatches;
        std::vector<float> mant_errs;
        
        for (int i = 0; i < topk; i++) {
            bool exp_mismatch = exps[i] != proof_exps[i];
            exp_mismatches.push_back(exp_mismatch);
            if (!exp_mismatch) {
                mant_errs.push_back(std::abs(mants[i] - proof_mants[i]));
            }
        }

        // Calculate statistics
        int exp_mismatch_count = std::count(exp_mismatches.begin(), exp_mismatches.end(), true);
        double mean = 0.0;
        double median = 0.0;
        
        if (!mant_errs.empty()) {
            mean = std::accumulate(mant_errs.begin(), mant_errs.end(), 0.0) / mant_errs.size();
            std::sort(mant_errs.begin(), mant_errs.end());
            median = mant_errs[mant_errs.size() / 2];
        } else {
            mean = std::pow(2, 64);
            median = std::pow(2, 64);
        }

        results.push_back({exp_mismatch_count, mean, median});
    }

    return results;
}

std::vector<VerificationResult> verify_proofs_bytes(
    const torch::Tensor& activations,
    const std::vector<std::string>& proofs,
    int decode_batching_size,
    int topk
) {
    std::vector<ProofPoly> proofs_poly;
    for (const auto& proof : proofs) {
        proofs_poly.push_back(ProofPoly::from_bytes(proof));
    }
    return verify_proofs(activations, proofs_poly, decode_batching_size, topk);
}

std::vector<VerificationResult> verify_proofs_base64(
    const torch::Tensor& activations,
    const std::vector<std::string>& proofs,
    int decode_batching_size,
    int topk
) {
    std::vector<ProofPoly> proofs_poly;
    for (const auto& proof : proofs) {
        proofs_poly.push_back(ProofPoly::from_base64(proof));
    }
    return verify_proofs(activations, proofs_poly, decode_batching_size, topk);
}



PYBIND11_MODULE(poly, m) {
    py::class_<ProofPoly>(m, "ProofPoly")
        .def(py::init<const std::vector<int>&, int>())
        .def("__call__", &ProofPoly::call)
        .def("__len__", &ProofPoly::length)
        .def_static("from_points", &ProofPoly::from_points)
        .def_static("from_points_tensor", &ProofPoly::from_points_tensor)
        .def_static("null", &ProofPoly::null)
        .def("to_bytes", &ProofPoly::to_bytes)
        .def("to_base64", &ProofPoly::to_base64)
        .def_static("from_bytes", &ProofPoly::from_bytes)
        .def_static("from_base64", &ProofPoly::from_base64)
        .def("__repr__", &ProofPoly::repr)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::pickle(
            [](const ProofPoly &p) { return p.to_tuple(); },
            [](const py::tuple &t) { return ProofPoly::from_tuple(t); }
        ))
        .def_readwrite("coeffs", &ProofPoly::coeffs)
        .def_readwrite("modulus", &ProofPoly::modulus);

    py::class_<VerificationResult>(m, "VerificationResult")
        .def(py::init<int, double, double>())
        .def(py::pickle(
            [](const VerificationResult &v) { return v.to_tuple(); },
            [](const py::tuple &t) { return VerificationResult::from_tuple(t); }
        ))
        .def_readwrite("exp_mismatches", &VerificationResult::exp_mismatches)
        .def_readwrite("mant_err_mean", &VerificationResult::mant_err_mean)
        .def_readwrite("mant_err_median", &VerificationResult::mant_err_median)
        .def("__repr__", &VerificationResult::repr)
        .def(py::self == py::self)
        .def(py::self != py::self);
        
    m.def("verify_proofs", &verify_proofs, 
          py::arg("activations"), 
          py::arg("proofs"),
          py::arg("decode_batching_size"),
          py::arg("topk")
    );

    m.def("verify_proofs_bytes", &verify_proofs_bytes, 
          py::arg("activations"), 
          py::arg("proofs"),
          py::arg("decode_batching_size"),
          py::arg("topk")
    );

    m.def("verify_proofs_base64", &verify_proofs_base64, 
          py::arg("activations"), 
          py::arg("proofs"),
          py::arg("decode_batching_size"),
          py::arg("topk")
    );
}
