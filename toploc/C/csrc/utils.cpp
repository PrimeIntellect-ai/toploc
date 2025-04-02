// Required PyTorch header for tensor operations
#include <torch/torch.h>
#include <omp.h>

// Namespace alias for pybind11
namespace py = pybind11;

// Get max number of CPU threads available
const int max_num_threads = std::thread::hardware_concurrency();

// Bit masks and constants for float32 and bfloat16 manipulation
namespace {
    // FP32: 1 bit sign, 8 bits exponent, 23 bits mantissa
    constexpr uint32_t FP32_EXP_MASK = 0x7F800000;  // bits 23-30
    constexpr uint32_t FP32_MANT_MASK = 0x007FFFFF; // bits 0-22
    constexpr int FP32_EXP_SHIFT = 23;
    
    // BF16: 1 bit sign, 8 bits exponent, 7 bits mantissa
    constexpr uint16_t BF16_EXP_MASK = 0x7F80;  // bits 7-14
    constexpr uint16_t BF16_MANT_MASK = 0x007F; // bits 0-6
    constexpr int BF16_EXP_SHIFT = 7;
}

// Main function to extract exponent and mantissa bits from tensor
std::tuple<std::vector<int32_t>, std::vector<int32_t>> get_fp_parts(
    const torch::Tensor& tensor,
    int num_threads = max_num_threads
) {
    // Input Validation
    TORCH_CHECK(tensor.device().is_cpu(), "Input tensor must be on CPU");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32 || tensor.dtype() == torch::kBFloat16,
               "Input tensor must be Float32 or BFloat16");
    TORCH_CHECK(num_threads > 0, "Number of threads must be positive");
    
    // Extract tensor properties
    bool is_bf16 = tensor.dtype() == torch::kBFloat16;
    size_t num_elements = tensor.numel();
    
    // Initialize vectors to store exponent and mantissa bits
    std::vector<int32_t> prefill_exps(num_elements);
    std::vector<int32_t> prefill_mants(num_elements);
    
    omp_set_num_threads(num_threads);
    
    if (is_bf16) {
        const uint16_t* bits_ptr = reinterpret_cast<const uint16_t*>(tensor.const_data_ptr<at::BFloat16>());
        #pragma omp parallel for
        for (size_t i = 0; i < num_elements; ++i) {
            uint16_t bits = bits_ptr[i];
            prefill_exps[i] = (bits & BF16_EXP_MASK) >> BF16_EXP_SHIFT;
            prefill_mants[i] = bits & BF16_MANT_MASK;
        }
    } else {
        const uint32_t* bits_ptr = reinterpret_cast<const uint32_t*>(tensor.const_data_ptr<float>());
        #pragma omp parallel for
        for (size_t i = 0; i < num_elements; ++i) {
            uint32_t bits = bits_ptr[i];
            prefill_exps[i] = (bits & FP32_EXP_MASK) >> FP32_EXP_SHIFT;
            prefill_mants[i] = bits & FP32_MANT_MASK;
        }
    }
    
    return std::make_tuple(std::move(prefill_exps), std::move(prefill_mants));
}

std::tuple<std::vector<int32_t>, std::vector<int32_t>> get_fp_parts_vec(
    const std::vector<uint16_t>& tensor,
    int num_threads = max_num_threads
) {
    // Extract tensor properties
    size_t num_elements = tensor.size();
    
    // Initialize vectors to store exponent and mantissa bits
    std::vector<int32_t> prefill_exps(num_elements);
    std::vector<int32_t> prefill_mants(num_elements);
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
        uint16_t bits = tensor[i];
        prefill_exps[i] = (bits & BF16_EXP_MASK) >> BF16_EXP_SHIFT;
        prefill_mants[i] = bits & BF16_MANT_MASK;
    }
    
    return std::make_tuple(std::move(prefill_exps), std::move(prefill_mants));
}

// Python module definition using pybind11
PYBIND11_MODULE(utils, m) {
    m.def(
        "get_fp_parts", &get_fp_parts, 
        "Get exponent and mantissa bits from float tensor (supports FP32 and BF16)",
        py::arg("tensor"),
        py::arg("num_threads") = max_num_threads
    );
}
