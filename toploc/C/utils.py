from typing import Tuple, List
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

UTILS_CSRC_PATH = Path(__file__).parent / "csrc" / "utils.cpp"

utils_ops = load(name="utils", sources=[UTILS_CSRC_PATH], extra_cflags=["-O3"], verbose=False)


def get_fp_parts(tensor: torch.Tensor) -> Tuple[List[int], List[int]]:
    """
    Get the exp and mantissa parts of a floating point tensor.

    Args:
        tensor: The input tensor.
        num_threads: The number of threads to use.
    Returns:
        exps: The exponent parts of the tensor.
        mantissas: The mantissa parts of the tensor.
    """
    return utils_ops.get_fp_parts(tensor)
