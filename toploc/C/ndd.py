from typing import Tuple, List
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

NDD_CSRC_PATH = Path(__file__).parent / "csrc" / "ndd.cpp"

ndd_ops = load(name="ndd", sources=[NDD_CSRC_PATH], extra_cflags=["-O3"], verbose=True)


def compute_newton_coefficients(x: List[int], y: List[int]) -> List[int]:
    """
    """
    return ndd_ops.compute_newton_coefficients(x, y)

def evaluate_polynomial(coefficients: List[int], x: int) -> int:
    """
    """
    return ndd_ops.evaluate_polynomial(coefficients, x)
