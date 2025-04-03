from typing import List, Union
import torch

class ProofPoly:
    coeffs: List[int]
    modulus: int

    def __init__(self, coeffs: List[int], modulus: int) -> None: ...
    def __call__(self, x: int) -> int: ...
    def __len__(self) -> int: ...
    @staticmethod
    def from_points(x: List[int], y: List[int]) -> "ProofPoly":
        """
        Create a polynomial from a list of x and y values.
        """
        ...

    @staticmethod
    def from_points_tensor(x: torch.Tensor, y: torch.Tensor) -> "ProofPoly":
        """
        Create a polynomial from a tensor of x and y values.
        x and y must be 1D tensors of the same length.
        x must be of dtype [int32, uint32, long]
        y must be of dtype [float16, bfloat16, float32]
        """
        ...

    @staticmethod
    def null(length: int) -> "ProofPoly":
        """
        Create a null polynomial of a given length.
        """
        ...

    def to_bytes(self) -> bytes:
        """
        Convert the polynomial to a bytes object.
        """
        ...

    def to_base64(self) -> str:
        """
        Convert the polynomial to a base64 string.
        """
        ...

    @staticmethod
    def from_base64(base64_str: str) -> "ProofPoly":
        """
        Create a polynomial from a base64 string.
        """
        ...

    @staticmethod
    def from_bytes(data: Union[str, bytes]) -> "ProofPoly":
        """
        Create a polynomial from a bytes object.
        """
        ...

    def __repr__(self) -> str: ...

class VerificationResult:
    exp_mismatches: int
    mant_err_mean: float
    mant_err_median: float

    def __init__(
        self, exp_mismatches: int, mant_err_mean: float, mant_err_median: float
    ) -> None: ...
    def __repr__(self) -> str: ...

def verify_proofs(
    activations: torch.Tensor,
    proofs: List[ProofPoly],
    decode_batching_size: int,
    topk: int,
) -> List[VerificationResult]:
    """
    Verify proofs for a given set of activations.

    Args:
        activations: A 2D tensor of shape (sequence_length, hidden_size)
        proofs: A list of ProofPoly objects
        decode_batching_size: The number of activations to process in a single batch
        topk: The number of top activations to consider for verification
    """
    ...

def verify_proofs_bytes(
    activations: torch.Tensor, proofs: List[str], decode_batching_size: int, topk: int
) -> List[VerificationResult]:
    """
    Verify proofs for a given set of activations.

    Args:
        activations: A 2D tensor of shape (sequence_length, hidden_size)
        proofs: A list of proof bytes
        decode_batching_size: The number of activations to process in a single batch
        topk: The number of top activations to consider for verification
    """
    ...

def verify_proofs_base64(
    activations: torch.Tensor, proofs: List[str], decode_batching_size: int, topk: int
) -> List[VerificationResult]:
    """
    Verify proofs for a given set of activations.

    Args:
        activations: A 2D tensor of shape (sequence_length, hidden_size)
        proofs: A list of proof strings in base64 format
        decode_batching_size: The number of activations to process in a single batch
        topk: The number of top activations to consider for verification
    """
    ...
