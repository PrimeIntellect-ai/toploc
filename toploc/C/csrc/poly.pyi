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
