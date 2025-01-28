from typing import List
import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Commit:
    model_name: str
    device: str
    dtype: str
    engine: str
    hashes: List[str]
    completion: List[int] | None
    input_tokens: int
    generation_config: dict

    def __repr__(self):
        _json = dataclasses.asdict(self)
        return json.dumps(_json)

    def to_file(self, path: str | Path):
        with open(path, "w") as f:
            f.write(repr(self))

    @classmethod
    def from_file(cls, path: str | Path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


import base64
from .ndd import compute_newton_coefficients, evaluate_polynomial
import torch

def find_injective_modulus(x: list[int]) -> int:
    for i in range(65497, 2 ** 15, -1):
        if len(set([j % i for j in x])) == len(x):
            return i
    raise ValueError("No injective modulus found!")

class ProofPoly:
    def __init__(self, coeffs: list[int], modulus: int):
        self.coeffs = coeffs
        self.modulus = modulus

    def __call__(self, x: int):
        return evaluate_polynomial(self.coeffs, x % self.modulus)

    def __len__(self):
        return len(self.coeffs)
    
    @classmethod
    def from_points(cls, x: list[int] | torch.Tensor, y: list[int] | torch.Tensor) -> 'ProofPoly':
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        if isinstance(y, torch.Tensor):
            if y.dtype == torch.bfloat16:
                y = y.view(dtype=torch.uint16).tolist()
            elif y.dtype == torch.float32:
                raise NotImplementedError("float32 not supported yet because interpolate has hardcode prime")
            else:
                y = y.tolist()
        modulus = find_injective_modulus(x)
        x = [i % modulus for i in x]
        return cls(compute_newton_coefficients(x, y), modulus)

    def to_base64(self):
        base64_encoded = base64.b64encode(self.to_bytes()).decode('utf-8')
        return base64_encoded
    
    def to_bytes(self):
        return self.modulus.to_bytes(2, byteorder='big', signed=False) + b''.join(coeff.to_bytes(2, byteorder='big', signed=False) for coeff in self.coeffs)
    
    @classmethod
    def from_bytes(cls, byte_data: bytes) -> 'ProofPoly':
        modulus = int.from_bytes(byte_data[:2], byteorder='big', signed=False)
        coeffs = [int.from_bytes(byte_data[i:i+2], byteorder='big', signed=False) for i in range(2, len(byte_data), 2)]
        return cls(coeffs, modulus)
    
    @classmethod
    def from_base64(cls, base64_encoded: str) -> 'ProofPoly':
        byte_data = base64.b64decode(base64_encoded)
        return cls.from_bytes(byte_data)

    def __repr__(self) -> str:
        return f'ProofPoly[{self.modulus}]({self.coeffs})'
