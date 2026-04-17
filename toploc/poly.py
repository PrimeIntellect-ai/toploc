from toploc.C.csrc.ndd import (
    evaluate_polynomials,
)
from toploc.C.csrc.poly import (
    ProofPoly,
    verify_proofs_base64 as c_verify_proofs_base64,
    verify_proofs_bytes as c_verify_proofs_bytes,
    verify_proofs as c_verify_proofs,
    VerificationResult,
)
from toploc.C.csrc.utils import get_fp_parts
import torch
import logging
from statistics import mean, median
from typing import Dict, Any, List
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


def find_injective_modulus(x: list[int]) -> int:
    for i in range(65497, 2**15, -1):
        if len(set([j % i for j in x])) == len(x):
            return i
    raise ValueError("No injective modulus found!")  # pragma: no cover


def build_proofs(
    activations: list[torch.Tensor],
    decode_batching_size: int,
    topk: int,
    skip_prefill: bool = False,
) -> list[ProofPoly]:
    proofs = []

    # In order to not crash, we return null proofs if there is an error
    try:
        # Prefill
        if not skip_prefill:
            flat_view = activations[0].view(-1)
            topk_indices = flat_view.abs().topk(topk).indices
            topk_values = flat_view[topk_indices]
            proof = ProofPoly.from_points_tensor(topk_indices, topk_values)
            proofs.append(proof)

        # Batched Decode
        for i in range(
            0 if skip_prefill else 1, len(activations), decode_batching_size
        ):
            flat_view = torch.cat(
                [i.view(-1) for i in activations[i : i + decode_batching_size]]
            )
            topk_indices = flat_view.abs().topk(topk).indices
            topk_values = flat_view[topk_indices]
            topk_indices = topk_indices.to("cpu")
            topk_values = topk_values.to("cpu")
            proof = ProofPoly.from_points_tensor(topk_indices, topk_values)
            proofs.append(proof)
    except Exception as e:
        logger.error(f"Error building proofs: {e}")
        proofs = [ProofPoly.null(topk)] * (
            1 + (len(activations) - 1 + decode_batching_size) // decode_batching_size
        )

    return proofs


def build_proofs_bytes(
    activations: list[torch.Tensor],
    decode_batching_size: int,
    topk: int,
    skip_prefill: bool = False,
) -> list[bytes]:
    return [
        proof.to_bytes()
        for proof in build_proofs(activations, decode_batching_size, topk, skip_prefill)
    ]


def build_proofs_base64(
    activations: list[torch.Tensor],
    decode_batching_size: int,
    topk: int,
    skip_prefill: bool = False,
) -> list[str]:
    return [
        proof.to_base64()
        for proof in build_proofs(activations, decode_batching_size, topk, skip_prefill)
    ]


def batch_activations(
    activations: list[torch.Tensor],
    decode_batching_size: int,
    skip_prefill: bool = False,
) -> list[torch.Tensor]:
    batches = []

    # Prefill
    if not skip_prefill:
        flat_view = activations[0].view(-1)
        batches.append(flat_view)

    # Batched Decode
    for i in range(0 if skip_prefill else 1, len(activations), decode_batching_size):
        flat_view = torch.cat(
            [i.view(-1) for i in activations[i : i + decode_batching_size]]
        )
        batches.append(flat_view)

    return batches


def verify_proofs(
    activations: list[torch.Tensor],
    proofs: list[ProofPoly],
    decode_batching_size: int,
    topk: int,
    skip_prefill: bool = False,
) -> list[VerificationResult]:
    if isinstance(activations, torch.Tensor) and skip_prefill:
        return c_verify_proofs(activations, proofs, decode_batching_size, topk)
    results = []
    for proof, chunk in zip(
        proofs,
        batch_activations(
            activations,
            decode_batching_size=decode_batching_size,
            skip_prefill=skip_prefill,
        ),
    ):
        chunk = chunk.view(-1).cpu()
        topk_indices = chunk.abs().topk(k=topk).indices.tolist()
        topk_values = chunk[topk_indices]
        y_values = evaluate_polynomials(proof.coeffs, topk_indices)
        proof_topk_values = torch.tensor(y_values, dtype=torch.uint16).view(
            dtype=torch.bfloat16
        )

        exps, mants = get_fp_parts(proof_topk_values)
        proof_exps, proof_mants = get_fp_parts(topk_values)

        exp_mismatches = [i != j for i, j in zip(exps, proof_exps)]
        mant_errs = [
            abs(i - j) for i, j, k in zip(mants, proof_mants, exp_mismatches) if not k
        ]
        if len(mant_errs) > 0:
            results.append(
                VerificationResult(
                    sum(exp_mismatches), mean(mant_errs), median(mant_errs)
                )
            )
        else:
            results.append(VerificationResult(sum(exp_mismatches), 2**64, 2**64))
    return results


def verify_proofs_bytes(
    activations: list[torch.Tensor],
    proofs: list[bytes],
    decode_batching_size: int,
    topk: int,
    skip_prefill: bool = False,
) -> list[VerificationResult]:
    if isinstance(activations, torch.Tensor) and skip_prefill:
        return c_verify_proofs_bytes(activations, proofs, decode_batching_size, topk)
    return verify_proofs(
        activations,
        [ProofPoly.from_bytes(proof) for proof in proofs],
        decode_batching_size,
        topk,
        skip_prefill,
    )


def verify_proofs_base64(
    activations: list[torch.Tensor],
    proofs: list[str],
    decode_batching_size: int,
    topk: int,
    skip_prefill: bool = False,
) -> list[VerificationResult]:
    if isinstance(activations, torch.Tensor) and skip_prefill:
        return c_verify_proofs_base64(activations, proofs, decode_batching_size, topk)
    return verify_proofs(
        activations,
        [ProofPoly.from_base64(proof) for proof in proofs],
        decode_batching_size,
        topk,
        skip_prefill,
    )


# New functions for saving/loading proofs with indices


@dataclass
class ProofWithIndices:
    """Container for a proof polynomial along with its indices."""
    proof: ProofPoly
    indices: List[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "proof_base64": self.proof.to_base64(),
            "indices": self.indices,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofWithIndices":
        """Create from a dictionary."""
        return cls(
            proof=ProofPoly.from_base64(data["proof_base64"]),
            indices=data["indices"],
        )


def build_proofs_with_indices(
    activations: list[torch.Tensor],
    decode_batching_size: int,
    topk: int,
    skip_prefill: bool = False,
) -> list[ProofWithIndices]:
    """Build proofs along with their corresponding indices.
    
    This function returns ProofWithIndices objects that contain both the proof
    polynomial and the indices used to create it. This allows saving and later
    verifying proofs without needing the original activations.
    
    Args:
        activations: List of activation tensors
        decode_batching_size: Number of activations to batch together
        topk: Number of top activations to consider
        skip_prefill: Whether to skip the prefill (first) activation
        
    Returns:
        List of ProofWithIndices objects
    """
    proofs_with_indices = []

    try:
        # Prefill
        if not skip_prefill:
            flat_view = activations[0].view(-1)
            topk_indices = flat_view.abs().topk(topk).indices
            topk_values = flat_view[topk_indices]
            proof = ProofPoly.from_points_tensor(topk_indices, topk_values)
            proofs_with_indices.append(ProofWithIndices(
                proof=proof,
                indices=topk_indices.tolist()
            ))

        # Batched Decode
        for i in range(
            0 if skip_prefill else 1, len(activations), decode_batching_size
        ):
            flat_view = torch.cat(
                [i.view(-1) for i in activations[i : i + decode_batching_size]]
            )
            topk_indices = flat_view.abs().topk(topk).indices
            topk_values = flat_view[topk_indices]
            topk_indices = topk_indices.to("cpu")
            topk_values = topk_values.to("cpu")
            proof = ProofPoly.from_points_tensor(topk_indices, topk_values)
            proofs_with_indices.append(ProofWithIndices(
                proof=proof,
                indices=topk_indices.tolist()
            ))
    except Exception as e:
        logger.error(f"Error building proofs with indices: {e}")
        # Return null proofs with empty indices on error
        num_proofs = 1 + (len(activations) - 1 + decode_batching_size) // decode_batching_size
        proofs_with_indices = [
            ProofWithIndices(proof=ProofPoly.null(topk), indices=[])
            for _ in range(num_proofs)
        ]

    return proofs_with_indices


def save_proofs_with_indices(
    proofs_with_indices: list[ProofWithIndices],
    file_path: str,
) -> None:
    """Save proofs with indices to a JSON file.
    
    Args:
        proofs_with_indices: List of ProofWithIndices objects
        file_path: Path to the output JSON file
    """
    data = [pwi.to_dict() for pwi in proofs_with_indices]
    with open(file_path, "w") as f:
        json.dump(data, f)


def load_proofs_with_indices(file_path: str) -> list[ProofWithIndices]:
    """Load proofs with indices from a JSON file.
    
    Args:
        file_path: Path to the input JSON file
        
    Returns:
        List of ProofWithIndices objects
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return [ProofWithIndices.from_dict(item) for item in data]


def proofs_with_indices_to_bytes(
    proofs_with_indices: list[ProofWithIndices],
) -> bytes:
    """Serialize proofs with indices to bytes.
    
    Args:
        proofs_with_indices: List of ProofWithIndices objects
        
    Returns:
        Serialized bytes containing all proofs and indices
    """
    data = [pwi.to_dict() for pwi in proofs_with_indices]
    return json.dumps(data).encode("utf-8")


def proofs_with_indices_from_bytes(data: bytes) -> list[ProofWithIndices]:
    """Deserialize proofs with indices from bytes.
    
    Args:
        data: Serialized bytes
        
    Returns:
        List of ProofWithIndices objects
    """
    items = json.loads(data.decode("utf-8"))
    return [ProofWithIndices.from_dict(item) for item in items]


def verify_proofs_with_indices(
    activations: torch.Tensor,
    proofs_with_indices: list[ProofWithIndices],
    decode_batching_size: int,
    topk: int,
    skip_prefill: bool = False,
) -> list[VerificationResult]:
    """Verify proofs using stored indices.
    
    This function verifies proofs using the indices stored in ProofWithIndices
    objects instead of recomputing them from the activations. This is useful
    when you have saved proofs and want to verify them against different activations.
    
    Args:
        activations: Activation tensor (sequence_length, hidden_size)
        proofs_with_indices: List of ProofWithIndices objects with stored indices
        decode_batching_size: Number of activations to batch together
        topk: Number of top activations to consider
        skip_prefill: Whether to skip the prefill activation
        
    Returns:
        List of VerificationResult objects
    """
    results = []
    
    batches = batch_activations(
        activations,
        decode_batching_size=decode_batching_size,
        skip_prefill=skip_prefill,
    )
    
    for proof_with_idx, chunk in zip(proofs_with_indices, batches):
        proof = proof_with_idx.proof
        stored_indices = proof_with_idx.indices
        
        if not stored_indices:
            # No indices stored, can't verify
            results.append(VerificationResult(0, 2**64, 2**64))
            continue
            
        chunk = chunk.view(-1).cpu()
        
        # Use stored indices to get values
        stored_indices_tensor = torch.tensor(stored_indices, dtype=torch.long)
        topk_values = chunk[stored_indices_tensor]
        
        # Evaluate polynomial at stored indices
        y_values = evaluate_polynomials(proof.coeffs, stored_indices)
        proof_topk_values = torch.tensor(y_values, dtype=torch.uint16).view(
            dtype=torch.bfloat16
        )

        exps, mants = get_fp_parts(proof_topk_values)
        proof_exps, proof_mants = get_fp_parts(topk_values)

        exp_mismatches = [i != j for i, j in zip(exps, proof_exps)]
        mant_errs = [
            abs(i - j) for i, j, k in zip(mants, proof_mants, exp_mismatches) if not k
        ]
        if len(mant_errs) > 0:
            results.append(
                VerificationResult(
                    sum(exp_mismatches), mean(mant_errs), median(mant_errs)
                )
            )
        else:
            results.append(VerificationResult(sum(exp_mismatches), 2**64, 2**64))
    
    return results
