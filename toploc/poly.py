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
