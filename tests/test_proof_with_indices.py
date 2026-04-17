"""Tests for proof saving with indices functionality."""
import pytest
import torch
import tempfile
import os
from toploc.poly import (
    ProofWithIndices,
    build_proofs_with_indices,
    save_proofs_with_indices,
    load_proofs_with_indices,
    proofs_with_indices_to_bytes,
    proofs_with_indices_from_bytes,
    verify_proofs_with_indices,
)
from toploc.C.csrc.poly import VerificationResult


@pytest.fixture
def sample_activations():
    torch.manual_seed(42)
    DIM = 16
    a = [torch.randn(3, DIM, dtype=torch.bfloat16)]
    for _ in range(3 * 2 + 1):
        a.append(torch.randn(DIM, dtype=torch.bfloat16))
    return a


def test_build_proofs_with_indices(sample_activations):
    """Test building proofs with indices."""
    proofs_with_indices = build_proofs_with_indices(
        sample_activations, decode_batching_size=2, topk=5
    )
    assert isinstance(proofs_with_indices, list)
    assert all(isinstance(pwi, ProofWithIndices) for pwi in proofs_with_indices)
    assert len(proofs_with_indices) == 5
    
    # Each proof should have indices
    for pwi in proofs_with_indices:
        assert len(pwi.indices) == 5  # topk=5
        assert all(isinstance(idx, int) for idx in pwi.indices)


def test_proof_with_indices_to_dict():
    """Test ProofWithIndices serialization to dict."""
    from toploc.C.csrc.poly import ProofPoly
    
    proof = ProofPoly([1, 2, 3], 65497)
    pwi = ProofWithIndices(proof=proof, indices=[10, 20, 30])
    
    d = pwi.to_dict()
    assert "proof_base64" in d
    assert "indices" in d
    assert d["indices"] == [10, 20, 30]


def test_proof_with_indices_from_dict():
    """Test ProofWithIndices deserialization from dict."""
    from toploc.C.csrc.poly import ProofPoly
    
    proof = ProofPoly([1, 2, 3], 65497)
    pwi = ProofWithIndices(proof=proof, indices=[10, 20, 30])
    
    d = pwi.to_dict()
    pwi2 = ProofWithIndices.from_dict(d)
    
    assert pwi2.proof.coeffs == pwi.proof.coeffs
    assert pwi2.proof.modulus == pwi.proof.modulus
    assert pwi2.indices == pwi.indices


def test_save_and_load_proofs_with_indices(sample_activations):
    """Test saving and loading proofs with indices to/from file."""
    proofs_with_indices = build_proofs_with_indices(
        sample_activations, decode_batching_size=2, topk=5
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        save_proofs_with_indices(proofs_with_indices, temp_path)
        loaded = load_proofs_with_indices(temp_path)
        
        assert len(loaded) == len(proofs_with_indices)
        for orig, load in zip(proofs_with_indices, loaded):
            assert orig.proof.coeffs == load.proof.coeffs
            assert orig.proof.modulus == load.proof.modulus
            assert orig.indices == load.indices
    finally:
        os.unlink(temp_path)


def test_proofs_with_indices_to_and_from_bytes(sample_activations):
    """Test serializing and deserializing proofs with indices to bytes."""
    proofs_with_indices = build_proofs_with_indices(
        sample_activations, decode_batching_size=2, topk=5
    )
    
    data = proofs_with_indices_to_bytes(proofs_with_indices)
    assert isinstance(data, bytes)
    
    loaded = proofs_with_indices_from_bytes(data)
    
    assert len(loaded) == len(proofs_with_indices)
    for orig, load in zip(proofs_with_indices, loaded):
        assert orig.proof.coeffs == load.proof.coeffs
        assert orig.proof.modulus == load.proof.modulus
        assert orig.indices == load.indices


def test_verify_proofs_with_indices(sample_activations):
    """Test verification of proofs using stored indices."""
    proofs_with_indices = build_proofs_with_indices(
        sample_activations, decode_batching_size=3, topk=4
    )
    
    # Verify with same activations (should pass)
    results = verify_proofs_with_indices(
        sample_activations,
        proofs_with_indices,
        decode_batching_size=3,
        topk=4,
    )
    
    assert isinstance(results, list)
    assert all(isinstance(r, VerificationResult) for r in results)
    assert len(results) == len(proofs_with_indices)
    assert all(r.exp_mismatches == 0 for r in results)
    assert all(r.mant_err_mean == 0 for r in results)
    assert all(r.mant_err_median == 0 for r in results)


def test_verify_proofs_with_indices_slightly_modified(sample_activations):
    """Test verification with slightly modified activations."""
    proofs_with_indices = build_proofs_with_indices(
        sample_activations, decode_batching_size=3, topk=4
    )
    
    # Slightly modify activations
    modified = [a * 1.01 for a in sample_activations]
    
    results = verify_proofs_with_indices(
        modified,
        proofs_with_indices,
        decode_batching_size=3,
        topk=4,
    )
    
    assert isinstance(results, list)
    assert all(isinstance(r, VerificationResult) for r in results)
    # Should have small errors due to modification
    assert all(r.exp_mismatches == 0 for r in results)
    assert all(0 < r.mant_err_mean <= 2 for r in results)


def test_build_proofs_with_indices_skip_prefill(sample_activations):
    """Test building proofs with indices and skip_prefill."""
    proofs_with_indices = build_proofs_with_indices(
        sample_activations[1:], decode_batching_size=2, topk=5, skip_prefill=True
    )
    assert isinstance(proofs_with_indices, list)
    assert all(isinstance(pwi, ProofWithIndices) for pwi in proofs_with_indices)
    assert len(proofs_with_indices) == 4


def test_build_proofs_with_indices_error_handling():
    """Test error handling in proof building with indices."""
    invalid_activations = [
        torch.randn(0, 16, dtype=torch.bfloat16),
        torch.randn(16, dtype=torch.bfloat16),
    ]
    proofs_with_indices = build_proofs_with_indices(
        invalid_activations, decode_batching_size=2, topk=5
    )
    assert isinstance(proofs_with_indices, list)
    # Should return null proofs with empty indices
    for pwi in proofs_with_indices:
        assert pwi.proof.modulus == 0
        assert pwi.indices == []
