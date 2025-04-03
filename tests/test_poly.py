import pickle
import pytest
import torch
import base64
from toploc.poly import (
    find_injective_modulus,
    build_proofs_bytes,
    build_proofs_base64,
    verify_proofs_bytes,
    verify_proofs_base64,
)
from toploc.C.csrc.poly import ProofPoly, VerificationResult


def test_find_injective_modulus():
    """Test finding injective modulus"""
    x = torch.randint(0, 4_000_000_000, (100,)).tolist()
    modulus = find_injective_modulus(x)
    assert isinstance(modulus, int)
    # Check that all values are unique under modulus
    modded = [i % modulus for i in x]
    assert len(set(modded)) == len(x)


@pytest.fixture
def sample_poly():
    return ProofPoly([1, 2, 3, 4], 65497)


def test_proof_poly_init(sample_poly):
    """Test initialization of ProofPoly"""
    assert sample_poly.coeffs == [1, 2, 3, 4]
    assert sample_poly.modulus == 65497


def test_proof_poly_call(sample_poly):
    """Test polynomial evaluation"""
    x = 42
    result = sample_poly(x)
    assert isinstance(result, int)
    assert result == (1 + 2 * x + 3 * x**2 + 4 * x**3) % 65497


def test_proof_poly_len(sample_poly):
    """Test length of polynomial"""
    assert len(sample_poly) == 4


def test_proof_poly_null():
    """Test null polynomial creation"""
    length = 5
    null_poly = ProofPoly.null(length)
    assert len(null_poly) == length
    assert null_poly.modulus == 0
    assert null_poly.coeffs == [0] * length


def test_proof_poly_from_points_list():
    """Test creation from list points"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    poly = ProofPoly.from_points(x, y)
    assert isinstance(poly, ProofPoly)
    assert len(poly.coeffs) > 0


def test_proof_poly_from_points_tensor():
    """Test creation from tensor points"""
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    poly = ProofPoly.from_points_tensor(x, y)
    assert isinstance(poly, ProofPoly)
    assert len(poly.coeffs) == 3
    assert poly(1) == 4
    assert poly(2) == 5
    assert poly(3) == 6


def test_proof_poly_from_points_bfloat16():
    """Test creation from bfloat16 tensor"""
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6], dtype=torch.bfloat16)
    poly = ProofPoly.from_points_tensor(x, y)
    assert isinstance(poly, ProofPoly)
    assert len(poly.coeffs) == 3


def test_proof_poly_to_base64(sample_poly):
    """Test base64 encoding"""
    encoded = sample_poly.to_base64()
    assert isinstance(encoded, str)
    # Verify it's valid base64
    base64.b64decode(encoded)


def test_proof_poly_to_bytes(sample_poly):
    """Test bytes conversion"""
    byte_data = sample_poly.to_bytes()
    assert isinstance(byte_data, bytes)
    assert len(byte_data) > 0


def test_proof_poly_from_bytes(sample_poly):
    """Test creation from bytes"""
    byte_data = sample_poly.to_bytes()
    reconstructed = ProofPoly.from_bytes(byte_data)
    assert reconstructed.coeffs == sample_poly.coeffs
    assert reconstructed.modulus == sample_poly.modulus


def test_proof_poly_from_base64(sample_poly):
    """Test creation from base64"""
    encoded = sample_poly.to_base64()
    reconstructed = ProofPoly.from_base64(encoded)
    assert reconstructed.coeffs == sample_poly.coeffs
    assert reconstructed.modulus == sample_poly.modulus


def test_proof_poly_repr(sample_poly):
    """Test string representation"""
    repr_str = repr(sample_poly)
    assert isinstance(repr_str, str)
    assert str(65497) in repr_str
    assert str([1, 2, 3, 4]) in repr_str


@pytest.fixture
def sample_activations():
    torch.manual_seed(42)
    DIM = 16
    a = [torch.randn(3, DIM, dtype=torch.bfloat16)]
    for _ in range(3 * 2 + 1):
        a.append(torch.randn(DIM, dtype=torch.bfloat16))
    return a


def test_build_proofs(sample_activations):
    """Test building proofs"""
    proofs = build_proofs_bytes(sample_activations, decode_batching_size=2, topk=5)
    assert isinstance(proofs, list)
    assert all(isinstance(p, bytes) for p in proofs)
    assert len(proofs) == 5


def test_build_proofs_base64(sample_activations):
    """Test building base64 proofs"""
    proofs = build_proofs_base64(sample_activations, decode_batching_size=2, topk=5)
    assert isinstance(proofs, list)
    assert all(isinstance(p, str) for p in proofs)
    # Verify each proof is valid base64
    for proof in proofs:
        base64.b64decode(proof)
    assert len(proofs) == 5


def test_build_proofs_skip_prefill(sample_activations):
    """Test building proofs with skip_prefill"""
    proofs = build_proofs_bytes(
        sample_activations[1:], decode_batching_size=2, topk=5, skip_prefill=True
    )
    assert isinstance(proofs, list)
    assert all(isinstance(p, bytes) for p in proofs)
    assert len(proofs) == 4

    proofs = build_proofs_base64(
        torch.randn(17, 16, dtype=torch.bfloat16),
        decode_batching_size=4,
        topk=5,
        skip_prefill=True,
    )
    assert isinstance(proofs, list)
    assert all(isinstance(p, str) for p in proofs)
    assert len(proofs) == 5


def test_build_proofs_error_handling():
    """Test error handling in proof building"""
    invalid_activations = [
        torch.randn(0, 16, dtype=torch.bfloat16),
        torch.randn(16, dtype=torch.bfloat16),
    ]
    proofs = build_proofs_bytes(invalid_activations, decode_batching_size=2, topk=5)
    assert isinstance(proofs, list)
    assert all(isinstance(p, bytes) for p in proofs)

    nullproof = ProofPoly.null(5).to_bytes()
    assert all(p == nullproof for p in proofs)


def test_build_proofs_edge_cases(sample_activations):
    """Test edge cases for proof building"""
    # Test with minimal topk
    proofs_min = build_proofs_bytes(sample_activations, decode_batching_size=2, topk=1)
    assert len(proofs_min) > 0

    # Test with large batching size
    proofs_large_batch = build_proofs_bytes(
        sample_activations, decode_batching_size=10, topk=5
    )
    assert len(proofs_large_batch) > 0

    # Test with only one prefill activation
    proofs_one = build_proofs_bytes(
        sample_activations[:1], decode_batching_size=2, topk=5
    )
    assert len(proofs_one) == 1

    # Test with only one activation and skip_prefill
    proofs_one_skip = build_proofs_bytes(
        sample_activations[1:1], decode_batching_size=2, topk=5, skip_prefill=True
    )
    assert len(proofs_one_skip) == 0


def test_verify_proofs_bytes(sample_activations):
    """Test verification of proofs in bytes format"""
    # Generate proofs in bytes format
    proofs_bytes = build_proofs_bytes(
        sample_activations, decode_batching_size=3, topk=4
    )

    results = verify_proofs_bytes(
        [i * 1.01 for i in sample_activations],
        proofs_bytes,
        decode_batching_size=3,
        topk=4,
    )

    assert isinstance(results, list)
    assert all(isinstance(r, VerificationResult) for r in results)
    assert len(results) == len(proofs_bytes)
    assert all(r.exp_mismatches == 0 for r in results)
    assert all(r.mant_err_mean > 0 and r.mant_err_mean <= 2 for r in results)
    assert all(r.mant_err_median > 0 and r.mant_err_median <= 2 for r in results)


def test_verify_proofs_base64(sample_activations):
    """Test verification of proofs in base64 format"""
    # Generate proofs in base64 format
    proofs_base64 = build_proofs_base64(
        sample_activations, decode_batching_size=2, topk=5
    )

    results = verify_proofs_base64(
        sample_activations, proofs_base64, decode_batching_size=2, topk=5
    )

    assert isinstance(results, list)
    assert all(isinstance(r, VerificationResult) for r in results)
    assert len(results) == len(proofs_base64)
    assert all(r.exp_mismatches == 0 for r in results)
    assert all(r.mant_err_mean == 0 for r in results)
    assert all(r.mant_err_median == 0 for r in results)


def test_verify_proofs_bytes_invalid(sample_activations):
    # Generate proofs in bytes format
    proofs_bytes = build_proofs_bytes(
        sample_activations, decode_batching_size=3, topk=4
    )

    results = verify_proofs_bytes(
        [i * 1.10 for i in sample_activations],
        proofs_bytes,
        decode_batching_size=3,
        topk=4,
    )

    print(results)
    assert isinstance(results, list)
    assert all(isinstance(r, VerificationResult) for r in results)
    assert len(results) == len(proofs_bytes)
    assert all(r.exp_mismatches <= 2 for r in results)
    assert all(r.mant_err_mean > 10 for r in results)
    assert all(r.mant_err_median > 10 for r in results)


def test_verify_proofs_base64_no_intersection_invalid(sample_activations):
    """Test verification of invalid base64 proofs"""
    # Generate invalid proofs in base64 format
    proofs_base64 = build_proofs_base64(
        sample_activations, decode_batching_size=2, topk=5
    )

    results = verify_proofs_base64(
        [i * 4 for i in sample_activations],
        proofs_base64,
        decode_batching_size=2,
        topk=5,
    )

    print(results)
    assert isinstance(results, list)
    assert all(isinstance(r, VerificationResult) for r in results)
    assert len(results) == len(proofs_base64)
    assert all(r.exp_mismatches == 5 for r in results)
    assert all(r.mant_err_mean > 2**32 for r in results)
    assert all(r.mant_err_median > 2**32 for r in results)


def test_verify_proofs_bytes_skip_prefill():
    """Test verification of valid bytes proofs with skip_prefill"""
    # Generate invalid proofs in bytes format
    activations = torch.randn(16, 16, dtype=torch.bfloat16)
    proofs_bytes = build_proofs_bytes(
        activations, decode_batching_size=3, topk=4, skip_prefill=True
    )

    assert len(proofs_bytes) == 6

    results = verify_proofs_bytes(
        activations, proofs_bytes, decode_batching_size=3, topk=4, skip_prefill=True
    )

    assert isinstance(results, list)
    assert all(isinstance(r, VerificationResult) for r in results)
    assert len(results) == len(proofs_bytes)
    print(results)
    assert all(r.exp_mismatches == 0 for r in results)
    assert all(r.mant_err_mean == 0 for r in results)
    assert all(r.mant_err_median == 0 for r in results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_verify_proofs_bytes_skip_prefill_gpu():
    """Test verification of valid bytes proofs with skip_prefill on GPU"""
    activations = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
    proofs_bytes = build_proofs_bytes(
        activations, decode_batching_size=3, topk=4, skip_prefill=True
    )

    assert len(proofs_bytes) == 6

    results = verify_proofs_bytes(
        activations, proofs_bytes, decode_batching_size=3, topk=4, skip_prefill=True
    )

    assert isinstance(results, list)
    assert all(isinstance(r, VerificationResult) for r in results)
    assert len(results) == len(proofs_bytes)
    assert all(r.exp_mismatches == 0 for r in results)
    assert all(r.mant_err_mean == 0 for r in results)
    assert all(r.mant_err_median == 0 for r in results)


def test_pickleable_VerificationResult():
    result = VerificationResult(1, 2, 3)
    result_pickled = pickle.dumps(result)
    result_unpickled = pickle.loads(result_pickled)
    assert result_unpickled == result
    assert result_unpickled != VerificationResult(2, 2, 3)


def test_pickleable_ProofPoly():
    poly = ProofPoly([1, 2, 3], 4)
    poly_pickled = pickle.dumps(poly)
    poly_unpickled = pickle.loads(poly_pickled)
    assert poly_unpickled == poly
    assert poly_unpickled != ProofPoly([1, 2, 3], 5)
    assert poly_unpickled != ProofPoly([1, 2, 4], 4)
