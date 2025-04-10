from toploc import verify_proofs_base64, verify_proofs_bytes, verify_proofs, ProofPoly
from toploc.poly import build_proofs_base64, build_proofs, build_proofs_bytes
import torch
import multiprocessing as mp
from torch.utils.benchmark import Timer

DECODE_BATCHING_SIZE = 32
TOPK = 128
SKIP_PREFILL = True
BENCH_ITERATIONS = 5
PRINT_RESULTS = False
SHAPE = (1000, 5120)


def get_deterministic_activations(
    shape: tuple[int, ...], seed: int, device: str
) -> torch.Tensor:
    torch.manual_seed(seed)
    activations = torch.randn(shape, dtype=torch.bfloat16, device="cpu")
    activations = activations.to(device)
    return activations


def subproc_pure(proofs: list[ProofPoly]):
    activations = get_deterministic_activations(SHAPE, 42, "cpu")

    time_ms = (
        Timer(
            stmt="verify_proofs(activations, proofs, decode_batching_size=DECODE_BATCHING_SIZE, topk=TOPK, skip_prefill=SKIP_PREFILL)",
            globals={
                "verify_proofs": verify_proofs,
                "activations": activations,
                "proofs": proofs,
                "DECODE_BATCHING_SIZE": DECODE_BATCHING_SIZE,
                "TOPK": TOPK,
                "SKIP_PREFILL": SKIP_PREFILL,
            },
        )
        .timeit(BENCH_ITERATIONS)
        .mean
        * 1000
    )
    print(f"Pure Time taken: {time_ms:.2f} ms")

    results = verify_proofs(
        activations,
        proofs,
        decode_batching_size=DECODE_BATCHING_SIZE,
        topk=TOPK,
        skip_prefill=SKIP_PREFILL,
    )
    for result in results:
        assert result.exp_mismatches == 0
        assert result.mant_err_mean == 0
        assert result.mant_err_median == 0
    if PRINT_RESULTS:
        print("Results:")
        print(*results, sep="\n")


def subproc_bytes(proofs: list[bytes]):
    activations = get_deterministic_activations(SHAPE, 42, "cpu")

    time_ms = (
        Timer(
            stmt="verify_proofs_bytes(activations, proofs, decode_batching_size=DECODE_BATCHING_SIZE, topk=TOPK, skip_prefill=SKIP_PREFILL)",
            globals={
                "verify_proofs_bytes": verify_proofs_bytes,
                "activations": activations,
                "proofs": proofs,
                "DECODE_BATCHING_SIZE": DECODE_BATCHING_SIZE,
                "TOPK": TOPK,
                "SKIP_PREFILL": SKIP_PREFILL,
            },
        )
        .timeit(BENCH_ITERATIONS)
        .mean
        * 1000
    )
    print(f"Bytes Time taken: {time_ms:.2f} ms")

    results = verify_proofs_bytes(
        activations,
        proofs,
        decode_batching_size=DECODE_BATCHING_SIZE,
        topk=TOPK,
        skip_prefill=SKIP_PREFILL,
    )
    for result in results:
        assert result.exp_mismatches == 0
        assert result.mant_err_mean == 0
        assert result.mant_err_median == 0
    if PRINT_RESULTS:
        print("Results:")
        print(*results, sep="\n")


def subproc_base64(proofs: list[str]):
    activations = get_deterministic_activations(SHAPE, 42, "cpu")

    time_ms = (
        Timer(
            stmt="verify_proofs_base64(activations, proofs, decode_batching_size=DECODE_BATCHING_SIZE, topk=TOPK, skip_prefill=SKIP_PREFILL)",
            globals={
                "verify_proofs_base64": verify_proofs_base64,
                "activations": activations,
                "proofs": proofs,
                "DECODE_BATCHING_SIZE": DECODE_BATCHING_SIZE,
                "TOPK": TOPK,
                "SKIP_PREFILL": SKIP_PREFILL,
            },
        )
        .timeit(BENCH_ITERATIONS)
        .mean
        * 1000
    )
    print(f"Base64 Time taken: {time_ms:.2f} ms")

    results = verify_proofs_base64(
        activations,
        proofs,
        decode_batching_size=DECODE_BATCHING_SIZE,
        topk=TOPK,
        skip_prefill=SKIP_PREFILL,
    )
    for result in results:
        assert result.exp_mismatches == 0
        assert result.mant_err_mean == 0
        assert result.mant_err_median == 0
    if PRINT_RESULTS:
        print("Results:")
        print(*results, sep="\n")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.manual_seed(42)
    activations = torch.randn(SHAPE, dtype=torch.bfloat16)

    # Pure Proofs
    proofs = build_proofs(
        activations,
        decode_batching_size=DECODE_BATCHING_SIZE,
        topk=TOPK,
        skip_prefill=SKIP_PREFILL,
    )
    proc = mp.Process(target=subproc_pure, args=(proofs,))
    proc.start()
    proc.join()

    # Bytes Proofs
    proofs_bytes = build_proofs_bytes(
        activations,
        decode_batching_size=DECODE_BATCHING_SIZE,
        topk=TOPK,
        skip_prefill=SKIP_PREFILL,
    )
    proc = mp.Process(target=subproc_bytes, args=(proofs_bytes,))
    proc.start()
    proc.join()

    # Base64 Proofs
    proofs_base64 = build_proofs_base64(
        activations,
        decode_batching_size=DECODE_BATCHING_SIZE,
        topk=TOPK,
        skip_prefill=SKIP_PREFILL,
    )
    proc = mp.Process(target=subproc_base64, args=(proofs_base64,))
    proc.start()
    proc.join()
