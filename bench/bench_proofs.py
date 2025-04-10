from toploc import verify_proofs_base64, verify_proofs_bytes, verify_proofs, ProofPoly
from toploc.poly import build_proofs_base64, build_proofs, build_proofs_bytes
import torch
import multiprocessing as mp
from triton.testing import do_bench

DECODE_BATCHING_SIZE = 32
TOPK = 128
SKIP_PREFILL = True
BENCH_ITERATIONS = 5
PRINT_RESULTS = True
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

    time_ms = do_bench(
        lambda: verify_proofs(
            activations,
            proofs,
            decode_batching_size=DECODE_BATCHING_SIZE,
            topk=TOPK,
            skip_prefill=SKIP_PREFILL,
        ),
        rep=BENCH_ITERATIONS,
        warmup=2,
    )
    print(f"Pure Time taken: {time_ms:.2f} ms")

    results = verify_proofs(
        activations,
        proofs,
        decode_batching_size=DECODE_BATCHING_SIZE,
        topk=TOPK,
        skip_prefill=SKIP_PREFILL,
    )
    if PRINT_RESULTS:
        print("Results:")
        print(*results, sep="\n")


def subproc_bytes(proofs: list[bytes]):
    activations = get_deterministic_activations(SHAPE, 42, "cpu")

    time_ms = do_bench(
        lambda: verify_proofs_bytes(
            activations,
            proofs,
            decode_batching_size=DECODE_BATCHING_SIZE,
            topk=TOPK,
            skip_prefill=SKIP_PREFILL,
        ),
        rep=BENCH_ITERATIONS,
        warmup=2,
    )
    print(f"Bytes Time taken: {time_ms:.2f} ms")

    results = verify_proofs_bytes(
        activations,
        proofs,
        decode_batching_size=DECODE_BATCHING_SIZE,
        topk=TOPK,
        skip_prefill=SKIP_PREFILL,
    )
    if PRINT_RESULTS:
        print("Results:")
        print(*results, sep="\n")


def subproc_base64(proofs: list[str]):
    activations = get_deterministic_activations(SHAPE, 42, "cpu")

    time_ms = do_bench(
        lambda: verify_proofs_base64(
            activations,
            proofs,
            decode_batching_size=DECODE_BATCHING_SIZE,
            topk=TOPK,
            skip_prefill=SKIP_PREFILL,
        ),
        rep=BENCH_ITERATIONS,
        warmup=2,
    )
    print(f"Base64 Time taken: {time_ms:.2f} ms")

    results = verify_proofs_base64(
        activations,
        proofs,
        decode_batching_size=DECODE_BATCHING_SIZE,
        topk=TOPK,
        skip_prefill=SKIP_PREFILL,
    )
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
