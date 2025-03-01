# TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference

[TOPLOC](https://arxiv.org/abs/2501.16007) leverages locality sensitive hashing of intermediate activations to verify that LLM providers are using authorized model configurations and settings.

The feature set includes:
- Detect unauthorized modifications to models, prompts, and precision settings
- 1000x reduction in storage requirements compared to full activation storage
- Validation speeds up to 100x faster than original inference
- Robust across different hardware configurations and implementations

For code used by experiments in our paper, check out: https://github.com/PrimeIntellect-ai/toploc-experiments

## Installation

```bash
pip install -U toploc
```

## Usage

### Build proofs from activations:
As bytes (more compact when stored in binary formats):
```python
import torch
from toploc import build_proofs_bytes

torch.manual_seed(42)
activations = [torch.randn(5, 16, dtype=torch.bfloat16), *(torch.randn(16, dtype=torch.bfloat16) for _ in range(10))]
proofs = build_proofs_bytes(activations, decode_batching_size=3, topk=4, skip_prefill=False)

print(f"Activation shapes: {[i.shape for i in activations]}")
print(f"Proofs: {proofs}")
```
```python
Activation shapes: [torch.Size([5, 16]), torch.Size([16]), torch.Size([16]), torch.Size([16]), torch.Size([16]), torch.Size([16]), torch.Size([16]), torch.Size([16]), torch.Size([16]), torch.Size([16]), torch.Size([16])]
Proofs: [b'\xff\xd9\x1bB+g\xbaKum', b'\xff\xd9\xcb\xb8\x9a\xf1\x86%T\xa0', b'\xff\xd9\xb4h\xda\xe6\xe4\xabA\xb6', b'\xff\xd9\x80d\xd6X0\xe2\xafs', b'\xff\xd9\xd2\x04d\xea\x91\x91\xf6\xd7']
```

As base64 (more compact when stored in text formats):
```python
import torch
from toploc import build_proofs_base64

torch.manual_seed(42)
activations = [torch.randn(1, 5, 16, dtype=torch.bfloat16), *(torch.randn(1, 16, dtype=torch.bfloat16) for _ in range(10))]
proofs = build_proofs_base64(activations, decode_batching_size=3, topk=4, skip_prefill=False)

print(f"Activation shapes: {[i.shape for i in activations]}")
print(f"Proofs: {proofs}")
```
```python
Activation shapes: [torch.Size([1, 5, 16]), torch.Size([1, 16]), torch.Size([1, 16]), torch.Size([1, 16]), torch.Size([1, 16]), torch.Size([1, 16]), torch.Size([1, 16]), torch.Size([1, 16]), torch.Size([1, 16]), torch.Size([1, 16]), torch.Size([1, 16])]
Proofs: ['/9kbQitnukt1bQ==', '/9nLuJrxhiVUoA==', '/9m0aNrm5KtBtg==', '/9mAZNZYMOKvcw==', '/9nSBGTqkZH21w==']
```

### Verify proofs:
```python
import torch
from toploc import verify_proofs_base64

torch.manual_seed(42)
activations = [torch.randn(1, 5, 16, dtype=torch.bfloat16), *(torch.randn(1, 16, dtype=torch.bfloat16) for _ in range(10))]
proofs = ['/9kbQitnukt1bQ==', '/9nLuJrxhiVUoA==', '/9m0aNrm5KtBtg==', '/9mAZNZYMOKvcw==', '/9nSBGTqkZH21w==']
# apply some jitter to the activations
activations = [i * 1.01 for i in activations]

results = verify_proofs_base64(activations, proofs, decode_batching_size=3, topk=4, skip_prefill=False)

print("Results:")
print(*results, sep="\n")
```
```python
Results:
VerificationResult(exp_intersections=4, mant_err_mean=1.75, mant_err_median=2.0)
VerificationResult(exp_intersections=4, mant_err_mean=2, mant_err_median=2.0)
VerificationResult(exp_intersections=4, mant_err_mean=1.25, mant_err_median=1.0)
VerificationResult(exp_intersections=4, mant_err_mean=1, mant_err_median=1.0)
VerificationResult(exp_intersections=4, mant_err_mean=2, mant_err_median=2.0)
```

# Citing

```bibtex
@misc{ong2025toploclocalitysensitivehashing,
      title={TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference}, 
      author={Jack Min Ong and Matthew Di Ferrante and Aaron Pazdera and Ryan Garner and Sami Jaghouar and Manveer Basra and Johannes Hagemann},
      year={2025},
      eprint={2501.16007},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2501.16007}, 
}
```
