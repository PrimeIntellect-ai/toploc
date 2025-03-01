# TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference

[TOPLOC](https://arxiv.org/abs/2501.16007) leverages locality sensitive hashing of intermediate activations to verify that LLM providers are using authorized model configurations and settings.

The feature set includes:
- Detect unauthorized modifications to models, prompts, and precision settings
- 1000x reduction in storage requirements compared to full activation storage
- Validation speeds up to 100x faster than original inference
- Robust across different hardware configurations and implementations

For code used by experiments in our paper, check out: https://github.com/PrimeIntellect-ai/toploc-experiments

### Installation

```bash
pip install -U toploc
```

### Usage

#### Build proofs from activations:
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

#### Verify proofs:
```python
import torch
from toploc import ProofPoly
from toploc.poly import batch_activations
from toploc.C.csrc.utils import get_fp_parts
from statistics import mean, median

torch.manual_seed(42)
activations = [torch.randn(1, 5, 16, dtype=torch.bfloat16), *(torch.randn(1, 16, dtype=torch.bfloat16) for _ in range(10))]
proofs = ['/9kbQitnukt1bQ==', '/9nLuJrxhiVUoA==', '/9m0aNrm5KtBtg==', '/9mAZNZYMOKvcw==', '/9nSBGTqkZH21w==']
proofs = [ProofPoly.from_base64(proof) for proof in proofs]

# apply some jitter to the activations
activations = [i * 1.01 for i in activations]

for index, (proof, chunk) in enumerate(zip(proofs, batch_activations(activations, decode_batching_size=3))):
    chunk = chunk.view(-1).cpu()
    topk_indices = chunk.abs().topk(k=4).indices.tolist()
    topk_values = chunk[topk_indices]
    proof_topk_values = torch.tensor([proof(i) for i in topk_indices], dtype=torch.uint16).view(dtype=torch.bfloat16)
    exps, mants = get_fp_parts(proof_topk_values)
    proof_exps, proof_mants = get_fp_parts(topk_values)

    exp_intersections = [i == j for i, j in zip(exps, proof_exps)]
    mant_errs = [abs(i - j) for i, j, k in zip(mants, proof_mants, exp_intersections) if k]
    print(f"=== Proof {index}")
    print(f"Exp intersections: {sum(exp_intersections)}")
    print(f"Mean mantissa error: {mean(mant_errs)}")
    print(f"Median mantissa error: {median(mant_errs)}")
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
