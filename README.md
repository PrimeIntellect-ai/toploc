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
