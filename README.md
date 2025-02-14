# TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference

[TOPLOC](https://arxiv.org/abs/2501.16007) is a novel method for verifiable inference that enables users to verify that LLM providers are using the correct model configurations and settings. It leverages locality sensitive hashing for intermediate activations to detect unauthorized modifications.

For code used in our experiments, check out: https://github.com/PrimeIntellect-ai/toploc-experiments

### Installation

```bash
pip install -U toploc
```

### Features

- Detect unauthorized modifications to models, prompts, and precision settings
- 1000x reduction in storage requirements compared to full activation storage
- Validation speeds up to 100x faster than original inference
- Robust across different hardware configurations and implementations
- Zero false positives/negatives in empirical testing

### Development Setup

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

2. Setup virtual environment:
This can take awhile because of the C extensions.
```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --dev
```

### Running Tests

Run the test suite:
```bash
uv run pytest tests
```

Run single file:
```bash
uv run pytest tests/test_utils.py
```

Run single test:
```bash
uv run pytest tests/test_utils.py::test_get_fp32_parts
```

### Code Quality

Install pre-commit hooks:
```bash
uv run pre-commit install
```

Run linting and formatting on all files:
```bash
pre-commit run --all-files
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
