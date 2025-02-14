# TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference

[TOPLOC](https://arxiv.org/abs/2501.16007) is a novel method for verifiable inference that enables users to verify that LLM providers are using the correct model configurations and settings. It leverages locality sensitive hashing for intermediate activations to detect unauthorized modifications.

## Features

- Detect unauthorized modifications to models, prompts, and precision settings
- 1000x reduction in storage requirements compared to full activation storage
- Validation speeds up to 100x faster than original inference
- Robust across different hardware configurations and implementations
- Zero false positives/negatives in empirical testing

## Key Components

### Proof Generation
- Extracts top-k values from the last hidden state
- Uses polynomial encoding for compact storage
- Generates verifiable proof during inference

### Validation
- Recalculates top-k features
- Compares exponent and mantissa differences
- Validates against predefined error thresholds

### Storage Requirements
- Only 258 bytes per 32 new tokens
- Compared to 262KB for full token embeddings (Llama-3.1-8B-Instruct)

## Integrations

### vLLM
TOPLOC is integrated with vLLM for efficient inference and validation as part of this repository. The integration allows you to leverage vLLM's optimized inference pipeline while maintaining verification capabilities.

### SGLang
We maintain a [fork of SGLang](https://github.com/PrimeIntellect-ai/sglang) that includes TOPLOC integration, enabling verifiable inference with SGLang's framework.

## How to use the code

### Installation

```bash
pip install -U toploc
```

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

### Run Experiments

This is an example on running validation with Llama-3.1-8B-Instruct over the ultrachat dataset.

First, generate the polynomial encodings for the model using:

```bash
python vllm_generate_poly.py --model_name meta-llama/Llama-3.1-8B-Instruct --tp 1 --n_samples 4 --save_dir signatures --max_decode_tokens 512 --dataset_name stingning/ultrachat --dtype bfloat16
```

This should create a directory called `signatures` with the polynomial encodings for the model.

You can then run validation with:

```bash
python vllm_validate_poly.py --decode_model_name meta-llama/Llama-3.1-8B-Instruct --validate_model_name meta-llama/Llama-3.1-8B-Instruct --tp 1 --n_samples 4 --save_dir just4 --max_decode_tokens 512 --dataset_name stingning/ultrachat --dtype bfloat16 --attn flash
```

If the verification passes, you should see:

```
VERIFICATION PASSED: Mantissa error mean: 0. below 10 and median: 0. below 8 and exp intersections: 100 below 90
```

And if it fails, you should see something like:

```
VERIFICATION FAILED: Mantissa error mean: 11.000000 above 10 or median: 10.000000 above 8 or exp intersections: 0 above 90  
```

## Citing

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
