# TOPLOC: Trustless Verifiable Inference for Large Language Models

[TOPLOC](https://arxiv.org/abs/2501.16007) is a novel method for verifiable inference that enables users to verify that LLM providers are using the correct model configurations and settings. It leverages locality sensitive hashing for intermediate activations to detect unauthorized modifications with 100% accuracy.

## Features

- Detect unauthorized modifications to models, prompts, and precision settings
- 1000x reduction in storage requirements compared to full activation storage
- Validation speeds up to 100x faster than original inference
- Robust across different hardware configurations and implementations
- Zero false positives/negatives in empirical testing

## integration

toploc is integrated with popular inference engine:

- Integrated with [vLLM](https://github.com/vllm-project/vllm) for efficient inference
- SGLang support via our [fork](https://github.com/PrimeIntellect-ai/sglang)

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
TOPLOC is integrated with vLLM for efficient inference and validation. The integration allows you to leverage vLLM's optimized inference pipeline while maintaining verification capabilities.

### SGLang
We maintain a [fork of SGLang](https://github.com/PrimeIntellect-ai/sglang) that includes TOPLOC integration, enabling verifiable inference with SGLang's programming model.

## How to use the code

### Installation

```bash
git clone https://github.com/PrimeIntellect/toploc.git
pip install -r requirements.txt
```

### run Experiments

This is an example on running validation with Llama-3.1-8B-Instruct over the ultrachat dataset.

First, generate the polynomial encodings for the model using:

```bash
python vllm_generate_poly.py --model_name meta-llama/Llama-3.1-8B-Instruct --tp 1 --n_samples 4 --save_dir just4 --max_decode_tokens 512 --dataset_name stingning/ultrachat --dtype bfloat16
```

this should create a directory called `just4` with the polynomial encodings for the model.

You can then run validation with:

```bash
python vllm_validate_poly.py --decode_model_name meta-llama/Llama-3.1-8B-Instruct --validate_model_name meta-llama/Llama-3.1-8B-Instruct --tp 1 --n_samples 4 --save_dir just4 --max_decode_tokens 512 --dataset_name stingning/ultrachat --dtype bfloat16 --attn flash
```

if the verification passes, you should see :

```
VERIFICATION PASSED: Mantissa error mean: 0. below 10 and median: 0. below 8 and exp intersections: 100 below 90
```

and if it fails, you should see something like:

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
