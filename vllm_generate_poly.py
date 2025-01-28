from tqdm import tqdm
from toploc.commits import ProofPoly
from vllm import LLM, SamplingParams
import torch
import argparse
from datasets import load_dataset
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run activation saving and inference generation with a language model.")
    
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Name of the model to use.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples to generate.")
    parser.add_argument("--save_dir", type=str, default="just4", help="Directory to save outputs.")
    parser.add_argument("--max_decode_tokens", type=int, default=512, help="Maximum number of decode tokens.")
    parser.add_argument("--decode_batching_size", type=int, default=32, help="Batching size for decoding.")
    parser.add_argument("--dataset_name", type=str, default="stingning/ultrachat", help="Dataset to load.")
    parser.add_argument("--system_prompt", type=str, default="None", help="System prompt to prepend to each input.")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    
    return parser.parse_args()

K = 128

def build_activation_commit(activations: list[torch.Tensor], decode_batching_size: int) -> list[str]:
    commits = []

    # Prefill
    flat_view = activations[0].view(-1)
    topk_indices = flat_view.abs().topk(K).indices
    topk_values = flat_view[topk_indices]
    commit = ProofPoly.from_points(topk_indices, topk_values).to_bytes()
    commits.append(commit)

    # Batched Decode
    for i in range(1, len(activations), decode_batching_size):
        flat_view = torch.cat([i.view(-1) for i in activations[i: i + decode_batching_size]])
        topk_indices = flat_view.abs().topk(K).indices
        topk_values = flat_view[topk_indices]
        commit = ProofPoly.from_points(topk_indices, topk_values).to_bytes()
        commits.append(commit)
        
    return commits

def main(args):
    if args.system_prompt != "None":
        raise NotImplementedError("System prompts are not yet supported.")

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, ignore_eos=True, max_tokens=args.max_decode_tokens + 1)
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tp,
        max_model_len=4096,
        enforce_eager=True,
        dtype=args.dtype,
    )
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    saved_activations = []
    def activation_saving_hook(module, input, output):
        saved_activations.append(output[0].detach().clone().cpu())
    saved_activations_handle = model.model.norm.register_forward_hook(activation_saving_hook)

    ds = load_dataset(args.dataset_name, split="train")
    prompts = [i['data'][0] for _, i in zip(range(args.n_samples), ds)]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_save_path = save_dir / f"outputs_{args.model_name.replace('/', '--')}.pt"

    saved_commits = []
    outputs = []
    for prompt in tqdm(prompts):
        output = llm.generate([prompt], sampling_params)
        input_ids = output[0].prompt_token_ids
        output_ids = output[0].outputs[0].token_ids
        output = torch.tensor([[*input_ids, *output_ids]])

        outputs.append(output)

        act_commit = build_activation_commit(saved_activations, args.decode_batching_size)
        saved_commits.append(act_commit)
        saved_activations = []

    torch.save(outputs, output_save_path)
    print(f"Saved outputs to {output_save_path}")

    savepath = save_dir / f"poly_{args.model_name.replace('/', '--')}_128.bin"
    with open(savepath, "wb") as f:
        for commit in saved_commits:
            for c in commit:
                f.write(c)
    print(f"Saved to {savepath}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
