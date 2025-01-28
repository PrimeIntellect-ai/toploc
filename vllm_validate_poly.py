import argparse
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
from statistics import mean, median
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from toploc.commits import ProofPoly


def parse_args():
    parser = argparse.ArgumentParser(description="Run validation on model activations and commits.")

    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save outputs and commits.")
    parser.add_argument("--decode_model_name", type=str, required=True, help="Model name used for decoding.")
    parser.add_argument("--validate_model_name", type=str, required=True, help="Model name used for validation.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--max_decode_tokens", type=int, default=512, help="Maximum number of tokens for decoding.")
    parser.add_argument("--decode_batching_size", type=int, default=32, help="Batch size for decoding.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for activations.")
    parser.add_argument("--attn", type=str, default="flash", help="Attention implementation for the model.")
    parser.add_argument("--scale_decode_mantissa", type=str, default="no", help="Scale decode mantissa.")

    return parser.parse_args()

SCALE_DECODE_MANTISSA = "no"

TMEAN = 10
TMEDIAN = 8
TEXP = 90

def check(activations: list[torch.Tensor], proof: list[str]) -> tuple[list[int], list[int], list[float], list[float]]:
    from toploc.C.utils import get_fp_parts
    topk_intersections: list[int] = []
    exp_intersections: list[int] = []
    mant_err_means: list[float] = []
    mant_err_medians: list[float] = []

    for act, b_poly in zip(activations, proof):
        flat_view = act.view(-1)
        prefill_topk_indices = flat_view.abs().topk(128).indices.tolist()
        prefill_topk_values = flat_view[prefill_topk_indices]
        
        poly = ProofPoly.from_bytes(b_poly)
        decode_topk_values = torch.tensor([poly(i) for i in prefill_topk_indices], dtype=torch.uint16).view(dtype=torch.bfloat16)
        decode_topk_indices = prefill_topk_indices

        prefill_exp, prefill_mants = get_fp_parts(prefill_topk_values)
        decode_exp, decode_mants = get_fp_parts(decode_topk_values)
        dec_i_2_topk_i = {index: i for i, index in enumerate(decode_topk_indices)}
        if SCALE_DECODE_MANTISSA == "down":
            decode_mants = [i // (2 ** 16) for i in decode_mants]
        elif SCALE_DECODE_MANTISSA == "up":
            decode_mants = [i * (2 ** 16) for i in decode_mants]

        topk_intersection = 0
        exp_intersection = 0
        mant_errs: list[float] = []

        for i, index in enumerate(prefill_topk_indices):
            if index in dec_i_2_topk_i:
                topk_intersection += 1
                if decode_exp[dec_i_2_topk_i[index]] == prefill_exp[i]:
                    exp_intersection += 1
                    mant_errs.append(abs(decode_mants[dec_i_2_topk_i[index]] - prefill_mants[i]))
        topk_intersections.append(topk_intersection)
        exp_intersections.append(exp_intersection)
        if len(mant_errs) == 0:
            mant_err_means.append(128.0)
            mant_err_medians.append(128.0)
        else:
            mant_err_means.append(mean(mant_errs))
            mant_err_medians.append(median(mant_errs))
            
    if mant_err_means > TMEAN or mant_err_medians > TMEDIAN or exp_intersections < TEXP:   
        print(f"VERIFICATION FAILED: Mantissa error mean: {mant_err_means} above {TMEAN} or median: {mant_err_medians} above {TMEDIAN} or exp intersections: {exp_intersections} below {TEXP}")
    else:
        print(f"VERIFICATION PASSED: Mantissa error mean: {mant_err_means} below {TMEAN} and median: {mant_err_medians} below {TMEDIAN} and exp intersections: {exp_intersections} above {TEXP}")
    return topk_intersections, exp_intersections, mant_err_means, mant_err_medians

def segment_prefill_activations(activations: torch.Tensor, max_decode_tokens: int, decode_batching_size: int) -> list[torch.Tensor]:
    ret: list[torch.Tensor] = [activations[:, :-max_decode_tokens]]
    for i in range(activations.size(1) - max_decode_tokens, activations.size(1), decode_batching_size):
        ret.append(activations[:, i:i+decode_batching_size])
    return ret

def main(args):
    global SCALE_DECODE_MANTISSA
    SCALE_DECODE_MANTISSA = args.scale_decode_mantissa
    if args.attn != "flash":
        raise NotImplementedError("Only flash attention is supported for now.")
    save_dir = Path(args.save_dir)
    outputs_path = save_dir / f'outputs_{args.decode_model_name.replace("/", "--")}.pt'
    outputs = torch.load(outputs_path)
    
    with open(save_dir / f'poly_{args.decode_model_name.replace("/", "--")}_128.bin', 'rb') as f:
        polys = [[f.read(258) for _j in range(1 + args.max_decode_tokens // args.decode_batching_size)] for _ in range(len(outputs))]
    

    llm = LLM(
        model=args.validate_model_name,
        tensor_parallel_size=args.tp,
        max_model_len=4096,
        dtype=args.dtype,
    )
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    saved_activations = []
    def activation_saving_hook(module, input, output):
        saved_activations.append(output[0].detach().clone().cpu())
    saved_activations_handle = model.model.norm.register_forward_hook(activation_saving_hook)

    names = []
    topk_intersections = []
    exp_intersections = []
    mant_err_means = []
    mant_err_medians = []

    for i, input_ids in tqdm(enumerate(outputs), total=len(outputs)):
        tokens_prompt = TokensPrompt(prompt_token_ids=input_ids[0][:-1].tolist())
        _ = llm.generate(tokens_prompt, SamplingParams(temperature=0.8, top_p=0.95, ignore_eos=True, max_tokens=1))
        activations = segment_prefill_activations(
            saved_activations[0].unsqueeze(0), args.max_decode_tokens, args.decode_batching_size
        )
        
        topk_res, exp_res, mant_means, mant_medians = check(activations, polys[i])
        #print(f"Topk: {topk_res}, Exp: {exp_res}, Mant Mean: {mant_means}, Mant Median: {mant_medians}")

        names.extend([f"Q{i}_{j}" for j in range(len(topk_res))])
        topk_intersections.extend(topk_res)
        exp_intersections.extend(exp_res)
        mant_err_means.extend(mant_means)
        mant_err_medians.extend(mant_medians)
        saved_activations = []

    df = pd.DataFrame({
        'Name': names,
        'Topk Intersections': topk_intersections,
        'Exp Intersections': exp_intersections,
        'Mant Err Means': mant_err_means,
        'Mant Err Medians': mant_err_medians
    })

    output_file = save_dir / f'poly_validation_{args.validate_model_name.replace("/", "--")}_{args.attn}_{args.dtype}_{args.tp}A100_on_{args.decode_model_name.replace("/", "--")}.parquet'
    df.to_parquet(output_file, index=False)
    print(df)

    del llm

if __name__ == "__main__":
    args = parse_args()
    main(args)
