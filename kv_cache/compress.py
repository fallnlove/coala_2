import argparse
import torch
import sys
from loguru import logger
from utils import set_seed, dump_to_huggingface_repos, load_model_and_tokenizer
from palu.rank_search import rank_search
from tqdm import tqdm
from palu.decomposition import compress_model
try:
    from run_lm_eval import run_lm_eval_zero_shot  # unused here; optional (pulls in lm_eval)
except Exception:
    run_lm_eval_zero_shot = None
import os

def compress(args):
    # set seed
    set_seed(args.seed)
    # load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    model.to(torch.device(args.device))
    # Step 1: Perform rank selection to get layer-wise compression rate
    search_results, rank_sum, total_rank = rank_search(model, tokenizer, args)
    # Step 2: Compress models
    compress_model(model, tokenizer, args, args.device, search_results)
    
    if args.dump_huggingface_model:
        save_folder = f"{args.model_id.split('/')[-1]}_ratio-{args.param_ratio_target}_gs-{args.head_group_size}-{args.search_method}-{args.decompose_method}"
        dump_to_huggingface_repos(model, tokenizer, save_folder, args)
        logger.info(f"Huggingface model is saved to {save_folder}", fg="green")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Pretrained model ID"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random Seed"
    )

    parser.add_argument(
        "--dump_huggingface_model", 
        action="store_true",
        help="Whether to dump huggingface model or not."
    )

    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Whether to use cached calibration results or not.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    parser.add_argument(
        "--n_fisher_calib_samples",
        type=int,
        default=32,
        help="Number of samples used for calibration.",
    )
    
    parser.add_argument(
        "--n_whiten_calib_samples",
        type=int,
        default=256,
        help="Number of samples used for calibration.",
    )

    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="Calibration dataset",
    )

    parser.add_argument(
        "--calib_seqlen",
        type=int,
        default=1024,
        help="Sequence length of the calibration dataset."
    )

    parser.add_argument(
        "--head_group_size",
        type=int,
        default=4,
        help="Group size for group-wise decomposition."
    )


    # Rank Search hyper-paramters
    parser.add_argument(
        "--param_ratio_target", 
        type=float,
        default=-1,
        help="Target param ratio"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose information or not."
    )
    
    parser.add_argument(
        "--search_method",
        type=str,
        default="fisher_uniform",
        choices=["fisher", "fisher_uniform", "uniform", "precomputed_ranks"],
        help="Search method",
    )

    parser.add_argument(
        "--ranks_file",
        type=str,
        default=None,
        help="Path to a JSON file with per-module ranks (coala2 rank-training checkpoint format). "
             "Required with --search_method=precomputed_ranks."
    )

    parser.add_argument(
        "--round_ranks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Round per-group ranks to multiples of 32 (kernel-friendly sizes). Default: "
             "fisher/fisher_uniform/uniform round, precomputed_ranks uses exact file ranks. "
             "Pass --round_ranks / --no-round_ranks to force either behavior for any method."
    )

    parser.add_argument(
        '--decompose_method',
        type=str,
        default='whiten',
        choices=['whiten', 'svd', 'coala'],
        help='Decomposition method'
    )

    parser.add_argument(
        "--coala_mu",
        type=float,
        default=1.0,
        help="COALA regularization strength (only used with --decompose_method=coala)."
    )

    parser.add_argument(
        "--coala_adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale COALA mu by the per-group approximation error (disable with --no-coala_adaptive)."
    )

    parser.add_argument(
        "--coala_balanced_split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rescale COALA factors as L=U*sqrt(S), R=(U^T W)/sqrt(S) so the latent per-dim "
             "scale matches the whiten split (product unchanged; matters for latent quantization). "
             "Disable with --no-coala_balanced_split."
    )

    parser.add_argument(
        "--coala_accumulate",
        type=int,
        default=8,
        help="TSQR streaming compaction factor for COALA activation collection: compact a module's "
             "buffered activations once they exceed accumulate * in_features columns (0 disables)."
    )

    args = parser.parse_args()

    if args.search_method == "precomputed_ranks" and args.ranks_file is None:
        parser.error("--ranks_file is required with --search_method=precomputed_ranks")
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    
    compress(args)