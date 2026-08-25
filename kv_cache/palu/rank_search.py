
import json
import os, click
import torch
import torch.nn as nn
from loguru import logger
from .model import AVAILABLE_MODELS
from .data_utils import get_calib_data
import math
from tqdm import tqdm

def rounding_search_result(config: dict, block_size=32):
    for module_name in config.keys():
        ranks = config[module_name]
        for i in range(len(ranks)):
            ranks[i] = max(1, round(ranks[i] / block_size)) * block_size
        config[module_name] = ranks
    return config

def replace_with_mean(data):
    result = {}
    for key, value in data.items():
        if value:
            mean_value = sum(value) / len(value)
            new_value = [mean_value] * len(value)
            result[key] = new_value
    return result

def split_values(data, group_number):
    result = {}
    for key, value in data.items():
        new_value = [v // group_number for v in value for _ in range(group_number)]
        result[key] = new_value
    return result


def load_precomputed_ranks(ranks_file, model_id):
    with open(ranks_file) as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "ranks" in payload:
        ckpt_model = payload.get("model")
        if ckpt_model is not None and str(ckpt_model) != model_id:
            logger.warning(
                f"[Rank search] Ranks file was produced for model '{ckpt_model}' "
                f"but current model is '{model_id}'; layer names may not match."
            )
        ranks = payload["ranks"]
    else:
        # legacy bare {name: rank} dict
        ranks = payload
    return {name: int(round(float(rank))) for name, rank in ranks.items()}

def split_rank_across_groups(module_rank, num_groups, group_dim, module_name):
    base, rem = divmod(module_rank, num_groups)
    ranks = [base + 1 if i < rem else base for i in range(num_groups)]
    clamped = [min(max(r, 1), group_dim) for r in ranks]
    if sum(clamped) != module_rank:
        logger.warning(
            f"[Rank search] {module_name}: rank {module_rank} adjusted to {sum(clamped)} "
            f"after per-group clamping to [1, {group_dim}]"
        )
    return clamped


def calib_fisher_info(model, calib_loader, device, use_cache=True):
    model.half()
    model.to(device)
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_calib_fisher_info.pt"

    logger.info(f"[Fisher] Search cache_file={cache_file}", fg="yellow")

    if os.path.exists(cache_file) and use_cache:
        logger.info(f"[Fisher] File {cache_file} exist.", fg="green")
        logger.info(f"[Fisher] Load cache_file={cache_file}", fg="yellow")
        all_fisher_info = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "attn" in name:
                module.fisher_info = all_fisher_info[name].to(module.weight.device)
        return
    model.eval()

    logger.info(f"[Fisher] No cache_file={cache_file}", fg="red")
    logger.info(f"[Fisher] Create fisher info list...", fg="yellow")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "attn" in name:
            module.fisher_info = 0

    # get fisher info
    for batch in tqdm(calib_loader):
        input_ids = batch["input_ids"][:, :-1].to(model.device)
        labels = batch["input_ids"][:, 1:].to(model.device)
        out = model(input_ids=input_ids, labels=labels)
        out[0].backward()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "attn" in name:
                module.fisher_info += module.weight.grad.detach().to(torch.float32).pow(2)
        model.zero_grad()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "attn" in name:
            module.fisher_info = module.fisher_info.div(len(calib_loader)).sqrt()

    # remove and save fisher_info
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "attn" in name:
            module._forward_hooks.clear()
            all_fisher_info[name] = module.fisher_info

    logger.info(f"[Fisher] Save the fisher info list to:  {cache_file}", fg="yellow")
    torch.save(all_fisher_info, cache_file)

def rank_search(model: nn.Module, tokenizer, args):
    logger.info(f"[Rank search] Do rank searching. Search method: {args.search_method}", fg="yellow")
    if args.search_method == "uniform":
        target_model_class = AVAILABLE_MODELS[model.config.model_type]["ModelForCausalLM"]
        total_rank = 0
        select_result = {}
        info = target_model_class.get_kv_info(model, args.head_group_size)
        
        for name, module in model.named_modules():
            if "k_proj" in name or "v_proj" in name:                
                module_rank = info.num_lr_groups * info.lr_group_dims
                total_rank += module_rank
                
                select_result.update({name: [info.lr_group_dims*args.param_ratio_target] * info.num_lr_groups})

        if args.round_ranks is not False:
            select_result = rounding_search_result(select_result)
        else:
            select_result = {k: [max(1, int(round(r))) for r in v] for k, v in select_result.items()}
        rank_sum = sum([sum(v) for k, v in select_result.items()])
        logger.info(f"[Rank search] KV-Cache Compression Ratio: {100-(rank_sum / total_rank * 100): .2f}%")
        return select_result, rank_sum, total_rank
    elif args.search_method == "fisher":
        # Prepare Fisher information
        calib_loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id, 2048, seqlen=args.calib_seqlen)
        calib_fisher_info(model, calib_loader, torch.device(args.device), args.use_cache)
        
        
        target_model_class = AVAILABLE_MODELS[model.config.model_type]["ModelForCausalLM"]
        total_rank = 0
        fisher_sum = 0.0
        fisher_info_dict = {}
        select_result = {}
        
        info = target_model_class.get_kv_info(model, args.head_group_size)
        for name, module in model.named_modules():
            if "k_proj" in name or "v_proj" in name:
                module_rank = info.num_lr_groups * info.lr_group_dims
                total_rank += module_rank
                
                select_result.update({name: [info.lr_group_dims] * info.num_lr_groups})
                
                fisher = module.fisher_info.reshape(info.num_lr_groups, -1, module.in_features)
                if not torch.isfinite(fisher).all():
                    logger.info(fisher)
                
                fisher_list = [torch.mean(fisher[i]).item() for i in range(info.num_lr_groups)]
                fisher_info_dict.update({name: fisher_list})
                fisher_sum += sum(fisher_list)


        target_rank = total_rank * args.param_ratio_target
        
        indexes = []
        select_result_float = {}

        for name, fisher in fisher_info_dict.items():
            ranks = []
            for i in range(len(fisher)):
                rank_float = target_rank * fisher[i] / fisher_sum
                
                ranks.append(rank_float)
                indexes.append((name, i))
                select_result[name][i] = min(select_result[name][i], math.floor(rank_float))

            select_result_float.update({name: ranks})
                
        indexes = sorted(indexes, key=lambda x: select_result_float[x[0]][x[1]] - select_result[x[0]][x[1]])
        dif = target_rank - sum([sum(v) for k, v in select_result.items()])


        while dif > 0:
            for i in range(len(indexes)):
                if select_result[indexes[i][0]][indexes[i][1]] == info.lr_group_dims:
                    continue
                select_result[indexes[i][0]][indexes[i][1]] += 1
                dif -= 1

                if dif == 0:
                    break

        if args.round_ranks is not False:
            select_result = rounding_search_result(select_result)
        rank_sum = sum([sum(v) for k, v in select_result.items()])
        logger.info(f"[Rank Search] KV-Cache Compression Ratio: {100-(rank_sum / total_rank * 100): .2f}%")

        return select_result, rank_sum, total_rank
    elif args.search_method == "fisher_uniform":
        # Prepare Fisher information
        calib_loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id, 2048, seqlen=args.calib_seqlen)
        calib_fisher_info(model, calib_loader, torch.device(args.device), args.use_cache)
        
        target_model_class = AVAILABLE_MODELS[model.config.model_type]["ModelForCausalLM"]
            
        total_rank = 0
        
        fisher_sum = 0.0
        fisher_info_dict = {}
        select_result = {}
        info = target_model_class.get_kv_info(model, model.config.num_key_value_heads)
        
        for name, module in model.named_modules():
            if "k_proj" in name or "v_proj" in name:
                module_rank = info.num_lr_groups * info.lr_group_dims
                total_rank += module_rank
                
                select_result.update({name: [info.lr_group_dims] * info.num_lr_groups})
                fisher = module.fisher_info.reshape(info.num_lr_groups, -1, module.in_features)
                
                fisher_list = [torch.mean(fisher[i]).item() for i in range(info.num_lr_groups)]
                fisher_info_dict.update({name: fisher_list})
                fisher_sum += sum(fisher_list)


        target_rank = total_rank * args.param_ratio_target
        
        indexes = []
        select_result_float = {}

        for name, fisher in fisher_info_dict.items():
            ranks = []
            for i in range(len(fisher)):
                rank_float = target_rank * fisher[i] / fisher_sum    
                ranks.append(rank_float)
                indexes.append((name, i))
                select_result[name][i] = min(select_result[name][i], math.floor(rank_float))

            select_result_float.update({name: ranks})
            
        indexes = sorted(indexes, key=lambda x: select_result_float[x[0]][x[1]] - select_result[x[0]][x[1]])
        dif = target_rank - sum([sum(v) for k, v in select_result.items()])


        while dif > 0:
            for i in range(len(indexes)):
                if select_result[indexes[i][0]][indexes[i][1]] == info.lr_group_dims:
                    continue
                select_result[indexes[i][0]][indexes[i][1]] += 1
                dif -= 1

                if dif == 0:
                    break
        
        select_result = split_values(select_result, model.config.num_key_value_heads//args.head_group_size)
        if args.round_ranks is not False:
            select_result = rounding_search_result(select_result)
        rank_sum = sum([sum(v) for k, v in select_result.items()])
        logger.info(f"[Rank Search] KV-Cache Compression Ratio: {100-(rank_sum / total_rank * 100): .2f}%")
        
        return select_result, rank_sum, total_rank
    elif args.search_method == "precomputed_ranks":
        precomputed = load_precomputed_ranks(args.ranks_file, args.model_id)

        target_model_class = AVAILABLE_MODELS[model.config.model_type]["ModelForCausalLM"]
        total_rank = 0
        select_result = {}
        info = target_model_class.get_kv_info(model, args.head_group_size)

        missing = []
        used = 0
        for name, module in model.named_modules():
            if "k_proj" in name or "v_proj" in name:
                module_rank = info.num_lr_groups * info.lr_group_dims
                total_rank += module_rank

                if name not in precomputed:
                    missing.append(name)
                    continue
                used += 1
                select_result.update({
                    name: split_rank_across_groups(precomputed[name], info.num_lr_groups, info.lr_group_dims, name)
                })

        if missing:
            raise ValueError(
                f"Ranks file {args.ranks_file} is missing {len(missing)} k_proj/v_proj modules, "
                f"e.g. {missing[:5]}"
            )
        logger.debug(f"[Rank search] Ignored {len(precomputed) - used} non-k/v entries in {args.ranks_file}")

        if args.round_ranks:
            select_result = rounding_search_result(select_result)
            # rounding can overshoot the per-group maximum when lr_group_dims % 32 != 0
            for name, ranks in select_result.items():
                select_result[name] = [min(r, info.lr_group_dims) for r in ranks]

        rank_sum = sum([sum(v) for k, v in select_result.items()])
        logger.info(f"[Rank search] KV-Cache Compression Ratio: {100-(rank_sum / total_rank * 100): .2f}%")
        return select_result, rank_sum, total_rank
    else:
        raise NotImplementedError