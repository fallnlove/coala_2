from loguru import logger
import torch.nn as nn
import torch
import os
import click
from tqdm import tqdm
from .data_utils import get_calib_data
from .model import HeadwiseLowRankModule

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def get_whiten_scale_matrix(model, tokenizer, args, dev):
    model_id = model.config._name_or_path
    calib_loader = get_calib_data(
        "wikitext2", 
        tokenizer, 
        model_id, 
        nsamples=256, 
        seqlen=2048
    )
    cache_file = f"cache/whiten/{model_id.replace('/','_')}_w2_scaling_matrices_fp16.pt"
    os.makedirs("cache/whiten", exist_ok=True)
    """
    cache format:
    [
        {
            "attn.q_proj": torch.Tensor,
            "attn.k_proj": torch.Tensor,
            "attn.v_proj": torch.Tensor,
            "attn.o_proj": torch.Tensor,
            "mlp.gate_proj": torch.Tensor,
            "mlp.up_proj": torch.Tensor,
            "mlp.down_proj": torch.Tensor
        },
        ... (stacked n times, in the order of model layers)
    ]
    """
    logger.info(f"[whiten] Calibration dataset: {args.calib_dataset}", fg="yellow")
    logger.info(f"[whiten] Search cache_file={cache_file}", fg="yellow")
    if os.path.exists(cache_file) and args.use_cache:
        logger.info(f"[whiten] File {cache_file} exist.", fg="green")
        logger.info(f"[whiten] Load scaling diag matrix from cache: {cache_file}", fg="yellow")
        scaling_matrics = torch.load(cache_file, map_location="cpu")


        layers = model.model.layers
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            subset = find_layers(layer) # Collect all linear layers
            for name in subset:
                if name in scaling_matrics[i]:
                    scaling_diag_matrix = scaling_matrics[i][name]
                    subset[name].scaling_diag_matrix = scaling_diag_matrix

        return 
    
    logger.info(f"No cache_file={cache_file}", fg="red")
    logger.info(f"Create whiten scale matrix dict...", fg="yellow")

    # Create Scaling Matrix with low-resource inference
    # Adapted from https://github.com/AIoT-MLSys-Lab/SVD-LLM/blob/main/SVDLLM.py
    # Here, inference are performed in an layer-wise manner.
    use_cache = model.config.use_cache
    model.config.use_cache = False
    #FIXME: This is not a good implementation...
    if "llama" in model_id or "mistral" in model_id or "vicuna" in model_id or "longchat":
        layers = model.model.layers
    elif "opt" in model_id:
        layers = model.model.decoder.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'kwargs': None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            # Capture the per-layer kwargs once (all calib samples share seqlen/mask/positions).
            # Works across transformers versions: old passes attention_mask/position_ids,
            # newer (>=4.4x) computes rotary at the model level and passes position_embeddings.
            if cache['kwargs'] is None:
                cache['kwargs'] = {k: kwargs.get(k) for k in
                                   ('attention_mask', 'position_ids', 'position_embeddings')}
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    layer_kwargs = cache['kwargs']
    # Newer transformers compute rotary embeddings at the model level and pass them to each
    # layer as position_embeddings=(cos, sin); recompute here if they weren't captured.
    if layer_kwargs.get('position_embeddings') is None and hasattr(model.model, 'rotary_emb'):
        pos_ids = layer_kwargs.get('position_ids')
        if pos_ids is None:
            pos_ids = torch.arange(inps.shape[1], device=dev).unsqueeze(0)
        pos_ids = pos_ids.to(dev)
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
        layer_kwargs = dict(layer_kwargs)
        layer_kwargs['position_ids'] = pos_ids
        layer_kwargs['position_embeddings'] = model.model.rotary_emb(inps[0:1].to(dev), pos_ids)
    # Build the constant kwargs passed to every layer during the replay.
    replay_kwargs = {}
    if layer_kwargs.get('attention_mask') is not None:
        replay_kwargs['attention_mask'] = layer_kwargs['attention_mask'].to(dev)
    if layer_kwargs.get('position_ids') is not None:
        replay_kwargs['position_ids'] = layer_kwargs['position_ids'].to(dev)
    if layer_kwargs.get('position_embeddings') is not None:
        replay_kwargs['position_embeddings'] = tuple(t.to(dev) for t in layer_kwargs['position_embeddings'])
    scaling_matrices = []
    logger.info("[Decomposition] Start to calculate the scaling matrix in layer-wise manner...")
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), **replay_kwargs)[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        layer_scaling_matrices = {}
        for name in subset:
            if not ("k_proj" in name or "v_proj" in name):
                continue
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().cuda()
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                subset[name].scaling_diag_matrix = scaling_diag_matrix
            except Exception as e:
                logger.warning("eigen scaling_diag_matrix is not positive!")
                if torch.isnan(raw_scaling_diag_matrix).any():
                    logger.warning("raw scaling_diag_matrix contains NaN!")
                elif torch.isinf(raw_scaling_diag_matrix).any():
                    logger.warning("raw scaling_diag_matrix contains Inf!")
                if not torch.equal(raw_scaling_diag_matrix, raw_scaling_diag_matrix.T):
                    logger.warning("raw scaling_diag_matrix is not a symmetric matrix!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-3) * torch.eye(raw_scaling_diag_matrix.shape[0]).cuda()
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                if torch.isnan(scaling_diag_matrix).any():
                    logger.warning("scaling_diag_matrix contains NaN!")
                elif torch.isinf(scaling_diag_matrix).any():
                    logger.warning("scaling_diag_matrix contains Inf!")
                del eigenvalues
                subset[name].scaling_diag_matrix = scaling_diag_matrix
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                logger.warning("scaling_diag_matrix is not full rank!")
                reg_inv =  1e-3 * torch.eye(scaling_diag_matrix.shape[0]).cuda() 
                scaling_diag_matrix += reg_inv
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                del reg_inv
            
            del scaling_matrix_inv
            layer_scaling_matrices[name] = scaling_diag_matrix.cpu()
            torch.cuda.empty_cache()
        scaling_matrices.append(layer_scaling_matrices)
        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()
        
    model.config.use_cache = use_cache
    if args.use_cache:
        torch.save(scaling_matrices, cache_file)
        logger.info(f"Save the whiten scale matrix dict to:  {cache_file}")

@torch.no_grad()
def get_coala_activation_matrix(model, tokenizer, args, dev):
    model_id = model.config._name_or_path
    calib_loader = get_calib_data(
        args.calib_dataset,
        tokenizer,
        model_id,
        nsamples=args.n_whiten_calib_samples,
        seqlen=args.calib_seqlen
    )
    cache_file = (
        f"cache/coala/{model_id.replace('/','_')}_{args.calib_dataset}"
        f"_n{args.n_whiten_calib_samples}_s{args.calib_seqlen}_activations.pt"
    )
    os.makedirs("cache/coala", exist_ok=True)

    target_linears = {
        name: module for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and ("k_proj" in name or "v_proj" in name)
    }

    logger.info(f"[coala] Calibration dataset: {args.calib_dataset}", fg="yellow")
    logger.info(f"[coala] Search cache_file={cache_file}", fg="yellow")
    if os.path.exists(cache_file) and args.use_cache:
        logger.info(f"[coala] File {cache_file} exist.", fg="green")
        logger.info(f"[coala] Load activation matrices from cache: {cache_file}", fg="yellow")
        activation_matrices = torch.load(cache_file, map_location="cpu")
        for name, module in target_linears.items():
            module.coala_activation_matrix = activation_matrices[name]
        return

    logger.info(f"No cache_file={cache_file}", fg="red")
    logger.info(f"[coala] Collect calibration activation matrices (TSQR compaction factor: {args.coala_accumulate})...", fg="yellow")

    # Adapted from coala2 vendor/coala/Injector.py (prepare_get_samples/_calculate/compactify):
    # keep the raw activation matrix X = [in_features, tokens] per module on CPU, replacing it
    # with the R^T factor of a QR whenever it grows past `accumulate * in_features` columns.
    # R^T has the same X X^T, so the COALA solve is unchanged while memory stays bounded.
    buffers = {name: [] for name in target_linears}

    def compactify(arr, condition):
        m = arr[0].shape[0]
        n = sum(t.shape[1] for t in arr)
        if condition(m, n):
            Y = torch.cat(arr, dim=1)
            _, R = torch.linalg.qr(Y.T.to(dev), mode='r')
            return [R.T.cpu()]
        return arr

    def make_hook(name):
        def hook(module, input):
            inp = input[0].detach()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            X = inp.float().flatten(0, 1).T.cpu()  # [in_features, batch*seq]
            buffers[name].append(X)
            if args.coala_accumulate > 0:
                buffers[name] = compactify(buffers[name], lambda m, n: n > args.coala_accumulate * m)
        return hook

    handles = [
        module.register_forward_pre_hook(make_hook(name))
        for name, module in target_linears.items()
    ]
    use_cache_config = model.config.use_cache
    model.config.use_cache = False
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    model.config.use_cache = use_cache_config
    for h in handles:
        h.remove()

    activation_matrices = {}
    for name, module in target_linears.items():
        X = torch.cat(compactify(buffers[name], lambda m, n: n > m), dim=1)
        module.coala_activation_matrix = X
        activation_matrices[name] = X
        buffers[name] = []

    if args.use_cache:
        torch.save(activation_matrices, cache_file)
        logger.info(f"Save the coala activation matrices to:  {cache_file}")

def compress_model_whiten(model, tokenizer, args, dev, selection_result):
    logger.info("Compressing model with whiten decomposition...")
    get_whiten_scale_matrix(model, tokenizer, args, dev)
    # Compress the model
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    logger.info(f"Start decompose the layer with selected ranks... #target layers: {len(selection_result.keys())}")
    for layername, selected_head_rank in tqdm(selection_result.items()):
        logger.debug(f"Decompose {layername} with ranks: {selected_head_rank}")
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
    
        head_wise_svd_linear = HeadwiseLowRankModule.from_linear_whiten(
            raw_linear,
            selected_head_rank
        )
        setattr(info["father"], info["name"],  head_wise_svd_linear)

def compress_model_coala(model, tokenizer, args, dev, selection_result):
    logger.info(f"Compressing model with COALA decomposition (mu={args.coala_mu}, adaptive={args.coala_adaptive}, balanced={args.coala_balanced_split})...")
    # NOTE: Collect per-module calibration activation matrices (coala2-style)
    get_coala_activation_matrix(model, tokenizer, args, dev)
    # Compress the model
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    logger.info(f"Start decompose the layer with selected ranks... #target layers: {len(selection_result.keys())}")
    for layername, selected_head_rank in tqdm(selection_result.items()):
        logger.debug(f"Decompose {layername} with ranks: {selected_head_rank}")
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]

        head_wise_svd_linear = HeadwiseLowRankModule.from_linear_coala(
            raw_linear,
            selected_head_rank,
            mu=args.coala_mu,
            adaptive=args.coala_adaptive,
            balanced=args.coala_balanced_split
        )
        setattr(info["father"], info["name"],  head_wise_svd_linear)

def compress_model_svd(model, selection_result):
    logger.info("Compressing model with svd decomposition...")
    # Compress the model
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    logger.info(f"Start decompose the layer with selected ranks... #target layers: {len(selection_result.keys())}")
    for layername, selected_head_rank in tqdm(selection_result.items()):
        logger.debug(f"Decompose {layername} with ranks: {selected_head_rank}")
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        print("head-wise svd", layername, raw_linear)
        head_wise_svd_linear = HeadwiseLowRankModule.from_linear(
            raw_linear,
            selected_head_rank
        )
        setattr(info["father"], info["name"],  head_wise_svd_linear)

# Wrapper for different decompose methods
def compress_model(model, tokenizer, args, dev, selection_result):
    if args.decompose_method == "whiten":
        compress_model_whiten(model, tokenizer, args, dev, selection_result)
    elif args.decompose_method == "coala":
        compress_model_coala(model, tokenizer, args, dev, selection_result)
    elif args.decompose_method == "svd":
        compress_model_svd(model, selection_result)
    else:
        raise ValueError(f"Decomposition method {args.decompose_method} is not supported.")