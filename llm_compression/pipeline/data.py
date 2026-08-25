"""Dataset loading and tokenization for calibration and perplexity.

Adapted (reference-only) from Dobi-SVD/utils/datautils.py. We keep the random
fixed-length window sampling but drop Dobi's ``.pt`` caching/SAVE/RECREATE
machinery; the HuggingFace ``datasets`` cache already memoizes raw downloads.

A "window" is a 1-D ``input_ids`` tensor of length ``seq_len``. Calibration
batches stack windows into 2-D ``[batch, seq_len]`` tensors (required by COALA's
forward pre-hook, which assumes ``[batch, seq, features]`` activations). The
perplexity helper stacks all windows into a single ``[num_samples, seq_len]``
tensor matching Dobi's ``evaluate_perplexity`` input.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

# Field holding the text body per dataset.
_TEXT_FIELD = {"wikitext2": "text", "ptb": "sentence"}

# Separator used to join documents/sentences for the contiguous protocol. PTB is
# joined with a single space (GPTQ/SVD-LLM convention); "\n\n" between PTB's very
# short sentences would add newline tokens the model never saw and inflate PPL.
_JOIN_SEP = {"wikitext2": "\n\n", "ptb": " "}

# Local PTB file per split (downloaded via jobs/download_ptb.sh).
_PTB_FILE = {
    "train": "ptb.train.txt",
    "validation": "ptb.valid.txt",
    "test": "ptb.test.txt",
}


def load_text_dataset(name: str, split: str, dataset_cache_dir: Path):
    """Load a raw text dataset split.

    * wikitext2 / ptb: auto-downloaded from the HuggingFace hub.
    * c4: read from pre-downloaded gz shards under ``<cache>/en/`` (same layout
      Dobi-SVD expects), since the full allenai/c4 config is huge.
    """
    cache = str(dataset_cache_dir)
    if name == "wikitext2":
        cfg = "wikitext-2-raw-v1"
        hf_split = "train" if split == "train" else "test"
        return load_dataset("wikitext", cfg, split=hf_split, cache_dir=cache)
    if name == "ptb":
        # Prefer local raw files under <cache>/ptb/ (downloaded via
        # jobs/download_ptb.sh).
        ptb_dir = dataset_cache_dir / "ptb"
        hf_split = split if split in _PTB_FILE else "validation"
        fpath = ptb_dir / _PTB_FILE[hf_split]
        if fpath.exists():
            lines = fpath.read_text(encoding="utf-8").splitlines()
            return Dataset.from_dict({"sentence": lines})
        # Fallback: the script-based HF dataset (needs internet to github).
        return load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split=hf_split,
            cache_dir=cache,
            trust_remote_code=True,
        )
    if name == "c4":
        shard = (
            "en/c4-train.00000-of-01024.json.gz"
            if split == "train"
            else "en/c4-validation.00000-of-00008.json.gz"
        )
        data_file = dataset_cache_dir / shard
        if not data_file.exists():
            raise FileNotFoundError(
                f"c4 shard not found at {data_file}. The allenai/c4 'en' json.gz "
                "shards are not auto-downloaded (the full config is huge). Run "
                "`bash jobs/download_c4.sh` (or see the Datasets section of README.md) "
                f"to fetch them into {dataset_cache_dir}/en/."
            )
        key = "train" if split == "train" else "validation"
        return load_dataset(
            "json", data_files={key: str(data_file)}, cache_dir=cache, split=key
        )
    raise ValueError(f"Unknown dataset {name!r}; expected wikitext2, ptb, or c4")


def _sample_window_from_doc(tokenizer, text: str, seq_len: int, rng: random.Random):
    """Tokenize a single document and return a random ``seq_len`` window, or None."""
    enc = tokenizer(text, return_tensors="pt")
    total = enc.input_ids.shape[1]
    if total < seq_len:
        return None
    start = 0 if total - seq_len - 1 < 0 else rng.randint(0, total - seq_len - 1)
    return enc.input_ids[:, start : start + seq_len].squeeze(0)


def make_token_windows(
    tokenizer,
    name: str,
    split: str,
    seq_len: int,
    num_samples: int,
    seed: int,
    dataset_cache_dir: Path,
) -> List[torch.Tensor]:
    """Produce ``num_samples`` random ``seq_len``-token windows from a dataset."""
    dataset_cache_dir = Path(dataset_cache_dir)
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    data = load_text_dataset(name, split, dataset_cache_dir)
    rng = random.Random(seed)

    windows: List[torch.Tensor] = []
    desc = f"tokenizing {name}:{split}"

    if name == "c4":
        # One window per randomly chosen document (re-draw until long enough).
        n_docs = len(data)
        with tqdm(total=num_samples, desc=desc, unit=" sample") as bar:
            while len(windows) < num_samples:
                doc = data[rng.randint(0, n_docs - 1)]["text"]
                window = _sample_window_from_doc(tokenizer, doc, seq_len, rng)
                if window is not None:
                    windows.append(window)
                    bar.update(1)
    else:
        # Concatenate the corpus, then sample windows from random char offsets.
        field = _TEXT_FIELD[name]
        tot_text = "\n\n".join(data[field])
        with tqdm(total=num_samples, desc=desc, unit=" sample") as bar:
            while len(windows) < num_samples:
                i = rng.randint(0, len(tot_text) - seq_len - 1)
                chunk = tot_text[i : i + seq_len * 10]
                window = _sample_window_from_doc(tokenizer, chunk, seq_len, rng)
                if window is not None:
                    windows.append(window)
                    bar.update(1)

    return windows


def calibration_batches(
    windows: List[torch.Tensor], batch_size: int
) -> Iterator[torch.Tensor]:
    """Yield ``[batch, seq_len]`` 2-D input_id tensors from 1-D windows."""
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        yield torch.stack(chunk, dim=0)


def ppl_input_ids(windows: List[torch.Tensor]) -> torch.Tensor:
    """Stack windows into a single ``[num_samples, seq_len]`` tensor."""
    return torch.stack(windows, dim=0)


def contiguous_chunks(
    tokenizer,
    name: str,
    split: str,
    seq_len: int,
    dataset_cache_dir: Path,
    max_chunks: int | None = None,
) -> torch.Tensor:
    """Standard whole-corpus perplexity layout: ``[n_chunks, seq_len]``.

    Tokenizes the entire split once and splits it into consecutive,
    non-overlapping ``seq_len`` chunks covering every token exactly once
    (GPTQ/SVD-LLM protocol). This is the apples-to-apples way to match published
    perplexities — used for small corpora like PTB. Large corpora (wikitext2/c4)
    would be too slow this way; they use the sampled-window path instead.
    """
    dataset_cache_dir = Path(dataset_cache_dir)
    data = load_text_dataset(name, split, dataset_cache_dir)
    field = _TEXT_FIELD.get(name, "text")
    sep = _JOIN_SEP.get(name, "\n\n")
    text = sep.join(data[field])

    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_chunks = ids.numel() // seq_len
    if n_chunks == 0:
        raise ValueError(
            f"{name}:{split} has only {ids.numel()} tokens (< seq_len={seq_len}); "
            "cannot form a contiguous chunk."
        )
    if max_chunks is not None:
        n_chunks = min(n_chunks, int(max_chunks))
    return ids[: n_chunks * seq_len].reshape(n_chunks, seq_len)
