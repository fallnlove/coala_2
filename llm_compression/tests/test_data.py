"""Data windowing shapes (CPU; downloads tiny wikitext2 + tokenizer)."""

import torch
from transformers import AutoTokenizer

from pipeline import data as data_mod
from tests.conftest import TINY_MODEL

SEQ_LEN = 16
N = 3


def _tokenizer():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_windows_are_1d_seqlen(tmp_path):
    windows = data_mod.make_token_windows(
        _tokenizer(), "wikitext2", "validation", SEQ_LEN, N, 0, tmp_path / "ds"
    )
    assert len(windows) == N
    for w in windows:
        assert w.dim() == 1 and w.shape[0] == SEQ_LEN


def test_calibration_batches_are_2d(tmp_path):
    windows = data_mod.make_token_windows(
        _tokenizer(), "wikitext2", "train", SEQ_LEN, N, 0, tmp_path / "ds"
    )
    batches = list(data_mod.calibration_batches(windows, batch_size=2))
    assert batches[0].dim() == 2
    assert batches[0].shape[1] == SEQ_LEN
    # total rows across batches equals number of windows
    assert sum(b.shape[0] for b in batches) == N


def test_ppl_input_ids_shape(tmp_path):
    windows = data_mod.make_token_windows(
        _tokenizer(), "wikitext2", "validation", SEQ_LEN, N, 0, tmp_path / "ds"
    )
    ids = data_mod.ppl_input_ids(windows)
    assert ids.shape == torch.Size([N, SEQ_LEN])


def test_contiguous_chunks_shape(tmp_path):
    # Whole-corpus contiguous layout: [n_chunks, seq_len], every chunk full width.
    ids = data_mod.contiguous_chunks(
        _tokenizer(), "wikitext2", "test", SEQ_LEN, tmp_path / "ds", max_chunks=5
    )
    assert ids.dim() == 2
    assert ids.shape[1] == SEQ_LEN
    assert 1 <= ids.shape[0] <= 5
