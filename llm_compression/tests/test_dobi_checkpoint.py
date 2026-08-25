"""Dobi rank-checkpoint save/load round-trip (pure, CPU)."""

import json

from omegaconf import OmegaConf

import pipeline.dobi as dobi


def _cfg(model_id="meta-llama/Llama-2-7b-hf", ratio=0.8, seq_len=2048):
    return OmegaConf.create(
        {
            "model": {"id": model_id},
            "compress": {
                "ratio": ratio,
                "dobi": {"rank_selection": {"seq_len": seq_len}},
            },
        }
    )


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "ranks.json"
    ranks = {
        "model.layers.0.self_attn.q_proj": 12.5,
        "model.layers.0.mlp.up_proj": 30.0,
    }
    cfg = _cfg()

    dobi._save_rank_checkpoint(str(path), ranks, cfg)
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["model"] == cfg.model.id
    assert payload["ratio"] == 0.8
    assert payload["seq_len"] == 2048
    assert payload["ranks"] == ranks

    loaded = dobi._load_rank_checkpoint(str(path), cfg)
    assert loaded == ranks


def test_checkpoint_loads_legacy_bare_dict(tmp_path):
    path = tmp_path / "legacy.json"
    ranks = {"a.q_proj": 5.0}
    path.write_text(json.dumps(ranks))  # bare {name: gamma}, no metadata wrapper
    assert dobi._load_rank_checkpoint(str(path), _cfg()) == ranks


def test_checkpoint_model_mismatch_still_loads_with_warning(tmp_path, capsys):
    path = tmp_path / "ranks.json"
    dobi._save_rank_checkpoint(str(path), {"x": 1.0}, _cfg(model_id="modelA"))
    loaded = dobi._load_rank_checkpoint(str(path), _cfg(model_id="modelB"))
    assert loaded == {"x": 1.0}
    assert "WARNING" in capsys.readouterr().out


def test_checkpoint_path_helper():
    cfg = OmegaConf.create(
        {"compress": {"dobi": {"rank_selection": {"checkpoint": "/tmp/r.json"}}}}
    )
    assert dobi._rank_checkpoint_path(cfg) == "/tmp/r.json"
