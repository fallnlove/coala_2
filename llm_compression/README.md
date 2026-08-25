# Unified LLM Compression Pipeline

One YAML-driven command that downloads datasets + a HuggingFace model, optionally
compresses it (COALA low-rank solvers **or** Dobi-SVD), and evaluates it on
common-sense-reasoning accuracy (lm-eval) and perplexity. Replaces COALA's
multi-script shell pipeline with a single process.

The compression code is **vendored** under `vendor/` (COALA's `coala`/`lib`
packages and Dobi-SVD's SVD/IPCA modules, copied out of the original `COALA/` and
`Dobi-SVD/` folders), so the pipeline runs standalone — those reference folders are
not needed at runtime. Dobi's perplexity/data helpers are reimplemented in
`pipeline/`.

## Layout

```
pipeline/        # the pipeline package
  run.py         # CLI entry point
  config.py      # YAML load/validate + COALA_Config builder
  model.py       # HF model/tokenizer load, dtype cast, save
  data.py        # wikitext2/c4/ptb sampling + contiguous protocol
  compress.py    # COALA inject flow (no HF Trainer) + routes to dobi
  dobi.py        # Dobi-SVD: rank selection + IPCA compression
  evaluate.py    # perplexity (per-phase dtype) + lm-eval reasoning
vendor/          # vendored coala/, lib/ (COALA) and dobi/ (Dobi-SVD math)
configs/         # baseline, coala, svd, asvd, dobi, smoke .yaml
tests/           # CPU tests (+ one @pytest.mark.gpu smoke)
```

## Install

Local (uv, Python 3.11):

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Conda:

```bash
conda env create -f environment.yml
conda activate coala_env
```

## Run

```bash
# Baseline (no compression) — evaluate the uncompressed model
python -m pipeline.run --config configs/baseline.yaml

# Compress with COALA, override the model from the CLI
python -m pipeline.run --config configs/coala.yaml model.id=meta-llama/Llama-3.2-1B

# Any key=value args are OmegaConf dotlist overrides
python -m pipeline.run --config configs/svd.yaml compress.ratio=0.6 eval.reasoning.enabled=false
```

Results are written to the `results_path` in the config (JSON: perplexity per
dataset + lm-eval reasoning metrics).

### Strategies

| strategy    | data-aware | device   | required config              |
|-------------|------------|----------|------------------------------|
| `empty`     | no         | CPU/GPU  | —                            |
| `svd`       | no         | **GPU**  | —                            |
| `svd_llm`   | yes        | **GPU**  | `params.accumulate`          |
| `svd_llm_2` | yes        | **GPU**  | `params.accumulate`          |
| `coala`     | yes        | **GPU**  | `params.{mu,adaptive,log_norms}`|
| `asvd`      | yes        | **GPU**  | `params.alpha`               |
| `swift_svd` | yes        | **GPU**  | `params.accumulate`          |
| `dobi`      | yes        | **GPU**  | `dobi.rank_selection.*`      |

All non-`empty` strategies require CUDA. COALA solvers use the `gesvd` cuSOLVER
driver; Dobi runs SVD + IPCA on GPU. Set `compress.enabled: false` for a baseline.

Reasoning tasks: piqa, hellaswag, boolq, winogrande, arc_easy, arc_challenge,
openbookqa.

## Datasets

Perplexity / calibration support three corpora (`eval.perplexity.datasets` and
`calibration.dataset`):

| dataset     | source                                 | download |
|-------------|----------------------------------------|----------|
| `wikitext2` | `wikitext` / `wikitext-2-raw-v1` on HF | automatic |
| `ptb`       | Penn Treebank raw `.txt` files         | **manual** (see below) |
| `c4`        | `allenai/c4` `en/*.json.gz` shards     | **manual** (see below) |

Fetch datasets once:

```bash
bash jobs/download_ptb.sh                 # -> ./datasets/ptb/  (~5 MB)
bash jobs/download_c4.sh                  # -> ./datasets/en/
# or with an explicit cache dir:
bash jobs/download_ptb.sh /path/to/cache  # -> /path/to/cache/ptb/
bash jobs/download_c4.sh  /path/to/cache  # -> /path/to/cache/en/
```

PTB downloads `ptb.{train,valid,test}.txt`. c4 downloads
`c4-validation.00000-of-00008.json.gz` (~40 MB, needed for c4 perplexity) and
`c4-train.00000-of-01024.json.gz` (~320 MB, only needed if you use c4 for
*calibration*).

Evaluate on all three at once, e.g.:

```bash
python -m pipeline.run --config configs/baseline.yaml \
  eval.perplexity.datasets='[wikitext2, ptb, c4]'
```

## Test

```bash
pytest -m "not gpu"      # CPU suite (config/data/baseline/empty-compress)
pytest -m gpu            # full calibration->inject->eval path (GPU)
```
