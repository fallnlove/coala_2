# COALA2 — Code Supplement

Anonymous code supplement. Two independent experiment trees, one per family of
results in the paper.

| Directory | What it does |
|---|---|
| [llm_compression/](llm_compression/) | **Weight compression.** One YAML-driven pipeline that loads a HuggingFace LLM, factorizes its linear layers with a chosen low-rank method, and evaluates perplexity + common-sense reasoning. Contains the COALA2 contribution: *learned non-uniform per-layer ranks*. |
| [kv_cache/](kv_cache/) | **KV-cache compression.** Palu (head-group low-rank KV projections) extended with the COALA solver as a decomposition backend and with per-layer ranks imported from `llm_compression`. |

Both trees are self-contained and have separate dependency sets, run them in
separate environments.
