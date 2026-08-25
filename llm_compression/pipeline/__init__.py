"""Unified LLM compression pipeline.

A single YAML-driven entry point that reuses COALA's compression solvers and
Dobi-SVD's perplexity/data routines to compress a HuggingFace model and evaluate
it on common-sense-reasoning accuracy and perplexity.
"""
