from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3Config, Gemma3ForCausalLM

from .config import ModelConfig, TrainingConfig


def get_dtype(training_cfg: TrainingConfig) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(training_cfg.dtype.lower(), torch.bfloat16)


def load_tokenizer(model_cfg: ModelConfig) -> AutoTokenizer:
    name_or_path = model_cfg.tokenizer_name or model_cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = model_cfg.max_position_embeddings
    return tokenizer


def build_model(model_cfg: ModelConfig, training_cfg: TrainingConfig, tokenizer) -> AutoModelForCausalLM:
    torch_dtype = get_dtype(training_cfg)
    if model_cfg.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2" if training_cfg.use_flash_attention else "eager",
        )
    else:
        config = Gemma3Config(
            vocab_size=model_cfg.vocab_size or len(tokenizer),
            hidden_size=model_cfg.hidden_size,
            num_hidden_layers=model_cfg.num_hidden_layers,
            num_attention_heads=model_cfg.num_attention_heads,
            num_key_value_heads=model_cfg.num_key_value_heads,
            intermediate_size=model_cfg.intermediate_size
            or int(model_cfg.hidden_size * 4),
            rms_norm_eps=model_cfg.rms_norm_eps,
            rope_theta=model_cfg.rope_theta,
            rope_traditional=model_cfg.rope_traditional,
            tie_word_embeddings=model_cfg.tie_word_embeddings,
            attention_dropout=model_cfg.attention_dropout,
            hidden_dropout=model_cfg.hidden_dropout,
            max_position_embeddings=model_cfg.max_position_embeddings,
            _attn_implementation="flash_attention_2" if training_cfg.use_flash_attention else "eager",
        )
        model = Gemma3ForCausalLM(config)
    return model


def build_chunk_causal_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Return a 4D additive mask (batch, 1, seq, seq) that blocks attention across unrelated chunks."""
    batch, seq_len = doc_ids.shape
    device = doc_ids.device
    causal = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
    doc_mismatch = doc_ids[:, None, :, None] != doc_ids[:, None, None, :]
    full_mask = causal[None, :, :, :] | doc_mismatch
    mask = torch.zeros((batch, seq_len, seq_len), device=device, dtype=torch.float32)
    mask = mask.masked_fill(full_mask, float("-inf"))
    return mask.unsqueeze(1)


def save_artifacts(output_dir: Path, model, tokenizer, run):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    run.save(str(output_dir / "config.json"), policy="now")
    run.save(str(output_dir / "pytorch_model.bin"), policy="now")
