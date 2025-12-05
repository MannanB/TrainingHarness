from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import get_cosine_schedule_with_warmup

from .config import MicrolmConfig
from .model import (
    build_chunk_causal_mask,
    build_model,
    get_dtype,
    load_tokenizer,
    save_artifacts,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PackedIterableDataset(IterableDataset):
    def __init__(
        self,
        data_dir: Path,
        file_pattern: str,
        tokenizer,
        seq_len: int,
        text_key: str,
        stop_sequences: List[str],
        repeat: bool = True,
    ) -> None:
        super().__init__()
        self.paths = sorted(data_dir.glob(file_pattern))
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_key = text_key
        self.stop_token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_sequences]
        self.repeat = repeat

    def _trim_at_stop(self, ids: List[int]) -> List[int]:
        for stop in self.stop_token_ids:
            if not stop:
                continue
            for idx in range(len(ids) - len(stop) + 1):
                if ids[idx : idx + len(stop)] == stop:
                    return ids[:idx]
        return ids

    def _iter_docs(self) -> Iterable[List[int]]:
        while True:
            for path in self.paths:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if self.text_key not in record:
                            continue
                        tokens = self.tokenizer.encode(
                            record[self.text_key], add_special_tokens=False
                        )
                        tokens = self._trim_at_stop(tokens)
                        if tokens:
                            yield tokens
            if not self.repeat:
                break

    def __iter__(self):
        doc_id = 0
        current_tokens: List[int] = []
        current_doc_ids: List[int] = []
        for tokens in self._iter_docs():
            doc_id += 1
            start = 0
            while start < len(tokens):
                take = min(self.seq_len - len(current_tokens), len(tokens) - start)
                current_tokens.extend(tokens[start : start + take])
                current_doc_ids.extend([doc_id] * take)
                start += take
                if len(current_tokens) == self.seq_len:
                    yield {
                        "input_ids": torch.tensor(current_tokens, dtype=torch.long),
                        "doc_ids": torch.tensor(current_doc_ids, dtype=torch.long),
                    }
                    current_tokens, current_doc_ids = [], []


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    doc_ids = torch.stack([b["doc_ids"] for b in batch], dim=0)
    return {"input_ids": input_ids, "doc_ids": doc_ids}


def create_dataloader(cfg: MicrolmConfig, tokenizer):
    dataset = PackedIterableDataset(
        data_dir=Path(cfg.data.dataset_dir),
        file_pattern=cfg.data.file_pattern,
        tokenizer=tokenizer,
        seq_len=cfg.data.seq_len,
        text_key=cfg.data.text_key,
        stop_sequences=cfg.data.stop_sequences,
        repeat=cfg.data.repeat,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def compute_grad_accum(cfg: MicrolmConfig) -> int:
    tokens_per_micro = cfg.training.batch_size * cfg.data.seq_len
    steps = max(1, cfg.training.tokens_per_step_target // tokens_per_micro)
    return steps


def compute_max_steps(cfg: MicrolmConfig) -> int:
    if cfg.training.max_steps:
        return cfg.training.max_steps
    tokens_per_step = max(
        cfg.training.batch_size * cfg.data.seq_len * compute_grad_accum(cfg),
        1,
    )
    return math.ceil(cfg.training.total_tokens / tokens_per_step)


def train(run, cfg: MicrolmConfig):
    set_seed(cfg.training.seed)
    tokenizer = load_tokenizer(cfg.model)
    model = build_model(cfg.model, cfg.training, tokenizer)
    device = torch.device(cfg.training.device)
    model.to(device)

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(use_reentrant=False)
    if cfg.training.compile:
        model = torch.compile(model)

    dataloader = create_dataloader(cfg, tokenizer)
    grad_accum = compute_grad_accum(cfg)
    max_steps = compute_max_steps(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.optimizer.warmup_steps,
        num_training_steps=max_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.dtype.lower() in {"fp16", "float16"})
    model.train()
    global_step = 0
    tokens_seen = 0

    for batch in dataloader:
        if global_step >= max_steps:
            break
        input_ids = batch["input_ids"].to(device)
        doc_ids = batch["doc_ids"].to(device)
        attn_mask = build_chunk_causal_mask(doc_ids)
        with torch.cuda.amp.autocast(dtype=get_dtype(cfg.training)):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=input_ids,
            )
            loss = outputs.loss / grad_accum
        scaler.scale(loss).backward()

        if (global_step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            tokens_this_step = cfg.training.batch_size * cfg.data.seq_len * grad_accum
            tokens_seen += tokens_this_step
            if (global_step + 1) % cfg.training.log_interval == 0:
                run.log(
                    {
                        "train/loss": loss.item() * grad_accum,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/step": global_step + 1,
                        "train/tokens": tokens_seen,
                    }
                )
            if (global_step + 1) % cfg.training.save_interval == 0:
                save_artifacts(Path("./out/microlm"), model, tokenizer, run)
            global_step += 1

    save_artifacts(Path("./out/microlm"), model, tokenizer, run)
