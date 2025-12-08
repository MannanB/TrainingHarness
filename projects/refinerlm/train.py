import math
from typing import Any, Dict, List, Optional

import torch
from torch.amp import GradScaler
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F
from transformers import T5TokenizerFast
from tqdm import tqdm

from .config import RefinerLMConfig
from .model import load_model

from torch.nn.attention import SDPBackend, sdpa_kernel

import wandb, os


def load_raw_dataset(cfg: RefinerLMConfig):
    return load_dataset(
        cfg.dataset,
        split=cfg.dataset_split,
        cache_dir=getattr(cfg, "dataset_cache_dir", None),
        token=getattr(cfg, "hf_token", None),
    )

def _make_block_causal_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = attention_mask.shape[0]
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    valid = attention_mask.bool()
    block = causal & valid.unsqueeze(0) & valid.unsqueeze(1)  # [S, S]
    return block.unsqueeze(0)  # [1, S, S]


def _finalize_chunk(buffer: List[int], chunk_size: int) -> Dict[str, torch.Tensor]:
    assert len(buffer) == chunk_size, "finalize_chunk expects full chunk"
    input_ids = torch.tensor(buffer, dtype=torch.long)
    block_mask = _make_block_causal_mask(torch.ones(chunk_size, dtype=torch.bool))
    return {
        "input_ids": input_ids,
        "block_mask": block_mask,
    }


def build_chunked_dataset(cfg: RefinerLMConfig, tokenizer) -> IterableDataset:
    limit = getattr(cfg, "test_samples_dataset", None)

    class PackedIterableDataset(IterableDataset):
        def __iter__(self_inner):
            raw = load_raw_dataset(cfg)
            try:
                total = len(raw) if limit is None else min(len(raw), limit)
            except TypeError:
                total = limit

            iterator = raw if limit is None else (row for idx, row in enumerate(raw) if idx < limit)
            # iterator = tqdm(iterator, total=total, desc="tokenize+pack", leave=False)

            chunk_size = cfg.chunk_size
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("Tokenizer must provide eos_token_id when no padding is used.")

            buffer: List[int] = []
            for record in iterator:
                text = record.get(cfg.dataset_field, "")
                tokens: List[int] = tokenizer.encode(text, add_special_tokens=False)
                if not tokens:
                    continue
                tokens.append(eos_id)

                idx = 0
                while idx < len(tokens):
                    space = chunk_size - len(buffer)
                    take = min(space, len(tokens) - idx)
                    buffer.extend(tokens[idx : idx + take])
                    idx += take
                    if len(buffer) == chunk_size:
                        yield _finalize_chunk(buffer, chunk_size)
                        buffer = []
            # drop incomplete tail to avoid padding entirely

    return PackedIterableDataset()


def build_dataloader(
    cfg: RefinerLMConfig,
    tokenizer,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = build_chunked_dataset(cfg, tokenizer)

    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
            "block_mask": torch.stack([b["block_mask"] for b in batch], dim=0),
        }

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=_collate,
    )


def _create_scheduler(optimizer, num_steps: int, warmup_ratio: float = 0.02):
    warmup_steps = max(1, int(num_steps * warmup_ratio))
    min_lr_scale = 0.1

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_scale + (1 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda)


def _prepare_tokenizer():
    tok = T5TokenizerFast.from_pretrained("t5-base")
    tok.padding_side = "right"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def train(run: wandb.Run, cfg: RefinerLMConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = False # FP16 is too unstable for now (probably need to tune more hyperparameters)
    tokenizer = _prepare_tokenizer()

    model = load_model(
        cfg,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id or tokenizer.pad_token_id,
    ).to(device).to(torch.float16 if use_fp16 else torch.float32)

    dataloader = build_dataloader(cfg, tokenizer, shuffle=False)

    tokens_per_step = cfg.batch_size * cfg.chunk_size * cfg.grad_accum_steps
    print(f"Tokens per step: {tokens_per_step}")
    total_steps = max(1, cfg.total_tokens // tokens_per_step)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr_max,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.adam_weight_decay,
    )
    scheduler = _create_scheduler(optimizer, total_steps, warmup_ratio=cfg.warmup_ratio)

    model.train()
    optimizer.zero_grad()
    if run is not None:
        run.watch(model, log_freq=100, log="all")

    global_step = 0
    micro_step = 0
    pbar = tqdm(total=total_steps, desc="train", leave=False)
    os.makedirs("./out", exist_ok=True)

    grad_norm = 0.0
    with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                                          SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]) if torch.cuda.is_available() else torch.enable_grad():
        data_iter = iter(dataloader)
        while global_step < total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["block_mask"],
                )
            logits = outputs.logits.float()  # [B, S, V]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            ) / cfg.grad_accum_steps

 
            loss.backward()
            micro_step += 1

            if global_step % 50 == 0:
                # Save model weights
                model_path = os.path.join("./out", f"refinerlm-{global_step}.pt")
                torch.save(model.state_dict(), model_path)
                if run is not None:
                    run.save(model_path, policy="now")



            if micro_step % cfg.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                true_loss = loss.item() * cfg.grad_accum_steps
                tokens_seen = global_step * tokens_per_step

                if run is not None:
                    run.log(
                        {
                            "train/loss": true_loss,
                            "train/lr": current_lr,
                            "train/grad_norm": grad_norm,
                            "train/global_step": global_step,
                            "train/tokens_seen": tokens_seen,
                        },
                        step=global_step,
                    )
                else:
                    # print(f"Step {global_step}: loss={true_loss:.4f}, lr={current_lr:.6e}, grad_norm={grad_norm:.4f}, tokens_seen={tokens_seen}")
                    pass
                pbar.set_postfix(loss=true_loss, lr=current_lr)
                pbar.update(1)

    pbar.close()


    # Save model weights
    model_path = os.path.join("./out", "refinerlm.pt")
    torch.save(model.state_dict(), model_path)
    if run is not None:
        run.save(model_path, policy="now")


    return model
