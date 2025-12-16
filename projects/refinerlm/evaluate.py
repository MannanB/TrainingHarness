import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import T5TokenizerFast

from .config import RefinerLMConfig
from .model import load_model
from .train import build_dataloader


def load_cfg(path: str) -> RefinerLMConfig:
    with open(path, "r") as f:
        data = json.load(f)
    return RefinerLMConfig(**data.get("input", data).get("config", data))


def prepare_tokenizer() -> T5TokenizerFast:
    tok = T5TokenizerFast.from_pretrained("t5-base")
    tok.padding_side = "right"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def find_checkpoints(directory: Path) -> List[Path]:
    checkpoints = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.startswith("refinerlm-")
    ]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {directory}")

    def _sort_key(path: Path):
        match = re.search(r"refinerlm-(\d+)", path.stem)
        return int(match.group(1)) if match else path.name

    return sorted(checkpoints, key=_sort_key)


def sample_batches(dataloader, num_batches: int) -> List[Dict[str, torch.Tensor]]:
    batches: List[Dict[str, torch.Tensor]] = []
    iterator = iter(dataloader)
    for _ in range(num_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batches.append(batch)
    if not batches:
        raise RuntimeError("Dataloader yielded no batches for evaluation.")
    return batches


def evaluate_model(
    model: torch.nn.Module,
    batches: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[List[float], List[float], List[float]]:
    model.eval()
    total_tokens = 0
    total_losses: List[float] = []
    total_correct: List[int] = []
    num_recursions: int = 0

    with torch.no_grad():
        for batch in batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["block_mask"],
                output_hidden_states=True,
            )
            logits = outputs.logits.float()

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )

            recursion_states = outputs.hidden_states

            if num_recursions == 0:
                num_recursions = len(recursion_states)
                total_losses = [0.0 for _ in range(num_recursions)]
                total_correct = [0 for _ in range(num_recursions)]
            elif len(recursion_states) != num_recursions:
                raise RuntimeError("Inconsistent recursion count during evaluation.")

            # baseline recursion 0 aligns to outputs.logits
            preds = shift_logits.argmax(dim=-1)
            total_losses[-1] += loss.item() * shift_labels.numel()
            total_correct[-1] += (preds == shift_labels).sum().item()

            # compute metrics for intermediate recursions using captured hidden states
            for idx, hidden in enumerate(recursion_states[:-1]):
                logits_r = model.lm_head(hidden)
                shift_logits_r = logits_r[:, :-1, :].contiguous()
                loss_r = F.cross_entropy(
                    shift_logits_r.view(-1, shift_logits_r.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean",
                )
                preds_r = shift_logits_r.argmax(dim=-1)
                total_losses[idx] += loss_r.item() * shift_labels.numel()
                total_correct[idx] += (preds_r == shift_labels).sum().item()

            total_tokens += shift_labels.numel()

    if total_tokens == 0:
        raise RuntimeError("No tokens processed during evaluation.")

    avg_losses = [loss_sum / total_tokens for loss_sum in total_losses]
    accuracies = [correct / total_tokens for correct in total_correct]
    perplexities = [math.exp(min(700, l)) for l in avg_losses]  # avoid overflow
    return avg_losses, accuracies, perplexities


def plot_metrics(
    steps: List[int],
    losses: List[float],
    accuracies: List[float],
    perplexities: List[float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

    axes[0].plot(steps, losses, marker="o")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(steps, accuracies, marker="o", color="green")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(steps, perplexities, marker="o", color="orange")
    axes[2].set_ylabel("Perplexity")
    axes[2].set_xlabel("Checkpoint step")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_recursion_metrics(
    steps: List[int],
    losses: List[List[float]],
    accuracies: List[List[float]],
    perplexities: List[List[float]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

    num_recursions = len(losses[0]) if losses else 0
    for r in range(num_recursions):
        loss_curve = [loss_list[r] for loss_list in losses]
        acc_curve = [acc_list[r] for acc_list in accuracies]
        ppl_curve = [ppl_list[r] for ppl_list in perplexities]
        label = f"r{r+1}"
        axes[0].plot(steps, loss_curve, marker="o", label=label)
        axes[1].plot(steps, acc_curve, marker="o", label=label)
        axes[2].plot(steps, ppl_curve, marker="o", label=label)

    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].set_ylabel("Perplexity")
    axes[2].set_xlabel("Checkpoint step")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    axes[0].legend(title="Checkpoint step", loc="best")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate refinerlm checkpoints.")
    parser.add_argument("--config-path", required=True, help="Path to config JSON.")
    parser.add_argument("--checkpoints-dir", required=True, help="Directory containing refinerlm-* checkpoints.")
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="How many batches to evaluate per checkpoint (default: grad_accum_steps from config).",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to save the metrics plot (default: <checkpoints-dir>/eval_metrics.png).",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config_path)
    tokenizer = prepare_tokenizer()
    dataloader = build_dataloader(cfg, tokenizer, shuffle=False)

    num_batches = args.num_batches or 1
    batches = sample_batches(dataloader, num_batches)

    checkpoints = find_checkpoints(Path(args.checkpoints_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steps: List[int] = []
    losses: List[float] = []
    accuracies: List[float] = []
    perplexities: List[float] = []
    rec_losses: List[List[float]] = []
    rec_accuracies: List[List[float]] = []
    rec_perplexities: List[List[float]] = []

    ckpt_bar = tqdm(checkpoints, desc="checkpoints")
    for ckpt_path in ckpt_bar:
        ckpt_bar.set_postfix_str(ckpt_path.name)

        model = load_model(
            cfg,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id or tokenizer.pad_token_id,
        ).to(device)

        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

        rec_loss_vals, rec_acc_vals, rec_ppl_vals = evaluate_model(model, batches, device)

        match = re.search(r"refinerlm-(\d+)", ckpt_path.stem)
        step = int(match.group(1)) if match else len(steps)

        steps.append(step)
        losses.append(rec_loss_vals[-1])
        accuracies.append(rec_acc_vals[-1])
        perplexities.append(rec_ppl_vals[-1])
        rec_losses.append(rec_loss_vals)
        rec_accuracies.append(rec_acc_vals)
        rec_perplexities.append(rec_ppl_vals)

        ckpt_bar.set_postfix(loss=f"{rec_loss_vals[-1]:.4f}", acc=f"{rec_acc_vals[-1]:.4f}", ppl=f"{rec_ppl_vals[-1]:.2f}")

    output_path = Path(args.output_path) if args.output_path else Path(args.checkpoints_dir) / "eval_metrics.png"
    plot_metrics(steps, losses, accuracies, perplexities, output_path)
    recursion_output_path = output_path.with_name(output_path.stem + "_recursions.png")
    plot_recursion_metrics(
        steps=steps,
        losses=rec_losses,
        accuracies=rec_accuracies,
        perplexities=rec_perplexities,
        output_path=recursion_output_path,
    )

    print("Checkpoint metrics:")
    for step, loss_list, acc_list, ppl_list in zip(steps, rec_losses, rec_accuracies, rec_perplexities):
        metric_str = " ".join(
            f"r{r}:loss={l:.4f} acc={a:.4f} ppl={p:.2f}"
            for r, (l, a, p) in enumerate(zip(loss_list, acc_list, ppl_list), start=1)
        )
        print(f"  step {step:>6}: {metric_str}")
    print(f"Saved final metric plot to {output_path}")
    print(f"Saved recursion-by-recursion plot to {recursion_output_path}")
    plt.show()

if __name__ == "__main__":
    main()
