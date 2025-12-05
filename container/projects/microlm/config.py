from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = None  # if set, load pretrained weights
    tokenizer_name: str = "google/gemma-2b"  # gemma tokenizer for text encoding
    vocab_size: Optional[int] = None  # only used when training from scratch
    hidden_size: int = 768
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: Optional[int] = None  # defaults to 4 * hidden_size when None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_traditional: bool = True
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    max_position_embeddings: int = 1024


@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 2000
    max_steps: Optional[int] = None
    grad_clip: float = 1.0


@dataclass
class DataConfig:
    dataset_dir: str = "./data/smollm_subset"
    file_pattern: str = "*.jsonl"
    text_key: str = "text"
    seq_len: int = 1024
    num_workers: int = 4
    repeat: bool = True
    tokenizer_name: Optional[str] = None  # fallback to model.tokenizer_name when None
    stop_sequences: List[str] = field(default_factory=lambda: ["\n\n", "<|eot_id|>"])
    pack_buffer_tokens: int = 20000


@dataclass
class TrainingConfig:
    batch_size: int = 4  # sequences per micro batch
    tokens_per_step_target: int = 500_000
    total_tokens: int = 2_000_000_000
    max_steps: Optional[int] = None  # derived from total_tokens if None
    log_interval: int = 20
    save_interval: int = 1000
    eval_interval: int = 1000
    device: str = "cuda"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    compile: bool = False
    seed: int = 42


@dataclass
class LoggingConfig:
    project: str = "microlm"
    run_name: Optional[str] = None
    wandb_mode: str = "online"
    log_gradients: bool = False


@dataclass
class MicrolmConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def merge_overrides(self, overrides: Dict[str, Any]):
        """Recursively merge a flat dict of overrides into the dataclasses."""
        for key, value in overrides.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
                for inner_k, inner_v in value.items():
                    if hasattr(current, inner_k):
                        setattr(current, inner_k, inner_v)
                continue
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
