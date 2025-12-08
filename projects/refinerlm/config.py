from pydantic import BaseModel

from typing import Optional

class RefinerLMConfig(BaseModel):
    # Model
    model_name: str = "refinerlm-base"
    hidden_size: int = 256
    intermediate_size: int = 512
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    dropout_rate: float = 0.1

    num_recursions: int = 6

    chunk_size: int = 2048
    chunk_overlap: int = 128
    total_tokens: int = 100_000_000
    batch_size: int = 8
    grad_accum_steps: int = 16

    lr_max: float = 8e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    adam_weight_decay: float = 1e-1
    warmup_ratio: float = 0.02

    dataset: str = "HuggingFaceTB/cosmopedia-100k"
    dataset_split: str = "train"
    dataset_field: str = "text"
    hf_token: Optional[str] = None  # HuggingFace token if needed

    test_samples_dataset: Optional[int] = None
