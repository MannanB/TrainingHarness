from pydantic import BaseModel

class MicroLMConfig(BaseModel):
    # Model
    model_name: str = "microlm-base"
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    dropout_rate: float = 0.1