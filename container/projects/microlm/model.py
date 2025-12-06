
from transformers import Gemma3Config, Gemma3ForCausalLM, Gemma3TextConfig
from .config import MicroLMConfig

def load_model(microLMConfig: MicroLMConfig):
    text_config = Gemma3TextConfig( # note tied word embeddings 
        vocab_size=262_208,
        hidden_size=microLMConfig.hidden_size,
        intermediate_size=microLMConfig.intermediate_size,
        num_hidden_layers=microLMConfig.num_hidden_layers,
        num_attention_heads=microLMConfig.num_attention_heads,
        num_key_value_heads=microLMConfig.num_key_value_heads,
        head_dim=microLMConfig.head_dim,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    config = Gemma3Config(text_config=text_config)
    model = Gemma3ForCausalLM(config)
    return model
