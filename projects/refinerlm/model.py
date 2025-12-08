
from transformers import Gemma3TextConfig
from .config import RefinerLMConfig
from typing import Optional, Dict
import torch
import torch.nn as nn
import copy

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import auto_docstring, can_return_tuple, logging
from transformers.generation import GenerationMixin


from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3PreTrainedModel,
    Gemma3TextScaledWordEmbedding,
    Gemma3DecoderLayer,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
    create_causal_mask,
    create_sliding_window_causal_mask,
    check_model_inputs,
)

logger = logging.get_logger(__name__)


def load_model(
    microLMConfig: RefinerLMConfig,
    vocab_size: int = 32_000,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    bos_token_id: int = 2,
):
    text_config = Gemma3TextConfig(  # note tied word embeddings
        vocab_size=vocab_size,
        hidden_size=microLMConfig.hidden_size,
        intermediate_size=microLMConfig.intermediate_size,
        num_hidden_layers=microLMConfig.num_hidden_layers,
        num_attention_heads=microLMConfig.num_attention_heads,
        num_key_value_heads=microLMConfig.num_key_value_heads,
        head_dim=microLMConfig.head_dim,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        hidden_dropout_prob=microLMConfig.dropout_rate,
        attention_dropout=microLMConfig.dropout_rate,
        attn_implementation="sdpa" # using torch.nn.attention sdpa_kernel to use flash attention (hopefully this works lol)
    )
    # config = Gemma3Config(text_config=text_config)
    model = RefinerGemma3ForCausalLM(text_config, n_recursions=microLMConfig.num_recursions, recursion_version=microLMConfig.recursion_version)

    print(f"Model loaded with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    return model



@auto_docstring
class RefinerGemma3TextModel(Gemma3PreTrainedModel):
    config: Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # TODO: raushan fix this after RoPE refactor. For now we hack it by reassigning thetas
        # when we want to create a local RoPE layer. Config defaults should hold values for global RoPE
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # NEW ARGS (optional, default keeps behavior identical)
        n_recursions: Optional[int] = None,
        recursion_version: int = 0,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        recursion_version:
            0 -> no special cross-recursion residual (just repeat the stack n_recursions times)
            1 -> simple residual between recursions (add output of each recursion to its input)
            2 -> interleaved input/output refiner (odd positions are outputs; we return only them)
        """

        # ------------------------------
        # defaults / basic sanity
        # ------------------------------
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # number of recursions – default 1
        if n_recursions is None or n_recursions < 1:
            n_recursions = 1

        # For non-standard paths, caching is messy; keep default behavior untouched when
        # user doesn't ask for recursion / fancy variants.
        if recursion_version == 2 or (n_recursions > 1 and recursion_version in (0, 1)):
            # Disable cache for multi-pass / interleaved mode – this is meant for training,
            # not for incremental autoregressive decoding with KV cache.
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with recursive/interleaved refinement. Setting `use_cache=False`."
                )
            use_cache = False
            past_key_values = None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # ------------------------------
        # Helper: build causal mask mapping from "normal" inputs
        # ------------------------------
        def build_causal_mask_mapping(
            input_embeds: torch.FloatTensor,
            token_attention_mask: Optional[torch.Tensor],
            cache_pos: torch.LongTensor,
            pos_ids: torch.LongTensor,
            pkv: Optional[Cache],
        ) -> Dict[str, torch.Tensor]:
            if isinstance(token_attention_mask, dict):
                # Already prepared by generate / external caller
                return token_attention_mask  # type: ignore

            mask_kwargs = {
                "config": self.config,
                "input_embeds": input_embeds,
                "attention_mask": token_attention_mask,
                "cache_position": cache_pos,
                "past_key_values": pkv,
                "position_ids": pos_ids,
            }
            return {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }


        if recursion_version == 2:
            hidden_states = inputs_embeds
            batch_size, seq_len, hidden_dim = hidden_states.shape

            # Interleave: [in0, out0, in1, out1, ...]
            interleaved_seq_len = seq_len * 2
            output_init = torch.zeros_like(hidden_states)

            interleaved_hidden = torch.empty(
                batch_size,
                interleaved_seq_len,
                hidden_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            interleaved_hidden[:, 0::2, :] = hidden_states       # input tokens
            interleaved_hidden[:, 1::2, :] = output_init         # output tokens

            # Build interleaved cache_position / position_ids
            base_cache_position = cache_position  # shape (seq_len,)
            interleaved_cache_position = torch.empty(
                interleaved_seq_len,
                dtype=base_cache_position.dtype,
                device=base_cache_position.device,
            )
            interleaved_cache_position[0::2] = base_cache_position * 2
            interleaved_cache_position[1::2] = base_cache_position * 2 + 1
            interleaved_position_ids = interleaved_cache_position.unsqueeze(0).expand(batch_size, -1)

            # ---- Get base 4D masks on the ORIGINAL sequence length T ----
            # We want base_full_mask, base_sliding_mask of shape [B, H, T, T].
            if isinstance(attention_mask, dict):
                base_full_mask = attention_mask["full_attention"]
                base_sliding_mask = attention_mask["sliding_attention"]
            elif attention_mask is not None and not isinstance(attention_mask, dict) and attention_mask.dim() == 4:
                # You passed a "normal" 4D mask already: [B, H, T, T] or [B, 1, T, T]
                base_full_mask = attention_mask
                base_sliding_mask = attention_mask
            else:
                # Build from a "normal" token-level mask / None
                base_mapping = build_causal_mask_mapping(
                    input_embeds=hidden_states,
                    token_attention_mask=attention_mask if (attention_mask is None or attention_mask.dim() == 2) else None,
                    cache_pos=cache_position,
                    pos_ids=position_ids,
                    pkv=None,
                )
                base_full_mask = base_mapping["full_attention"]
                base_sliding_mask = base_mapping["sliding_attention"]

            def expand_interleaved_mask(base_mask: torch.Tensor) -> Optional[torch.Tensor]:
                if base_mask is None:
                    return None
                # base_mask: [B, H or 1, T, T]
                B, H, Tq, Tk = base_mask.shape
                assert Tq == seq_len and Tk == seq_len, (
                    f"Base mask seq_len mismatch: got {base_mask.shape}, expected T={seq_len}"
                )

                L = seq_len * 2
                # Use the most negative value as "forbidden"
                forbidden_val = base_mask.min()

                # Start with everything forbidden
                out = torch.full(
                    (B, H, L, L),
                    forbidden_val,
                    dtype=base_mask.dtype,
                    device=base_mask.device,
                )

                q_idx = torch.arange(seq_len, device=base_mask.device)
                k_idx = torch.arange(seq_len, device=base_mask.device)

                input_q = 2 * q_idx       # even positions
                output_q = 2 * q_idx + 1  # odd positions
                input_k = 2 * k_idx       # even positions
                # output_k = 2 * k_idx + 1  # odd positions (kept forbidden)

                # 1) Inputs attend to inputs, copying original pattern
                #    out[..., 2i, 2j] = base_mask[..., i, j]
                out[..., input_q[:, None], input_k] = base_mask

                # 2) Outputs attend to inputs, also copying original pattern
                #    out[..., 2i+1, 2j] = base_mask[..., i, j]
                out[..., output_q[:, None], input_k] = base_mask

                # 3) Any key at odd positions stays at `forbidden_val` (no one attends to outputs),
                #    which is already set by initialization.
                return out

            interleaved_causal_mask_mapping = {
                "full_attention": expand_interleaved_mask(base_full_mask),
                "sliding_attention": expand_interleaved_mask(base_sliding_mask),
            }

            # Position embeddings for interleaved sequence
            position_embeddings_global = self.rotary_emb(interleaved_hidden, interleaved_position_ids)
            position_embeddings_local = self.rotary_emb_local(interleaved_hidden, interleaved_position_ids)

            # Decoder layers with recursion. Keep the input tokens "fresh" each recursion.
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None

            fixed_input_tokens = interleaved_hidden[:, 0::2, :].clone()
            hidden_states_interleaved = interleaved_hidden

            for r in range(n_recursions):
                # refresh input tokens to original embeddings at each recursion
                hidden_states_interleaved = hidden_states_interleaved.clone()
                hidden_states_interleaved[:, 0::2, :] = fixed_input_tokens

                for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                    if output_hidden_states:
                        # store only logical tokens (inputs) for convenience
                        all_hidden_states += (hidden_states_interleaved[:, 0::2, :],)

                    layer_outputs = decoder_layer(
                        hidden_states_interleaved,
                        position_embeddings_global=position_embeddings_global,
                        position_embeddings_local=position_embeddings_local,
                        attention_mask=interleaved_causal_mask_mapping[decoder_layer.attention_type],
                        position_ids=interleaved_position_ids,
                        past_key_value=None,
                        output_attentions=output_attentions,
                        use_cache=False,
                        cache_position=interleaved_cache_position,
                        **kwargs,
                    )

                    hidden_states_interleaved = layer_outputs[0]

                    if output_attentions:
                        all_self_attns += (layer_outputs[1],)

            # Final norm, then take ONLY the output tokens (odd positions)
            hidden_states_interleaved = self.norm(hidden_states_interleaved)
            final_hidden_states = hidden_states_interleaved[:, 1::2, :]  # (B, T, D)

            if output_hidden_states:
                all_hidden_states += (final_hidden_states,)

            return BaseModelOutputWithPast(
                last_hidden_state=final_hidden_states,
                past_key_values=None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )


        # First build the usual causal masks on the original sequence
        causal_mask_mapping = build_causal_mask_mapping(
            input_embeds=inputs_embeds,
            token_attention_mask=attention_mask,
            cache_pos=cache_position,
            pos_ids=position_ids,
            pkv=past_key_values,
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        num_layers = self.config.num_hidden_layers
        layers = self.layers[:num_layers]


        if n_recursions == 1 and recursion_version == 0:
            # EXACT ORIGINAL BEHAVIOR
            for decoder_layer in layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

        # Recursive variants on the original (non-interleaved) sequence
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        for r in range(n_recursions):
            if recursion_version == 1:
                # simple residual between recursions
                residual_rec_in = hidden_states
            else:
                residual_rec_in = None

            for decoder_layer in layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            if recursion_version == 1 and residual_rec_in is not None:
                hidden_states = hidden_states + residual_rec_in

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@auto_docstring
class RefinerGemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma3TextConfig
    base_model_prefix = "language_model"

    def __init__(self, config: Gemma3TextConfig, n_recursions: int = 1, recursion_version: int = 0):
        super().__init__(config)
        self.model = RefinerGemma3TextModel(config)
        self.n_recursions = n_recursions
        self.recursion_version = recursion_version
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep  = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma3ForCausalLM

        >>> model = Gemma3ForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""

        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            n_recursions=self.n_recursions,
            recursion_version=self.recursion_version,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )