"""Chotu model definition. Frozen decisions live in CLAUDE.md; change them only via docs/DECISIONS.md.

Every Qwen3Config field that matters is set explicitly: the library defaults
(head_dim=128, tie_word_embeddings=False, 32 KV heads, 32k positions) would silently
build a different model.
"""
from transformers import Qwen3Config

TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M"  # used as-is, 49,152 tokens
EXPECTED_PARAMS = 180_925_056  # asserted by tests/test_model_size.py
PRETRAINING_WINDOW = 2_048
FINAL_CONTEXT = 8_192
LONG_CONTEXT_ROPE_THETA = 500_000.0


def chotu_config(**overrides) -> Qwen3Config:
    cfg = dict(
        vocab_size=49_152,
        hidden_size=640,
        intermediate_size=1_792,
        num_hidden_layers=32,
        num_attention_heads=10,
        num_key_value_heads=5,
        head_dim=64,
        max_position_embeddings=FINAL_CONTEXT,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        use_sliding_window=False,
        bos_token_id=0,  # <|endoftext|>
        eos_token_id=0,
        pad_token_id=0,
        initializer_range=0.02,
    )
    cfg.update(overrides)
    theta = cfg.pop("rope_theta")
    cfg.setdefault("rope_parameters", {"rope_type": "default", "rope_theta": theta})
    config = Qwen3Config(**cfg)
    config.rope_theta = config.rope_parameters["rope_theta"]
    return config


def long_context_config(**overrides) -> Qwen3Config:
    """Stage 2: extend training to 8k with a longer RoPE wavelength."""
    cfg = dict(max_position_embeddings=FINAL_CONTEXT, rope_theta=LONG_CONTEXT_ROPE_THETA)
    cfg.update(overrides)
    return chotu_config(**cfg)


def tiny_config() -> Qwen3Config:
    """~7M-parameter twin for rehearsals (Sessions 3-4). Same layout, smaller sizes."""
    return chotu_config(hidden_size=128, intermediate_size=384, num_hidden_layers=4,
                        num_attention_heads=2, num_key_value_heads=1)
