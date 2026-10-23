"""Chotu model definition. Frozen decisions live in CLAUDE.md; change them only via docs/DECISIONS.md.

Every Qwen3Config field that matters is set explicitly: the library defaults
(head_dim=128, tie_word_embeddings=False, 32 KV heads, 32k positions) would silently
build a different model.
"""
from transformers import Qwen3Config

TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M"  # used as-is, 49,152 tokens
EXPECTED_PARAMS = 180_925_056  # asserted by tests/test_model_size.py


def chotu_config(**overrides) -> Qwen3Config:
    cfg = dict(
        vocab_size=49_152,
        hidden_size=640,
        intermediate_size=1_792,
        num_hidden_layers=32,
        num_attention_heads=10,
        num_key_value_heads=5,
        head_dim=64,
        max_position_embeddings=2_048,
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
    return Qwen3Config(**cfg)


def tiny_config() -> Qwen3Config:
    """~7M-parameter twin for rehearsals (Sessions 3-4). Same layout, smaller sizes."""
    return chotu_config(hidden_size=128, intermediate_size=384, num_hidden_layers=4,
                        num_attention_heads=2, num_key_value_heads=1)
