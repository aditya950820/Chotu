"""Gate for Session 1: the model must be exactly the frozen design."""
import torch
from transformers import Qwen3ForCausalLM

from chotu.config import EXPECTED_PARAMS, chotu_config


def build():
    with torch.device("meta"):  # no memory needed just to count
        return Qwen3ForCausalLM(chotu_config())


def test_param_count():
    n = sum(p.numel() for p in build().parameters())
    assert n == EXPECTED_PARAMS, f"{n:,} params, expected {EXPECTED_PARAMS:,}"
    assert n < 200_000_000


def test_frozen_fields():
    m = build()
    c = m.config
    assert c.head_dim == 64 and c.num_key_value_heads == 5 and c.tie_word_embeddings
    assert m.lm_head.weight is m.model.embed_tokens.weight or m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    attn = m.model.layers[0].self_attn
    assert hasattr(attn, "q_norm") and hasattr(attn, "k_norm"), "QK-norm missing"
    assert attn.q_proj.bias is None, "biases must be off"


def breakdown():
    m = build()
    emb = m.model.embed_tokens.weight.numel()
    layer = sum(p.numel() for p in m.model.layers[0].parameters())
    attn = sum(p.numel() for p in m.model.layers[0].self_attn.parameters())
    mlp = sum(p.numel() for p in m.model.layers[0].mlp.parameters())
    print(f"embeddings (shared with output): {emb/1e6:6.2f}M")
    print(f"per layer: {layer/1e6:.2f}M  (attention {attn/1e6:.2f}M, MLP {mlp/1e6:.2f}M)")
    print(f"32 layers: {32*layer/1e6:6.2f}M")
    print(f"total: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")


if __name__ == "__main__":
    breakdown()
