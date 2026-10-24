"""Chotu X-ray: look inside the real tokenizer, data and model.

Everything here reads real objects (the SmolLM2 tokenizer, your shards, a Chotu model or
checkpoint) and never changes them. Works on the untrained model now and on checkpoints later,
so the same picture can be compared before and after training.
"""
import html
from collections import Counter

import numpy as np

EOT = 0
_PALETTE = ["#fde68a", "#bfdbfe", "#bbf7d0", "#fecaca", "#ddd6fe", "#fed7aa", "#a5f3fc", "#fbcfe8"]


# ---------- 1. Tokenizer lens ----------
def token_lens(tok, text):
    """HTML: the text cut into its real tokens, each coloured, with its id underneath."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    spans = []
    for i, t in enumerate(ids):
        piece = tok.decode([t]).replace(" ", "␣").replace("\n", "↵")
        spans.append(
            f"<span style='display:inline-block;margin:2px;padding:2px 4px;border-radius:4px;"
            f"background:{_PALETTE[i % len(_PALETTE)]};color:#111;font-family:monospace;text-align:center'>"
            f"{html.escape(piece)}<br><small style='color:#555'>{t}</small></span>")
    words = max(1, len(text.split()))
    head = (f"<div style='font-family:sans-serif;margin-bottom:4px'><b>{len(ids)} tokens</b> for "
            f"{len(text)} characters, {words} words ({len(ids) / words:.2f} tokens per word)</div>")
    return head + "<div>" + "".join(spans) + "</div>"


# ---------- 2. Shard microscope ----------
def shard_window(tok, shard_path, start=None, n=120, seed=0):
    """HTML: n raw token ids from a real shard, decoded, with document boundaries marked."""
    arr = np.memmap(shard_path, dtype=np.uint16, mode="r")
    if start is None:
        start = int(np.random.default_rng(seed).integers(0, max(1, len(arr) - n)))
    window = arr[start:start + n].tolist()
    out = []
    for t in window:
        if t == EOT:
            out.append("<span style='background:#ef4444;color:white;padding:1px 6px;border-radius:4px;"
                       "margin:0 3px'>⟨end of document⟩</span><br>")
        else:
            out.append(f"<span title='id {t}' style='border-bottom:1px dotted #999'>{html.escape(tok.decode([t]))}</span>")
    ids_line = " ".join(str(t) for t in window[:40])
    return (f"<div style='font-family:sans-serif'><b>{shard_path}</b>, tokens {start:,} to {start + n:,} "
            f"of {len(arr):,}</div><div style='font-family:monospace;color:#666;margin:6px 0'>"
            f"what the model sees: {ids_line} …</div><div style='line-height:1.9'>{''.join(out)}</div>")


def token_frequency(shard_paths, max_tokens=50_000_000):
    """Count token ids across real shards (up to max_tokens). Returns (Counter, total)."""
    counts = np.zeros(65_536, dtype=np.int64)
    total = 0
    for p in shard_paths:
        arr = np.memmap(p, dtype=np.uint16, mode="r")
        take = arr[: max(0, max_tokens - total)]
        counts += np.bincount(take, minlength=65_536)
        total += len(take)
        if total >= max_tokens:
            break
    return counts, total


def plot_token_frequency(tok, counts, total, top=25):
    import matplotlib.pyplot as plt
    nz = np.sort(counts[counts > 0])[::-1]
    cum = np.cumsum(nz) / total
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].loglog(np.arange(1, len(nz) + 1), nz)
    ax[0].set(title="Rank vs count (log-log): a few tokens dominate", xlabel="token rank", ylabel="count")
    for k in [10, 100, 1000, 10000]:
        if k <= len(cum):
            ax[1].axvline(k, color="#ccc", lw=0.8)
            ax[1].text(k, cum[k - 1], f" top {k}: {cum[k - 1]:.0%}", fontsize=9)
    ax[1].semilogx(np.arange(1, len(cum) + 1), cum)
    ax[1].set(title="Share of all text covered by the top-k tokens", xlabel="k", ylabel="cumulative share")
    order = np.argsort(counts)[::-1][:top]
    labels = [repr(tok.decode([int(i)]))[1:-1] or "∅" for i in order]
    ax[2].barh(range(top)[::-1], counts[order] / total * 100)
    ax[2].set_yticks(range(top)[::-1], labels, fontsize=8)
    ax[2].set(title=f"Top {top} tokens (% of all tokens)", xlabel="%")
    fig.suptitle(f"{total:,} real tokens from your shards; {int((counts > 0).sum()):,} of 49,152 vocab ids used")
    fig.tight_layout()
    return fig


# ---------- 3. Parameter map ----------
def param_breakdown(model):
    """Where the parameters live, by component, from the real model object."""
    m = model.model
    rows = {"token embeddings (shared with output)": m.embed_tokens.weight.numel()}
    att = mlp = norm = 0
    for layer in m.layers:
        att += sum(p.numel() for p in layer.self_attn.parameters())
        mlp += sum(p.numel() for p in layer.mlp.parameters())
        norm += sum(p.numel() for p in layer.input_layernorm.parameters()) + sum(
            p.numel() for p in layer.post_attention_layernorm.parameters())
    rows[f"attention ({len(m.layers)} layers)"] = att
    rows[f"MLP ({len(m.layers)} layers)"] = mlp
    rows["norms"] = norm + m.norm.weight.numel()
    total = sum(p.numel() for p in model.parameters())
    return rows, total


def plot_param_map(model):
    import matplotlib.pyplot as plt
    rows, total = param_breakdown(model)
    l0 = model.model.layers[0]
    fig, ax = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw={"width_ratios": [3, 2]})
    left = 0
    colors = ["#60a5fa", "#f59e0b", "#10b981", "#9ca3af"]
    for (name, n), c in zip(rows.items(), colors):
        ax[0].barh(0, n, left=left, color=c)
        if n / total > 0.03:
            ax[0].text(left + n / 2, 0, f"{name}\n{n / 1e6:.1f}M ({n / total:.0%})", ha="center", va="center", fontsize=9)
        left += n
    ax[0].set(title=f"All {total:,} parameters", yticks=[], xlabel="parameters")
    parts = {
        "q_proj": l0.self_attn.q_proj.weight.numel(), "k_proj": l0.self_attn.k_proj.weight.numel(),
        "v_proj": l0.self_attn.v_proj.weight.numel(), "o_proj": l0.self_attn.o_proj.weight.numel(),
        "gate_proj": l0.mlp.gate_proj.weight.numel(), "up_proj": l0.mlp.up_proj.weight.numel(),
        "down_proj": l0.mlp.down_proj.weight.numel(),
    }
    ax[1].bar(parts.keys(), [v / 1e6 for v in parts.values()],
              color=["#f59e0b"] * 4 + ["#10b981"] * 3)
    ax[1].set(title="Inside one layer (millions)", ylabel="M")
    ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


# ---------- 4. Forward-pass X-ray ----------
def xray(model, tok, text, top_k=3):
    """Run one sentence through the model and capture what happens inside.

    Returns dict with tokens, attentions [layer][head, q, k], per-layer hidden-state size,
    how much each layer changed the signal, and the logit lens: the model's best guess for
    the next token if it stopped after each layer.
    """
    import torch
    model.eval()
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model(**enc, output_attentions=True, output_hidden_states=True)
        hs = out.hidden_states  # embeddings + one per layer
        lens_probs, lens_top = [], []
        for h in hs:
            logits = model.lm_head(model.model.norm(h))[0]
            p = torch.softmax(logits.float(), -1)
            v, i = p.topk(top_k, -1)
            lens_probs.append(v.numpy())
            lens_top.append(i.numpy())
        norms = np.array([h[0].float().norm(dim=-1).numpy() for h in hs])
        change = np.array([(hs[k + 1] - hs[k])[0].float().norm(dim=-1).mean().item() / max(1e-9, hs[k][0].float().norm(dim=-1).mean().item())
                           for k in range(len(hs) - 1)])
        final = torch.softmax(out.logits[0, -1].float(), -1)
        fv, fi = final.topk(10)
    return {
        "tokens": [tok.decode([t]) for t in enc["input_ids"][0].tolist()],
        "attentions": [a[0].float().numpy() for a in out.attentions],
        "hidden_norms": norms, "layer_change": change,
        "lens_top": np.stack(lens_top), "lens_probs": np.stack(lens_probs),
        "next": [(tok.decode([int(i)]), float(v)) for v, i in zip(fv, fi)],
        "tok": tok,
    }


def plot_attention(x, layer=0, max_heads=10):
    import matplotlib.pyplot as plt
    att = x["attentions"][layer]
    h = min(max_heads, att.shape[0])
    cols = 5
    rows = (h + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    toks = [t.replace("\n", "↵") for t in x["tokens"]]
    for k, ax in enumerate(np.array(axes).ravel()):
        if k >= h:
            ax.axis("off")
            continue
        ax.imshow(att[k], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"layer {layer} · head {k}", fontsize=9)
        ax.set_xticks(range(len(toks)), toks, rotation=90, fontsize=7)
        ax.set_yticks(range(len(toks)), toks, fontsize=7)
    fig.suptitle("Attention: row = the word doing the looking, column = the earlier word it looks at (bright = strong)")
    fig.tight_layout()
    return fig


def plot_logit_lens(x, every=1):
    """Table: rows = layers, columns = positions. Cell = the model's top guess for the NEXT token if it stopped there."""
    import matplotlib.pyplot as plt
    tok = x["tok"]
    layers = list(range(0, x["lens_top"].shape[0], every))
    if layers[-1] != x["lens_top"].shape[0] - 1:
        layers.append(x["lens_top"].shape[0] - 1)
    P = x["lens_probs"][layers, :, 0]
    fig, ax = plt.subplots(figsize=(1.1 * len(x["tokens"]) + 2, 0.32 * len(layers) + 1.5))
    # colour scale from the layers only: the embedding row is ~certain because of tied embeddings
    vmax = max(1e-4, float(P[1:].max())) if len(layers) > 1 else float(P.max())
    ax.imshow(np.minimum(P, vmax), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for r, L in enumerate(layers):
        for c in range(len(x["tokens"])):
            g = tok.decode([int(x["lens_top"][L, c, 0])]).replace("\n", "↵")
            ax.text(c, r, g[:8], ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(x["tokens"])), [t.replace("\n", "↵") for t in x["tokens"]], rotation=45, fontsize=8)
    ax.set_yticks(range(len(layers)), ["embed" if L == 0 else f"layer {L}" for L in layers], fontsize=7)
    ax.set_title("Logit lens: the next-token guess after each layer (colour = confidence)")
    fig.tight_layout()
    return fig


def plot_layer_signal(x):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.5))
    ax[0].plot(x["hidden_norms"].mean(1), marker="o", ms=3)
    ax[0].set(title="Size of the hidden state after each layer (avg over words)", xlabel="layer (0 = embeddings)")
    ax[1].bar(range(1, len(x["layer_change"]) + 1), x["layer_change"])
    ax[1].set(title="How much each layer changes the signal (relative)", xlabel="layer")
    fig.tight_layout()
    return fig
