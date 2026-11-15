// Talk to the Twins: the 7M twin's tokenizer and forward pass in plain JavaScript (no libraries).
// Same maths as Hugging Face Qwen3: RMSNorm, attention with QK-norm and RoPE, grouped KV heads, SwiGLU MLP, tied output.
// Checked against PyTorch by test_twins.js (same next-token guesses, logits within float rounding).
(function (root) {
  // ---------- tokenizer: SmolLM2 byte-level BPE, digits split one by one ----------
  function byteMap() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice(); let n = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    const enc = {}, dec = {};
    bs.forEach((b, i) => { enc[b] = String.fromCharCode(cs[i]); dec[String.fromCharCode(cs[i])] = b; });
    return { enc, dec };
  }
  const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  function Tokenizer(json) {
    this.vocab = json.vocab;
    this.ids = new Map(json.vocab.map((t, i) => [t, i]));
    this.ranks = new Map(json.merges.map((m, i) => [m, i]));
    this.bm = byteMap(); this.cache = new Map(); this.utf8 = new TextEncoder();
  }
  Tokenizer.prototype.bpe = function (word) {
    if (this.cache.has(word)) return this.cache.get(word);
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + ' ' + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = parts[best], b = parts[best + 1], out = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { out.push(a + b); i++; } else out.push(parts[i]);
      }
      parts = out;
    }
    const ids = parts.map((p) => this.ids.get(p));
    this.cache.set(word, ids);
    return ids;
  };
  const SPECIAL = /(<\|endoftext\|>|<\|im_start\|>|<\|im_end\|>)/;  // special markers stay whole
  Tokenizer.prototype.encode = function (text) {
    const out = [];
    for (const part of text.split(SPECIAL)) {
      if (!part) continue;
      if (SPECIAL.test(part)) { out.push(this.ids.get(part)); continue; }
      out.push(...this.encodePlain(part));
    }
    return out;
  };
  Tokenizer.prototype.encodePlain = function (text) {
    const out = [];
    for (const run of text.split(/([0-9\uFF10-\uFF19])/u)) {  // ASCII (and fullwidth) digits become their own pieces first
      if (!run) continue;
      for (const m of run.matchAll(PAT)) {
        const bytes = this.utf8.encode(m[0]);
        let s = ''; for (const b of bytes) s += this.bm.enc[b];
        out.push(...this.bpe(s));
      }
    }
    return out;
  };
  Tokenizer.prototype.decode = function (ids) {
    const bytes = [];
    for (const id of ids) for (const ch of this.vocab[id]) { const b = this.bm.dec[ch]; if (b !== undefined) bytes.push(b); }
    return new TextDecoder().decode(new Uint8Array(bytes));
  };
  Tokenizer.prototype.piece = function (id) { return this.decode([id]); };

  // ---------- model ----------
  function halfToFloat(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
      out[i] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return out;
  }
  function Twin(buffer, manifest) {
    const all = halfToFloat(new Uint16Array(buffer)); this.spec = manifest.spec; this.w = {};
    let off = 0;
    for (const [name, shape] of manifest.tensors) { const n = shape.reduce((a, b) => a * b, 1); this.w[name] = all.subarray(off, off + n); off += n; }
    const S = this.spec, half = S.head_dim / 2;
    this.invFreq = new Float32Array(half);
    for (let i = 0; i < half; i++) this.invFreq[i] = 1 / Math.pow(S.rope_theta, (2 * i) / S.head_dim);
    this.reset();
  }
  Twin.prototype.reset = function () {
    const S = this.spec; this.pos = 0;
    this.kc = []; this.vc = [];
    for (let l = 0; l < S.layers; l++) { this.kc.push([]); this.vc.push([]); }
  };
  function matvec(W, x, rows, cols, out) {
    for (let r = 0; r < rows; r++) { let s = 0; const o = r * cols; for (let c = 0; c < cols; c++) s += W[o + c] * x[c]; out[r] = s; }
    return out;
  }
  function rmsnorm(x, w, eps, out, off = 0, n = x.length) {
    let ss = 0; for (let i = 0; i < n; i++) ss += x[off + i] * x[off + i];
    const k = 1 / Math.sqrt(ss / n + eps);
    for (let i = 0; i < n; i++) out[off + i] = x[off + i] * k * w[i];
    return out;
  }
  Twin.prototype.rope = function (v, off, pos) {
    const half = this.spec.head_dim / 2;
    for (let i = 0; i < half; i++) {
      const a = pos * this.invFreq[i], c = Math.cos(a), s = Math.sin(a), x1 = v[off + i], x2 = v[off + i + half];
      v[off + i] = x1 * c - x2 * s; v[off + i + half] = x2 * c + x1 * s;
    }
  };
  // feed one token, get the logits for the next one
  Twin.prototype.step = function (id) {
    const S = this.spec, H = S.hidden, hd = S.head_dim, nq = S.heads, nkv = S.kv_heads, F = S.ffn, w = this.w, pos = this.pos;
    const x = Float32Array.from(w['model.embed_tokens.weight'].subarray(id * H, id * H + H));
    const h = new Float32Array(H), q = new Float32Array(nq * hd), att = new Float32Array(nq * hd), o = new Float32Array(H);
    const g = new Float32Array(F), u = new Float32Array(F);
    for (let l = 0; l < S.layers; l++) {
      const p = 'model.layers.' + l + '.';
      rmsnorm(x, w[p + 'input_layernorm.weight'], S.eps, h);
      matvec(w[p + 'self_attn.q_proj.weight'], h, nq * hd, H, q);
      const k = matvec(w[p + 'self_attn.k_proj.weight'], h, nkv * hd, H, new Float32Array(nkv * hd));
      const v = matvec(w[p + 'self_attn.v_proj.weight'], h, nkv * hd, H, new Float32Array(nkv * hd));
      for (let i = 0; i < nq; i++) { rmsnorm(q, w[p + 'self_attn.q_norm.weight'], S.eps, q, i * hd, hd); this.rope(q, i * hd, pos); }
      for (let i = 0; i < nkv; i++) { rmsnorm(k, w[p + 'self_attn.k_norm.weight'], S.eps, k, i * hd, hd); this.rope(k, i * hd, pos); }
      this.kc[l].push(k); this.vc[l].push(v);
      const T = this.kc[l].length, sc = new Float32Array(T), scale = 1 / Math.sqrt(hd);
      for (let i = 0; i < nq; i++) {
        const kv = Math.floor(i / (nq / nkv)) * hd, qo = i * hd; let mx = -Infinity;
        for (let t = 0; t < T; t++) { const kt = this.kc[l][t]; let s = 0; for (let d = 0; d < hd; d++) s += q[qo + d] * kt[kv + d]; sc[t] = s * scale; if (sc[t] > mx) mx = sc[t]; }
        let z = 0; for (let t = 0; t < T; t++) { sc[t] = Math.exp(sc[t] - mx); z += sc[t]; }
        for (let d = 0; d < hd; d++) att[qo + d] = 0;
        for (let t = 0; t < T; t++) { const vt = this.vc[l][t], a = sc[t] / z; for (let d = 0; d < hd; d++) att[qo + d] += a * vt[kv + d]; }
      }
      matvec(w[p + 'self_attn.o_proj.weight'], att, H, nq * hd, o);
      for (let i = 0; i < H; i++) x[i] += o[i];
      rmsnorm(x, w[p + 'post_attention_layernorm.weight'], S.eps, h);
      matvec(w[p + 'mlp.gate_proj.weight'], h, F, H, g);
      matvec(w[p + 'mlp.up_proj.weight'], h, F, H, u);
      for (let i = 0; i < F; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
      matvec(w[p + 'mlp.down_proj.weight'], g, H, F, o);
      for (let i = 0; i < H; i++) x[i] += o[i];
    }
    rmsnorm(x, w['model.norm.weight'], S.eps, h);
    this.pos++;
    return matvec(w['model.embed_tokens.weight'], h, S.vocab, H, new Float32Array(S.vocab));  // tied: the word list is also the output
  };
  Twin.prototype.feed = function (ids) { let lg = null; for (const id of ids) lg = this.step(id); return lg; };

  // ---------- choosing the next token ----------
  function softmax(lg, temp = 1) {
    let mx = -Infinity; for (let i = 0; i < lg.length; i++) if (lg[i] > mx) mx = lg[i];
    const p = new Float32Array(lg.length); let z = 0;
    for (let i = 0; i < lg.length; i++) { p[i] = Math.exp((lg[i] - mx) / temp); z += p[i]; }
    for (let i = 0; i < lg.length; i++) p[i] /= z;
    return p;
  }
  function topk(arr, k) {
    const idx = []; for (let i = 0; i < arr.length; i++) {
      if (idx.length < k) { idx.push(i); idx.sort((a, b) => arr[b] - arr[a]); } else if (arr[i] > arr[idx[k - 1]]) { idx[k - 1] = i; idx.sort((a, b) => arr[b] - arr[a]); }
    }
    return idx;
  }
  // opts: {temp (0 = top guess), topK, repPenalty, history, rand}
  function choose(logits, opts) {
    const lg = Float32Array.from(logits);
    if (opts.repPenalty && opts.repPenalty !== 1) for (const id of new Set(opts.history)) lg[id] = lg[id] > 0 ? lg[id] / opts.repPenalty : lg[id] * opts.repPenalty;
    if (!opts.temp) { let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i; return b; }
    const keep = topk(lg, opts.topK || 40), p = softmax(Float32Array.from(keep.map((i) => lg[i])), opts.temp);
    let r = opts.rand(), acc = 0;
    for (let j = 0; j < keep.length; j++) { acc += p[j]; if (r <= acc) return keep[j]; }
    return keep[keep.length - 1];
  }
  function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }

  const api = { Tokenizer, Twin, softmax, topk, choose, rng };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Twins = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
