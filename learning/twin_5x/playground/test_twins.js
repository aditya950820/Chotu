// Checks twins.js against Python: same token ids as the real tokenizer, same predictions as PyTorch.
//   node learning/twin_5x/playground/test_twins.js <folder made by export.py, plus tok_tests.json>
const fs = require('fs'), path = require('path'), T = require('./twins.js');
const dir = process.argv[2];
const tok = new T.Tokenizer(JSON.parse(fs.readFileSync(path.join(dir, 'tokenizer.json'))));
let bad = 0;
const tests = JSON.parse(fs.readFileSync(path.join(dir, 'tok_tests.json')));
for (const t of tests) {
  const got = tok.encode(t.text);
  if (JSON.stringify(got) !== JSON.stringify(t.ids)) { bad++; console.log('TOKENIZER MISMATCH', JSON.stringify(t.text.slice(0, 60)), got.slice(0, 12), t.ids.slice(0, 12)); }
  if (tok.decode(t.ids) !== t.text) console.log('decode differs (ok if only spacing):', JSON.stringify(t.text.slice(0, 40)));
}
console.log(`tokenizer: ${tests.length - bad}/${tests.length} match`);
const man = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json'))), chk = JSON.parse(fs.readFileSync(path.join(dir, 'check.json')));
for (const key of ['twin140', 'twin700']) {
  const buf = fs.readFileSync(path.join(dir, key + '.bin'));
  const m = new T.Twin(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), man);
  let worst = 0, lg;
  const t0 = Date.now();
  chk.ids.forEach((id, i) => { lg = m.step(id); let s = 0; for (const v of lg) s += Math.abs(v); worst = Math.max(worst, Math.abs(s - chk.models[key].logit_sum_abs[i]) / chk.models[key].logit_sum_abs[i]); });
  const top = T.topk(lg, 5), ref = chk.models[key];
  const same = JSON.stringify(top) === JSON.stringify(ref.last_top5);
  const vdiff = Math.max(...top.map((id, j) => Math.abs(lg[id] - ref.last_logits_top5[j])));
  if (!same || worst > 1e-3 || vdiff > 1e-2) bad++;
  console.log(`${key}: top-5 ${same ? 'same' : 'DIFFERENT'} (${top.map((i) => JSON.stringify(tok.piece(i))).join(' ')}), max logit diff ${vdiff.toExponential(1)}, worst row drift ${worst.toExponential(1)}, ${((Date.now() - t0) / chk.ids.length).toFixed(1)} ms/token`);
}
process.exit(bad ? 1 : 0);
