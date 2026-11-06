// Transformers, from text to the next token. Numbers are hand-picked to be plausible and easy to read;
// they illustrate the mechanism, they are not taken from a real model.
import { C, cell, bar, arc, edge, plate, item, tag } from './kit.js';
import { polyline } from '../lib/sampling.js';
import { add } from '../lib/vec.js';

const softmax = (s, T = 1) => { const m = Math.max(...s), e = s.map((v) => Math.exp((v - m) / T)), z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); };
const pct = (p) => `${Math.round(p * 100)}%`;
const val = (k, d) => { const s = Math.sin(k * 12.9898 + d * 78.233) * 43758.5453; return (s - Math.floor(s)) * 2 - 1; };
const merge = (shapes) => ({
  parts: shapes.map((s) => s.parts),
  lo: [0, 1, 2].map((k) => Math.min(...shapes.map((s) => s.lo[k]))), hi: [0, 1, 2].map((k) => Math.max(...shapes.map((s) => s.hi[k]))),
});

// ------------------------------------------------------------------ tokens -> embeddings
function embeddings() {
  const out = [], toks = ['un', 'believ', 'ably', ' good'], ids = [403, 12177, 1598, 1695], rows = [1, 3, 5, 6], D = 6, S = 0.4;
  const widths = toks.map((t) => 0.3 * t.trim().length + 0.45);
  const layout = (gap) => { let x = 0; const c = widths.map((w) => { const cx = x + w / 2; x += w + gap; return cx; }); const mid = (x - gap) / 2; return c.map((v) => v - mid); };
  const tight = layout(0.03), apart = layout(0.55);
  const tokItems = (xs, col = C.cyan) => toks.map((t, k) => item(`t${k}`, cell([xs[k], 0, 0], [widths[k], 0.6, 0.6]), col, { label: `token "${t}"`, info: `id ${ids[k]}` }));
  const tokTags = (xs) => toks.map((t, k) => tag(`t${k}`, [xs[k], 0, 0.4], `"${t}"`, { big: true }));
  const idTags = (xs) => ids.map((v, k) => tag(`id${k}`, [xs[k], -0.6, 0], `id ${v}`, { color: C.amber }));
  const table = (lift = -1) => {
    const it = [], tg = [tag('tbl', [-1.65, 6.85, 0], 'embedding table · one row per token id', { color: C.dim })];
    for (let r = 0; r < 8; r++) for (let d = 0; d < D; d++) {
      const k = rows.indexOf(r);
      if (k >= 0 && k <= lift) continue;
      const v = k >= 0 ? val(k, d) : val(r + 9, d);
      it.push(item(k >= 0 ? `e${k}.${d}` : `bg${r}.${d}`, cell([(d - 2.5) * S, 6.4 - r * S, 0], [0.34, 0.34, 0.34], 0.5), k >= 0 ? (v > 0 ? C.cyan : C.pink) : C.dim, { glow: k >= 0 ? 0.4 + 0.9 * Math.abs(v) : 0.45 }));
    }
    rows.forEach((r, k) => { if (k > lift) tg.push(tag(`row${k}`, [-1.55, 6.4 - r * S, 0], `${ids[k]} →`, { color: C.amber })); });
    return { it, tg };
  };
  const vec = (k, xs, dy = 0) => Array.from({ length: D }, (_, d) => { const v = val(k, d) + dy * val(k + 20, d) * 0.35; return item(`e${k}.${d}`, cell([xs[k], 0.75 + d * S, 0], [0.34, 0.34, 0.34], 0.5), v > 0 ? C.cyan : C.pink, { glow: 0.4 + 0.9 * Math.min(1, Math.abs(v)) }); });
  const pe = (k, xs, onto = false) => Array.from({ length: D }, (_, d) => item(`p${k}.${d}`, cell([xs[k] + (onto ? 0 : 0.48), 0.75 + d * S, 0], [0.22, 0.22, 0.22], 0.6), C.violet, { glow: 0.5 + 0.5 * Math.abs(Math.sin(k + d * 0.9)) }));

  out.push({ caption: 'A model never sees letters. First the text has to be cut into tokens.', line: -1, items: tokItems(tight), tags: [tag('str', [0, 0, 0.4], '"unbelievably good"', { big: true })] });
  out.push({ caption: 'The tokenizer splits it into sub-word pieces it learned from data (BPE).', line: 0, items: tokItems(apart), tags: tokTags(apart) });
  out.push({ caption: 'Each piece maps to an integer id in a fixed vocabulary (ids here are illustrative).', line: 1, items: tokItems(apart), tags: [...tokTags(apart), ...idTags(apart)] });
  let T = table();
  out.push({ caption: 'The embedding table has one learned row of numbers per id. Find each token’s row.', line: 2, items: [...tokItems(apart), ...T.it], tags: [...tokTags(apart), ...idTags(apart), ...T.tg] });
  T = table(3);
  out.push({ caption: 'Copy those rows out. Each token is now a vector (6 numbers here, ~4,000 in a real LLM).', line: 2, items: [...tokItems(apart), ...T.it, ...toks.flatMap((_, k) => vec(k, apart))], tags: [...tokTags(apart), ...T.tg] });
  out.push({ caption: 'Vectors carry meaning but not order. Build a position signal for slot 0, 1, 2, 3…', line: 3, items: [...tokItems(apart), ...T.it, ...toks.flatMap((_, k) => [...vec(k, apart), ...pe(k, apart)])], tags: [...tokTags(apart), ...T.tg, ...toks.map((_, k) => tag(`pos${k}`, [apart[k] + 0.48, 3.4, 0], `pos ${k}`, { color: C.violet }))] });
  out.push({ caption: '…and add it in. Same word, different position, slightly different vector.', line: 3, items: [...tokItems(apart), ...T.it, ...toks.flatMap((_, k) => [...vec(k, apart, 1), ...pe(k, apart, true)])], tags: [...tokTags(apart), ...T.tg] });
  out.push({ caption: 'One vector per token. This list of vectors is what every transformer layer reads and rewrites.', line: 4, items: [...tokItems(apart), ...T.it, ...toks.flatMap((_, k) => vec(k, apart, 1))], tags: [...tokTags(apart), ...T.tg] });
  return out;
}

// ------------------------------------------------------------------ self-attention
const WORDS = ['The', 'cat', 'sat', 'because', 'it', 'was', 'tired'];
const SCORES = [
  [2.0, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1],
  [1.2, 2.0, 0.3, 0.1, 0.2, 0.1, 0.3],
  [0.3, 1.9, 1.0, 0.2, 0.1, 0.2, 0.4],
  [0.1, 0.6, 1.2, 1.4, 0.3, 0.2, 0.3],
  [0.1, 2.6, 0.4, 0.2, 1.1, 0.3, 0.9],
  [0.1, 0.8, 0.3, 0.2, 1.6, 1.0, 0.5],
  [0.1, 1.7, 0.3, 0.3, 1.2, 0.8, 1.0],
];
function attention() {
  const out = [], n = WORDS.length, X = (i) => (i - (n - 1) / 2) * 1.4, QKV = 0.36;
  const toks = (hl = -1) => WORDS.map((w, i) => item(`t${i}`, cell([X(i), 0, 0], [1.2, 0.55, 0.55]), i === hl ? C.amber : C.cyan, { label: `"${w}"`, info: i === hl ? 'the query token' : 'a token in the context' }));
  const words = () => WORDS.map((w, i) => tag(`w${i}`, [X(i), 0, 0.38], w, { big: true }));
  const qkv = (glowK = null, glowV = null, q = -1) => WORDS.flatMap((_, i) => [
    item(`q${i}`, cell([X(i) - 0.42, 0.75, 0], [QKV, QKV, QKV], 0.5), C.pink, { glow: q < 0 || q === i ? 1 : 0.35, label: `query of "${WORDS[i]}"`, info: 'what this token is looking for' }),
    item(`k${i}`, cell([X(i), 0.75, 0], [QKV, QKV, QKV], 0.5), C.amber, { glow: glowK ? glowK[i] : 1, label: `key of "${WORDS[i]}"`, info: 'what this token offers to be found by' }),
    item(`v${i}`, cell([X(i) + 0.42, 0.75, 0], [QKV, QKV, QKV], 0.5), C.cyan, { glow: glowV ? glowV[i] : 1, label: `value of "${WORDS[i]}"`, info: 'what this token passes on if attended to' }),
  ]);
  const arcs = (q, w, showPct) => {
    const wmax = Math.max(...w), items = [], tags = [];
    WORDS.forEach((_, j) => {
      const a = [X(q) - 0.42, 0.95, 0], b = [X(j), 0.95, 0], h = 0.35 + Math.abs(X(j) - X(q)) * 0.22;
      items.push(item(`arc${j}`, arc(a, b, h), C.white, { glow: 0.12 + 1.3 * (w[j] / wmax) ** 1.3 }));
      const mid = [(a[0] + b[0]) / 2, 0.95 + h + 0.22, 0];
      tags.push(tag(`s${j}`, mid, showPct ? pct(w[j]) : w[j].toFixed(1), { color: w[j] === wmax ? C.amber : C.dim }));
    });
    return { items, tags };
  };
  out.push({ caption: 'Self-attention lets every token look at every other token and decide what matters.', line: -1, items: toks(), tags: words() });
  out.push({ caption: 'Each token vector is multiplied by three learned matrices: a Query, a Key and a Value.', line: 0, items: [...toks(), ...qkv()], tags: [...words(), tag('qkvL', [X(0) - 1.2, 0.75, 0], 'Q K V', { color: C.dim })] });
  const it = 4;
  let A = arcs(it, SCORES[it], false);
  out.push({ caption: 'What does "it" refer to? Compare its query with every key (dot product) to get a score.', line: 1, items: [...toks(it), ...qkv(null, null, it), ...A.items], tags: [...words(), ...A.tags] });
  const w = softmax(SCORES[it].map((s) => s * 1.4));
  A = arcs(it, w, true);
  out.push({ caption: `Softmax turns scores into weights that sum to 100%. "it" attends mostly to "cat" (${pct(w[1])}).`, line: 2, items: [...toks(it), ...qkv(null, null, it), ...A.items], tags: [...words(), ...A.tags] });
  const outPos = [X(it), 2.9, 0];
  out.push({ caption: 'The new "it" is the weighted sum of all values. It now carries information about the cat.', line: 3,
    items: [...toks(it), ...qkv(null, w.map((x) => 0.25 + 3 * x), it), ...A.items, item('out', cell(outPos, [0.5, 0.5, 0.5], 0.6), C.green, { glow: 1.3, label: 'output for "it"', info: 'Σ weight × value' })],
    tags: [...words(), ...A.tags, tag('outT', add(outPos, [0, 0.5, 0]), `${pct(w[1])}·cat + ${pct(w[4])}·it + …`, { color: C.green })] });
  const sat = 2, w2 = softmax(SCORES[sat].map((s) => s * 1.4));
  A = arcs(sat, w2, true);
  out.push({ caption: 'Every token does this at the same time. Here "sat" asks who did the sitting, and finds "cat".', line: 2, items: [...toks(sat), ...qkv(null, null, sat), ...A.items], tags: [...words(), ...A.tags] });
  // full attention matrix
  const M = SCORES.map((r) => softmax(r.map((s) => s * 1.4)));
  const grid = (mask) => {
    const items = [], tags = [], G = 0.5, x0 = -((n - 1) / 2) * G, y0 = 5.4;
    M.forEach((row, i) => {
      const r = mask ? softmax(row.map((_, j) => (j > i ? -1e9 : SCORES[i][j] * 1.4))) : row;
      r.forEach((p, j) => items.push(item(`m${i}.${j}`, cell([x0 + j * G, y0 - i * G, 0], [0.42, 0.42, 0.42], 0.7), j > i && mask ? C.dim : C.amber, { glow: mask && j > i ? 0.15 : 0.15 + 2.2 * p })));
      tags.push(tag(`mr${i}`, [x0 - 0.9, y0 - i * G, 0], WORDS[i], { color: C.dim }));
    });
    WORDS.forEach((wd, j) => tags.push(tag(`mc${j}`, [x0 + j * G, y0 + 0.5, 0], wd.slice(0, 3), { color: C.dim })));
    tags.push(tag('mT', [0, y0 + 1.0, 0], mask ? 'causal attention: each row only sees the past' : 'attention weights: row = query, column = key', { color: C.white }));
    return { items, tags };
  };
  let Gm = grid(false);
  out.push({ caption: 'Stack every token’s weights and you get the attention matrix. Bright = strong attention.', line: 4, items: [...toks(), ...qkv(), ...Gm.items], tags: [...words(), ...Gm.tags] });
  Gm = grid(true);
  out.push({ caption: 'GPT-style models mask the future: a token may only attend to itself and tokens before it.', line: 5, items: [...toks(), ...qkv(), ...Gm.items], tags: [...words(), ...Gm.tags] });
  out.push({ caption: 'Real models run many of these heads in parallel, each learning a different kind of relation.', line: -1, items: [...toks(), ...qkv(), ...Gm.items], tags: [...words(), ...Gm.tags] });
  return out;
}

// ------------------------------------------------------------------ next-token prediction
function nextToken() {
  const out = [], X = (i) => -3.6 + i * 1.3, H = 4.2, bx = (i) => -2.2 + i * 1.1, BASE = 3.0;
  let seq = ['The', 'cat', 'sat', 'on', 'the'];
  const row = (hl = -1) => seq.map((w, i) => item(`t${i}`, cell([X(i), 0, 0], [1.1, 0.55, 0.55]), i === hl ? C.green : C.cyan));
  const words = () => seq.map((w, i) => tag(`t${i}`, [X(i), 0, 0.38], w, { big: true }));
  const model = (on = 0.6) => [item('model', plate([0.3, 1.5, 0], [9.2, 0.7, 1.3], 0.5), C.violet, { glow: on, label: 'transformer', info: 'a stack of attention + feed-forward layers' }),
    ...seq.map((_, i) => item(`up${i}`, edge([X(i), 0.3, 0], [X(i), 1.15, 0], { trim: 0, arrow: true }), C.white, { glow: on * 0.8 }))];
  const bars = (cands, p, pick = -1) => {
    const items = [item('head', edge([X(seq.length - 1), 1.85, 0], [0, BASE - 0.15, 0], { trim: 0, arrow: true }), C.white, { glow: 0.7 })], tags = [];
    cands.forEach((w, i) => {
      items.push(item(`bar${i}`, bar(bx(i), p[i] * H, { base: BASE, w: 0.7 }), i === pick ? C.green : C.amber, { glow: i === pick ? 1.3 : 0.9, label: `"${w}"`, info: `probability ${pct(p[i])}` }));
      tags.push(tag(`bw${i}`, [bx(i), BASE - 0.3, 0.4], w === '.' ? '"."' : w, { big: i === pick }), tag(`bp${i}`, [bx(i), BASE + p[i] * H + 0.3, 0], pct(p[i]), { color: C.dim }));
    });
    return { items, tags };
  };
  const tagModel = tag('mLbl', [0.3, 1.5, 0.75], 'transformer layers', { color: C.violet });
  out.push({ caption: 'A language model does one thing: given the text so far, predict the next token.', line: -1, items: row(), tags: words() });
  out.push({ caption: 'The whole sequence goes through the stack of layers (attention + feed-forward, repeated).', line: 0, items: [...row(), ...model(1.1)], tags: [...words(), tagModel] });
  const c1 = ['mat', 'floor', 'couch', 'bed', 'roof'], s1 = [2.0, 0.85, 0.45, 0.1, -0.4];
  let B = bars(c1, softmax(s1));
  out.push({ caption: 'The last position’s vector becomes a score for every word in the vocabulary. Softmax → probabilities.', line: 1, items: [...row(), ...model(), ...B.items], tags: [...words(), tagModel, ...B.tags] });
  B = bars(c1, softmax(s1, 2.2));
  out.push({ caption: 'Temperature 2.2 flattens the distribution: more surprising, more random picks.', line: 2, items: [...row(), ...model(), ...B.items], tags: [...words(), tagModel, ...B.tags, tag('temp', [3.6, BASE + 2.3, 0], 'T = 2.2', { color: C.pink, big: true })] });
  B = bars(c1, softmax(s1, 0.4));
  out.push({ caption: 'Temperature 0.4 sharpens it: safe, predictable picks.', line: 2, items: [...row(), ...model(), ...B.items], tags: [...words(), tagModel, ...B.tags, tag('temp', [3.6, BASE + 2.3, 0], 'T = 0.4', { color: C.pink, big: true })] });
  B = bars(c1, softmax(s1), 0);
  const p1 = softmax(s1);
  out.push({ caption: `Sample from it (T = 1). "mat" wins with ${pct(p1[0])}.`, line: 3,
    items: [...row(), ...model(), ...B.items, item(`t${seq.length}`, cell([bx(0), BASE + p1[0] * H + 0.75, 0], [1.1, 0.55, 0.55]), C.green)],
    tags: [...words(), tagModel, ...B.tags, tag(`t${seq.length}`, [bx(0), BASE + p1[0] * H + 0.75, 0.38], 'mat', { big: true })] });
  seq = [...seq, 'mat'];
  out.push({ caption: 'Append it to the input…', line: 4, items: [...row(seq.length - 1), ...model(0.4)], tags: [...words(), tagModel] });
  const c2 = ['.', 'and', ',', 'while', 'all'], s2 = [1.6, 0.8, 0.4, 0.0, -0.6], p2 = softmax(s2);
  B = bars(c2, p2);
  out.push({ caption: '…and run the whole thing again. New context, new distribution.', line: 1, items: [...row(), ...model(1.1), ...B.items], tags: [...words(), tagModel, ...B.tags] });
  B = bars(c2, p2, 0);
  out.push({ caption: `"." wins with ${pct(p2[0])}.`, line: 3,
    items: [...row(), ...model(), ...B.items, item(`t${seq.length}`, cell([bx(0), BASE + p2[0] * H + 0.75, 0], [1.1, 0.55, 0.55]), C.green)],
    tags: [...words(), tagModel, ...B.tags, tag(`t${seq.length}`, [bx(0), BASE + p2[0] * H + 0.75, 0.38], '"."', { big: true })] });
  seq = [...seq, '.'];
  out.push({ caption: 'That loop, one token at a time, is all text generation is. Each step re-reads everything before it.', line: 4, items: [...row(seq.length - 1), ...model(0.4)], tags: [...words(), tagModel] });
  return out;
}

// ------------------------------------------------------------------ the transformer block, exploded
function block() {
  const cols = (y, k) => merge(Array.from({ length: k }, (_, i) => cell([(i - (k - 1) / 2) * 0.7, y, 0], [0.32, 0.9, 0.32], 0.4)));
  const wave = (() => { const pts = []; for (let i = 0; i <= 60; i++) { const x = -1.6 + (3.2 * i) / 60; pts.push([x, 0.95 + 0.18 * Math.sin(x * 4), 0]); } return { parts: polyline(pts, 0.01, 6), lo: [-1.6, 0.75, -0.1], hi: [1.6, 1.15, 0.1] }; })();
  const resid = merge([cell([-2.9, 3.1, 0], [0.12, 6.2, 0.12], 1), edge([-2.9, 2.0, 0], [-1.6, 2.0, 0], { trim: 0 }), edge([-2.9, 4.6, 0], [-2.7, 4.6, 0], { trim: 0 })]);
  const items = [
    item('in', cols(0, 5), C.cyan, { label: 'Input vectors', info: 'one vector per token (embedding + position)', ex: [0, -1.4, 0], stage: 0 }),
    item('pos', wave, C.violet, { label: 'Positional encoding', info: 'adds where each token sits in the sequence', ex: [2.4, -0.6, 0], stage: 0.1 }),
    item('qkv', merge([plate([-0.9, 1.75, 0], [0.7, 0.12, 0.9]), plate([0, 1.75, 0], [0.7, 0.12, 0.9]), plate([0.9, 1.75, 0], [0.7, 0.12, 0.9])]), C.pink, { label: 'Q, K, V projections', info: 'three learned matrices make queries, keys and values', ex: [0, -0.6, 0], stage: 0.15 }),
    ...[0, 1, 2, 3].map((h) => item(`head${h}`, plate([0, 2.45, (h - 1.5) * 0.55], [2.4, 0.08, 0.42], 0.7), C.amber, { label: h === 0 ? 'Attention heads' : `Head ${h + 1}`, info: 'each head learns its own pattern of who-looks-at-whom', ex: [(h - 1.5) * 1.6, 0.2, (h - 1.5) * 1.2], stage: 0.25, glow: 0.8 + 0.1 * h })),
    item('proj', plate([0, 3.0, 0], [2.4, 0.1, 1.0]), C.amber, { label: 'Output projection', info: 'merges all heads back into one vector per token', ex: [0, 0.5, 0], stage: 0.3 }),
    item('norm1', plate([0, 3.5, 0], [3.0, 0.05, 1.2], 0.6), C.white, { label: 'Add & LayerNorm', info: 'adds the input back (residual) and rescales', ex: [0, 0.9, 0], stage: 0.35, glow: 0.7 }),
    item('ffn1', plate([0, 4.15, 0], [5.0, 0.14, 1.2]), C.green, { label: 'Feed-forward: expand', info: '4× wider layer, where most of the parameters live', ex: [0, 1.4, 0], stage: 0.4 }),
    item('ffn2', plate([0, 4.75, 0], [2.4, 0.14, 1.0]), C.green, { label: 'Feed-forward: project back', info: 'squeezes it back to the model width', ex: [0, 1.9, 0], stage: 0.45 }),
    item('norm2', plate([0, 5.25, 0], [3.0, 0.05, 1.2], 0.6), C.white, { label: 'Add & LayerNorm 2', info: 'second residual connection', ex: [0, 2.3, 0], stage: 0.5, glow: 0.7 }),
    item('res', resid, C.blue, { label: 'Residual stream', info: 'the highway every layer reads from and writes back to', ex: [-1.8, 0, 0], stage: 0.2 }),
    item('out', cols(6.1, 5), C.cyan, { label: 'Output vectors', info: 'same shape as the input, now context-aware; feeds the next block', ex: [0, 3.1, 0], stage: 0.55 }),
  ];
  return [{ caption: 'One transformer block. Open your hand to pull it apart, point at a part to learn what it does. GPT-scale models stack ~100 of these.', line: -1, items, tags: [] }];
}

export const TRANSFORMERS = [
  { name: 'Tokens → Embeddings', category: 'Transformers', color: C.violet, fact: 'text → tokens → ids → vectors', still: true, tilt: 6, viewYaw: 0, frames: embeddings,
    code: ['tokens = tokenizer(text)', 'ids = [vocab[t] for t in tokens]', 'x = E[ids]            # embedding lookup', 'x = x + P[0:len(ids)]  # add positions', '# x: (seq_len, d_model)'] },
  { name: 'Self-Attention', category: 'Transformers', color: C.violet, fact: 'softmax(QKᵀ / √d) · V', still: true, tilt: 6, viewYaw: 0, frames: attention,
    code: ['Q, K, V = x @ Wq, x @ Wk, x @ Wv', 'scores = Q @ K.T / sqrt(d)', 'weights = softmax(scores)', 'out = weights @ V', '# one row of weights per token', 'scores[future] = -inf   # causal mask'] },
  { name: 'Next-Token Prediction', category: 'Transformers', color: C.violet, fact: 'logits → softmax → sample → repeat', still: true, tilt: 6, viewYaw: 0, frames: nextToken,
    code: ['h = transformer(tokens)', 'logits = h[-1] @ W_vocab', 'p = softmax(logits / T)', 'next = sample(p)', 'tokens.append(next)'] },
  { name: 'Transformer Block', category: 'Transformers', color: C.violet, fact: 'attention + feed-forward + residuals', still: false, tilt: 14, viewYaw: 0.5, frames: block,
    code: null },
];
