// Scenes driven by a real model. tools/gpt2_attention.py runs GPT-2 and writes data/real/*.json;
// nothing here is hand-picked except the prompts.
import { C, cell, bar, edge, plate, item, tag } from './kit.js';
import { add } from '../lib/vec.js';

const load = async (name) => {
  try { const r = await fetch(new URL(`../../data/real/${name}.json`, import.meta.url)); return r.ok ? await r.json() : null; }
  catch { return null; }
};
const [ATTN, INJ] = await Promise.all([load('real-attention'), load('prompt-injection')]);

const pct = (p) => `${p >= 0.1 ? Math.round(p * 100) : p >= 0.001 ? (p * 100).toFixed(1) : (p * 100).toPrecision(2)}%`;
const show = (t) => t.replace(/\n/g, '↵').trim() || '·';
const RH = 1.25;

function layout(tokens, maxW = 10.5) {
  const disp = tokens.map(show), w = disp.map((d) => Math.min(2.4, 0.17 * d.length + 0.42));
  const pos = [], rows = [[]];
  let x = 0;
  w.forEach((wi, i) => { if (x + wi > maxW && rows.at(-1).length) { rows.push([]); x = 0; } rows.at(-1).push(i); pos[i] = x + wi / 2; x += wi + 0.14; });
  const width = Math.max(...rows.map((r) => pos[r.at(-1)] + w[r.at(-1)] / 2));
  rows.forEach((r, ri) => r.forEach((i) => { pos[i] = [pos[i] - width / 2, -ri * RH, 0]; }));
  return { disp, w, pos, rows: rows.length };
}

function sceneKit(d) {
  const L = layout(d.tokens), n = d.tokens.length, q = d.query, span = new Set(d.span || []);
  const big = n <= 14;
  const toks = (hl = {}) => d.tokens.map((_, i) => item(`t${i}`, cell(L.pos[i], [L.w[i], 0.5, 0.5], 0.35), hl[i] || (i === q ? C.pink : span.has(i) ? C.red : C.cyan),
    { glow: hl[i] ? 1.3 : span.has(i) ? 1.1 : 0.85, label: `token ${i}: "${L.disp[i]}"`, info: span.has(i) ? 'part of the injected line' : i === q ? 'the query token' : 'context token' }));
  const words = () => d.tokens.map((_, i) => tag(`w${i}`, add(L.pos[i], [0, 0, 0.36]), L.disp[i], { big, color: span.has(i) ? [1, 0.75, 0.72] : undefined }));
  const links = (wts) => {
    const top = Math.max(1e-9, ...wts.slice(0, q)), items = [], tags = [];
    const order = wts.map((v, j) => [v, j]).filter(([, j]) => j < q).sort((a, b) => b[0] - a[0]);
    const from = add(L.pos[q], [0, 0.3, 0]);
    for (const [v, j] of order) {
      if (v < 0.004) continue;
      items.push(item(`l${j}`, edge(from, add(L.pos[j], [0, 0.3, 0]), { trim: 0.12 }), span.has(j) ? C.red : C.white, { glow: 0.05 + 1.5 * (v / top) ** 1.2 }));
    }
    order.slice(0, 3).forEach(([v, j]) => tags.push(tag(`pc${j}`, add(L.pos[j], [0, 0.5, 0]), pct(v), { color: C.amber })));
    return { items, tags };
  };
  const base = -L.rows * RH - 4.0;
  const nextBars = () => {
    const pmax = d.next[0].p, items = [], tags = [];
    d.next.slice(0, 5).forEach((t, i) => {
      const x = (i - 2) * 1.5, h = 2.6 * (t.p / pmax);
      items.push(item(`nb${i}`, bar(x, h, { base, w: 0.8 }), i ? C.amber : C.green, { label: `"${show(t.token)}"`, info: `GPT-2 probability ${pct(t.p)}` }));
      tags.push(tag(`nw${i}`, [x, base - 0.35, 0.4], `"${show(t.token)}"`, { big: !i }), tag(`np${i}`, [x, base + h + 0.3, 0], pct(t.p), { color: C.dim }));
    });
    tags.push(tag('nT', [-4.6, base + 3.2, 0], 'next token', { color: C.green }));
    return { items, tags };
  };
  return { L, n, q, span, toks, words, links, nextBars, base };
}

function realAttention() {
  const d = ATTN, K = sceneKit(d), { q } = K, qt = `"${show(d.tokens[q])}"`, out = [];
  const frame = (caption, line, extra = {}) => out.push({ caption, line, items: [...K.toks(extra.hl), ...(extra.items || [])], tags: [...K.words(), ...(extra.tags || [])] });
  frame(`Real GPT-2 small (124M parameters). This is how it splits the sentence into ${K.n} tokens. We'll follow ${qt}.`, 0);
  let A = K.links(d.mean_by_layer[0]);
  frame(`Layer 1, averaged over ${d.heads} heads: where does ${qt} look? ${pct(d.sink_by_layer[0])} of it goes to the very first token.`, 1, A);
  const avgSink = d.sink_by_layer.reduce((a, b) => a + b, 0) / d.layers, mid = Math.floor(d.layers / 2);
  A = K.links(d.mean_by_layer[mid]);
  frame(`That pull to the first token is an "attention sink": ${pct(avgSink)} on average across layers, a parking spot when nothing else is useful. Next: remove it and renormalize.`, 1, { ...A, hl: { 0: C.amber } });
  for (let l = 0; l < d.layers; l++) {
    const w = d.clean_by_layer[l], j = w.slice(0, q).reduce((b, v, k) => (v > w[b] ? k : b), 1);
    A = K.links(w);
    frame(`Layer ${l + 1} (sink removed): ${qt} looks most at "${show(d.tokens[j])}" (${pct(w[j])}).`, 2, A);
  }
  const B = d.best_head;
  A = K.links(B.weights);
  frame(`Strongest single link: layer ${B.layer + 1}, head ${B.head + 1} sends ${pct(B.weight)} of ${qt}'s attention to "${show(d.tokens[B.target])}".`, 3, { ...A, hl: { [B.target]: C.green } });
  const N = K.nextBars();
  frame(`And GPT-2's real next-token guesses after the full sentence. Top pick: "${show(d.next[0].token)}" (${pct(d.next[0].p)}).`, 4, N);
  return out;
}

function promptInjection() {
  const d = INJ, K = sceneKit(d), { q } = K, out = [];
  const frame = (caption, line, extra = {}) => out.push({ caption, line, items: [...K.toks(extra.hl), ...(extra.items || [])], tags: [...K.words(), ...(extra.tags || [])] });
  const chart = (upTo) => {
    const items = [], tags = [], s = d.span_share_by_layer, x0 = -((d.layers - 1) / 2) * 0.62, by = K.base + 0.2;
    for (let l = 0; l <= upTo; l++) {
      const x = x0 + l * 0.62;
      items.push(item(`sb${l}`, bar(x, 0.1 + s[l] * 4.5, { base: by, w: 0.42 }), C.red, { glow: l === upTo ? 1.3 : 0.75, label: `layer ${l + 1}`, info: `${pct(s[l])} of the attention on the injected line` }));
      tags.push(tag(`sl${l}`, [x, by - 0.3, 0], String(l + 1), { color: C.dim }));
    }
    tags.push(tag('cT', [x0 - 0.4, by + 3.4, 0], `share of attention on the injected line, by layer`, { color: C.red }), tag('cv', [x0 + upTo * 0.62, by + 0.25 + s[upTo] * 4.5 + 0.3, 0], pct(s[upTo]), { color: C.white }));
    return { items, tags };
  };
  const both = (a, b) => ({ items: [...a.items, ...b.items], tags: [...a.tags, ...b.tags] });
  frame(`A summarization prompt with an instruction hidden inside the email (red): ${d.span.length} of ${K.n} tokens.`, 0);
  frame(`The summary starts at the last token, "${show(d.tokens[q])}". GPT-2 decides what to write from here.`, 1, { hl: { [q]: C.pink } });
  for (let l = 0; l < d.layers; l++) {
    frame(`Layer ${l + 1}: ${pct(d.span_share_by_layer[l])} of its attention (sink removed) lands on the injected line, which is ${pct(d.span_fraction)} of the text.`, 2, both(K.links(d.clean_by_layer[l]), chart(l)));
  }
  const S = d.span_head;
  frame(`Hungriest head: layer ${S.layer + 1}, head ${S.head + 1} puts ${pct(S.share)} of its attention on the injected line.`, 3, both(K.links(S.weights), chart(d.layers - 1)));
  const P = d.probe, N = K.nextBars();
  frame(`GPT-2's top next token: "${show(d.next[0].token)}" (${pct(d.next[0].p)}).${P ? ` Chance it starts "${P.text.trim()}": ${pct(P.p)}, rank ${P.rank.toLocaleString()} of ${P.vocab.toLocaleString()}.` : ''}`, 4, N);
  frame('GPT-2 isn’t instruction-tuned, so this shows mechanics, not a jailbreak: the injected line is part of what the model reads as it forms the answer. That is why untrusted text needs its own channel.', -1, N);
  return out;
}

const missing = () => [{ caption: 'No GPT-2 data yet. Run: python tools/gpt2_attention.py  (see README), then refresh.', line: -1,
  items: [item('ph', plate([0, 0, 0], [6, 0.1, 1.2]), C.dim)], tags: [tag('ph', [0, 0.6, 0], 'run tools/gpt2_attention.py', { color: C.dim, big: true })] }];

const still = { still: true, tilt: 4, viewYaw: 0 };
export const REAL = [
  { name: 'Real GPT-2 Attention', category: 'Real Model', color: C.green, fact: 'actual attention weights from GPT-2 small', ...still, frames: ATTN ? realAttention : missing,
    code: ['out = gpt2(tokens, output_attentions=True)', 'w = out.attentions[layer][0, :, query].mean(0)', 'w = w[1:] / w[1:].sum()      # drop the attention sink', 'best = argmax over (layer, head, token)', 'p = softmax(out.logits[0, -1])'] },
  { name: 'Prompt Injection', category: 'Real Model', color: C.red, fact: 'how much GPT-2 attends to a hidden instruction', ...still, frames: INJ ? promptInjection : missing,
    code: ['span = tokens of the injected line', 'w = attention of the last token, sink removed', 'share = w[span].sum()          # per layer', 'head = argmax share over all heads', 'p("HACKED") = softmax(logits)[id]'] },
];
