// Turns frame `step` of a scene into GPU buffers. Each item id owns a fixed slice of particles in every
// frame, sampled with the same per-id random stream, so an item that only moves keeps every particle on
// the same relative spot and glides as one piece. Slices of absent items, plus a reserve, rest as dust.
import { makeRng } from '../lib/rng.js';
import { flatten, allocate } from '../lib/sampling.js';

const DUST = 0.12;                       // share of particles that always stay dust
const DUST_TINT = [0.05, 0.07, 0.1];
const FIT_RADIUS = 1.18;
const memo = new WeakMap();

function hashStr(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function hash01(i, k) {
  let h = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(k + 1, 0xc2b2ae35);
  h ^= h >>> 13; h = Math.imul(h, 0x27d4eb2f); h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}
/** Fixed dust spot for particle i: a loose shell just outside the scene. */
function dustAt(i, out, j) {
  const u = hash01(i, 1) * 2 - 1, a = hash01(i, 2) * Math.PI * 2, r = 1.02 + 0.3 * hash01(i, 3), s = Math.sqrt(1 - u * u);
  out[j] = r * s * Math.cos(a); out[j + 1] = r * u * 0.75; out[j + 2] = r * s * Math.sin(a);
}

export function prepare(def) {
  let P = memo.get(def);
  if (P) return P;
  const frames = def.frames();
  const ids = new Map();
  const corners = [];
  for (const f of frames) {
    for (const it of f.items) {
      const w = flatten(it.shape.parts).reduce((a, p) => a + p.weight, 0) * (it.weight ?? 1);
      const e = ids.get(it.id);
      if (!e) ids.set(it.id, { id: it.id, w }); else e.w = Math.max(e.w, w);
      const { lo, hi } = it.shape;
      for (const x of [lo[0], hi[0]]) for (const y of [lo[1], hi[1]]) for (const z of [lo[2], hi[2]]) corners.push([x, y, z]);
    }
    for (const t of f.tags || []) corners.push(t.at);
  }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of corners) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
  const center = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
  let r = 1e-6;
  for (const p of corners) r = Math.max(r, Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]));
  const explodable = frames.some((f) => f.items.some((it) => it.ex));
  P = { frames, ids: [...ids.values()], center, scale: 0.9 / r, explodable };
  memo.set(def, P);
  return P;
}

export const sceneSteps = (def) => prepare(def).frames.length;

export function buildScene(def, step, n) {
  const t0 = performance.now();
  const P = prepare(def), { center, scale } = P;
  step = Math.max(0, Math.min(P.frames.length - 1, step));
  const f = P.frames[step];
  const present = new Map(f.items.map((it) => [it.id, it]));
  const counts = allocate(P.ids.map((e) => e.w), n - Math.round(n * DUST));

  const target = new Float32Array(n * 3), offset = new Float32Array(n * 4), tint = new Float32Array(n * 3), pick = new Float32Array(n).fill(-1);
  const labels = [], labelOf = new Map();
  for (const it of f.items) {
    if (!it.label) continue;
    labelOf.set(it.id, labels.length);
    labels.push({ label: it.label, info: it.info || '', color: it.color, count: 0, stage: it.stage || 0, centroid: [0, 0, 0], offset: (it.ex || [0, 0, 0]).map((v) => v * scale) });
  }

  let off = 0;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  P.ids.forEach((e, k) => {
    const c = counts[k], it = present.get(e.id);
    if (!it) { for (let i = off; i < off + c; i++) { dustAt(i, target, i * 3); tint.set(DUST_TINT, i * 3); } off += c; return; }
    const prims = flatten(it.shape.parts), rng = makeRng(hashStr(e.id));
    const pc = allocate(prims.map((p) => p.weight), c), tmp = new Float64Array(c * 3);
    let o = 0;
    prims.forEach((p, i) => { p.sample(pc[i], rng, tmp, o); o += pc[i]; });
    const g = it.glow ?? 1, col = it.color.map((v) => v * g), ex = (it.ex || [0, 0, 0]).map((v) => v * scale), stage = it.stage || 0;
    const li = labelOf.has(e.id) ? labelOf.get(e.id) : -1, L = labels[li];
    for (let i = 0; i < c; i++) {
      const j = (off + i) * 3, q = (off + i) * 4;
      for (let a = 0; a < 3; a++) {
        const v = (tmp[i * 3 + a] - center[a]) * scale;
        target[j + a] = v; offset[q + a] = ex[a];
        lo[a] = Math.min(lo[a], v + ex[a]); hi[a] = Math.max(hi[a], v + ex[a]);
        if (L) L.centroid[a] += v;
      }
      offset[q + 3] = stage; tint.set(col, j); pick[off + i] = li;
    }
    if (L) L.count += c;
    off += c;
  });
  for (let i = off; i < n; i++) { dustAt(i, target, i * 3); tint.set(DUST_TINT, i * 3); }
  for (const L of labels) L.centroid = L.centroid.map((v) => v / Math.max(L.count, 1));

  let exCenter = [0, 0, 0], fitScale = 1;
  if (P.explodable && lo[0] < Infinity) {
    exCenter = [0, 1, 2].map((a) => (lo[a] + hi[a]) / 2);
    const r = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
    fitScale = Math.min(1, FIT_RADIUS / Math.max(r, 1e-6));
  }
  const tags = (f.tags || []).map((t) => {
    const it = t.id ? present.get(t.id) : null;
    return { key: t.key, text: t.text, color: t.color || [0.92, 0.96, 1], big: !!t.big,
      at: t.at.map((v, a) => (v - center[a]) * scale), off: (it?.ex || [0, 0, 0]).map((v) => v * scale), stage: it?.stage || 0 };
  });
  return { n, target, offset, tint, pick, labels, groups: labels.map((l) => l.label), fitScale, exCenter,
    tags, caption: f.caption || '', line: f.line ?? -1, step, steps: P.frames.length, explodable: P.explodable, ms: performance.now() - t0 };
}
