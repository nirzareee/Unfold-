// Building blocks for steppable scenes. A scene is a list of frames; each frame lists the items on screen
// (id, shape, colour) plus text tags. Items keep their particles across frames by id, so when an element
// moves between frames its particles glide with it; items that leave dissolve into the background dust.
import { boxEdges, boxSurface, line, polyline } from '../lib/sampling.js';
import { sphere } from '../lib/shapes.js';
import { add, sub, mul, norm, len } from '../lib/vec.js';

export const C = {
  cyan: [0.4, 0.88, 1.0], amber: [1.0, 0.78, 0.32], pink: [1.0, 0.42, 0.78], green: [0.42, 1.0, 0.6],
  violet: [0.68, 0.55, 1.0], red: [1.0, 0.38, 0.36], white: [0.88, 0.94, 1.0], dim: [0.32, 0.42, 0.58], blue: [0.38, 0.58, 1.0],
};

const span = (c, s) => ({ lo: [c[0] - s[0] / 2, c[1] - s[1] / 2, c[2] - s[2] / 2], hi: [c[0] + s[0] / 2, c[1] + s[1] / 2, c[2] + s[2] / 2] });
const spanPts = (pts, pad = 0) => {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of pts) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k] - pad); hi[k] = Math.max(hi[k], p[k] + pad); }
  return { lo, hi };
};

/** Wireframe box with a faint skin. */
export function cell(c, s = [0.9, 0.9, 0.9], fill = 0.3) {
  return { parts: [boxEdges(c, s, 0.006, 6), boxSurface(c, s, fill, null, true, true)], ...span(c, s) };
}

/** Vertical bar standing on y = base. */
export const bar = (x, h, { w = 0.62, z = 0, base = 0 } = {}) => cell([x, base + h / 2, z], [w, Math.max(h, 0.05), w], 0.35);

/** Glowing node. */
export function node(c, r = 0.36) {
  return { parts: [sphere(c, r, 2.2), sphere(c, r * 0.35, 3)], ...span(c, [2 * r, 2 * r, 2 * r]) };
}

/** Edge between two points, trimmed so it stops at node surfaces; optional arrow head. */
export function edge(a, b, { trim = 0.4, arrow = false, r = 0.008, density = 3 } = {}) {
  const d = norm(sub(b, a)), p0 = add(a, mul(d, trim)), p1 = sub(b, mul(d, trim));
  const parts = [line(p0, p1, r, density)];
  if (arrow) {
    const side = norm([-d[1], d[0], 0]), back = sub(p1, mul(d, 0.22));
    parts.push(line(p1, add(back, mul(side, 0.12)), r, density * 2), line(p1, sub(back, mul(side, 0.12)), r, density * 2));
  }
  return { parts, ...spanPts([p0, p1], 0.1) };
}

/** Arc from a to b bulging upward by h (attention links). */
export function arc(a, b, h, { density = 4, r = 0.01 } = {}) {
  const pts = [];
  for (let i = 0; i <= 28; i++) {
    const t = i / 28;
    pts.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t + 4 * h * t * (1 - t), a[2] + (b[2] - a[2]) * t]);
  }
  return { parts: polyline(pts, r, density), ...spanPts(pts, 0.1) };
}

/** Flat plate (a matrix / layer seen edge-on). */
export const plate = (c, s, fill = 0.5) => cell(c, s, fill);

/** Bracket-like rail under a range [x0, x1] at height y. */
export function rail(x0, x1, y, z = 0) {
  const pts = [[x0, y + 0.18, z], [x0, y, z], [x1, y, z], [x1, y + 0.18, z]];
  return { parts: polyline(pts, 0.006, 4), ...spanPts(pts, 0.05) };
}

export const dist = (a, b) => len(sub(a, b));

/** item(id, shape, color, extra): extra = { label, info, ex: explode offset, stage, glow } */
export const item = (id, shape, color, extra = {}) => ({ id, shape, color, ...extra });
/** tag(key, at, text, extra): extra = { color, big, id: item to follow in the exploded view } */
export const tag = (key, at, text, extra = {}) => ({ key, at, text, ...extra });
