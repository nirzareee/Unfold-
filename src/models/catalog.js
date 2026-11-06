// Everything that can be summoned, in display order, plus the builder that turns a scene step into GPU buffers:
//   target (xyz, normalized into the unit globe), offset (xyz explode vector + w stage), tint (rgb per particle).
import { DSA } from '../scenes/dsa.js';
import { DSA2 } from '../scenes/dsa2.js';
import { DSA3 } from '../scenes/dsa3.js';
import { TRANSFORMERS } from '../scenes/transformer.js';
import { TRANSFORMERS2 } from '../scenes/transformer2.js';
import { TRANSFORMERS3 } from '../scenes/transformer3.js';
import { REAL } from '../scenes/real.js';
import { buildScene, sceneSteps, prepare } from '../scenes/build.js';

const ORDER = ['Stack', 'Queue', 'Linked List', 'Hash Table', 'LRU Cache', 'Min-Heap', 'Trie',
  'Bubble Sort', 'Merge Sort', 'Quicksort', 'Binary Search', 'BST Insert', 'Breadth-First Search', 'Topological Sort', 'Dijkstra', 'Dynamic Programming (LCS)',
  'Tokens → Embeddings', 'Self-Attention', 'Multi-Head Attention', 'Transformer Block', 'Mixture of Experts', 'Next-Token Prediction', 'Beam Search', 'KV Cache',
  'Real GPT-2 Attention', 'Prompt Injection'];
const ALL = [...DSA, ...DSA2, ...DSA3, ...TRANSFORMERS, ...TRANSFORMERS2, ...TRANSFORMERS3, ...REAL];
export const CATALOG = ORDER.map((name) => ALL.find((d) => d.name === name)).map((d) => ({ ...d, kind: 'scene' }));

// scenes behave like the original "machines": hand openness drives the exploded view, parts can be picked
export const isMachine = (i) => !!CATALOG[i];
export const stepsOf = (i) => (CATALOG[i] ? sceneSteps(CATALOG[i]) : 1);
export const explodableOf = (i) => (CATALOG[i] ? prepare(CATALOG[i]).explodable : false);
export const keyOf = (i, step) => i * 1000 + step;

/** Build step `step` of scene `index` with exactly n particles. Deterministic. */
export function buildModel(index, n, seed = 0, step = 0) {
  const def = CATALOG[index];
  return { index, def, ...buildScene(def, step, n) };
}
