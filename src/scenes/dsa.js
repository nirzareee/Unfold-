// Data structures and algorithms as steppable scenes. Each scene runs the real algorithm and records a
// frame per meaningful step, so the visuals can't drift from what the code does.
import { C, cell, bar, node, edge, rail, plate, item, tag } from './kit.js';
import { add } from '../lib/vec.js';

const front = (c, dz = 0.5) => add(c, [0, 0, dz]);

// ------------------------------------------------------------------ stack
function stack() {
  const out = [], S = [];
  let uid = 0, popped = null;
  const y = (i) => 0.5 + i * 1.05;
  const draw = (caption, line, hiTop = false) => {
    const items = [item('base', plate([0, -0.08, 0], [1.5, 0.06, 1.2]), C.dim)], tags = [];
    S.forEach((e, i) => {
      const c = [0, y(i), 0], top = hiTop && i === S.length - 1;
      items.push(item(e.id, cell(c), top ? C.amber : C.cyan, { label: `stack[${i}]`, info: `holds ${e.v}`, glow: top ? 1.25 : 1 }));
      tags.push(tag(e.id, front(c), String(e.v), { big: true }), tag(`i${i}`, [0.75, y(i), 0], String(i), { color: C.dim }));
    });
    tags.push(S.length ? tag('top', [-1.1, y(S.length - 1), 0], 'top →', { color: C.amber }) : tag('top', [-1.1, y(0), 0], 'empty', { color: C.dim }));
    if (popped) {
      const c = [2.4, y(S.length), 0];
      items.push(item(popped.id, cell(c), C.pink));
      tags.push(tag(popped.id, front(c), String(popped.v), { big: true }), tag('ret', [2.4, c[1] + 0.8, 0], 'returned', { color: C.pink }));
    }
    out.push({ caption, line, items, tags });
  };
  const push = (v) => { popped = null; S.push({ id: `s${uid++}`, v }); draw(`push(${v}): it goes on top. O(1).`, 0, true); };
  const pop = () => { popped = S.pop(); draw(`pop(): the top comes off and is returned → ${popped.v}. O(1).`, 1); };
  draw('An empty stack. Last in, first out (LIFO).', -1);
  [4, 9, 2, 7].forEach(push);
  draw(`peek(): read the top without removing it → ${S[S.length - 1].v}.`, 2, true);
  pop(); pop(); push(5);
  popped = null;
  draw('Only the top is ever touched, so every operation is O(1). Undo history and the call stack work this way.', -1);
  return out;
}

// ------------------------------------------------------------------ queue
function queue() {
  const out = [], Q = [];
  let uid = 0, served = null;
  const x = (i) => -1.5 + i * 1.05;
  const draw = (caption, line, hi = -1) => {
    const items = [item('rail', plate([1.1, -0.55, 0], [7.4, 0.05, 1.1]), C.dim)], tags = [];
    Q.forEach((e, i) => {
      const c = [x(i), 0, 0];
      items.push(item(e.id, cell(c), i === hi ? C.amber : C.cyan, { label: `position ${i}`, info: `holds ${e.v}`, glow: i === hi ? 1.25 : 1 }));
      tags.push(tag(e.id, front(c), String(e.v), { big: true }));
    });
    if (Q.length) {
      tags.push(tag('front', [x(0), -0.95, 0], 'front', { color: C.green }), tag('rear', [x(Q.length - 1), 0.85, 0], 'rear', { color: C.amber }));
    } else tags.push(tag('front', [x(0), -0.95, 0], 'empty', { color: C.dim }));
    if (served) {
      const c = [x(-2), 0, 0];
      items.push(item(served.id, cell(c), C.pink));
      tags.push(tag(served.id, front(c), String(served.v), { big: true }), tag('ret', [c[0], 0.85, 0], 'served', { color: C.pink }));
    }
    out.push({ caption, line, items, tags });
  };
  const enq = (v) => { served = null; Q.push({ id: `q${uid++}`, v }); draw(`enqueue(${v}): join at the rear.`, 0, Q.length - 1); };
  const deq = () => { served = Q.shift(); draw(`dequeue(): the front leaves first → ${served.v}. Everyone moves up.`, 1); };
  draw('An empty queue. First in, first out (FIFO).', -1);
  [3, 8, 1, 6].forEach(enq);
  draw(`front(): peek at who is next → ${Q[0].v}.`, 2, 0);
  deq(); deq(); enq(4); enq(9); deq();
  served = null;
  draw('Arrival order is preserved. Printers, task schedulers and BFS all use queues.', -1);
  return out;
}

// ------------------------------------------------------------------ linked list
function linkedList() {
  const out = [];
  const NODE = [1.0, 0.7, 0.7];
  const draw = (caption, line, nodes, links, { headAt = 0, hi = {}, gone = null } = {}) => {
    const items = [], tags = [];
    const at = Object.fromEntries(nodes.map((n) => [n.v, n.p]));
    for (const n of nodes) {
      items.push(item(`n${n.v}`, cell(n.p, NODE), hi[n.v] || C.cyan, { label: `node ${n.v}`, info: 'value + pointer to the next node' }));
      tags.push(tag(`n${n.v}`, front(n.p, 0.4), String(n.v), { big: true }));
    }
    for (const [a, b] of links) {
      if (b === null) { const p = at[a]; tags.push(tag('null', add(p, [1.5, 0, 0]), 'null', { color: C.dim })); items.push(item(`next${a}`, edge(add(p, [0.5, 0, 0]), add(p, [1.25, 0, 0]), { trim: 0, arrow: true }), C.dim)); }
      else items.push(item(`next${a}`, edge(at[a], at[b], { trim: 0.6, arrow: true }), C.white, { glow: 0.8 }));
    }
    const h = nodes.find((n) => n.v === headAt)?.p;
    if (h) tags.push(tag('head', add(h, [0, 0.8, 0]), 'head', { color: C.amber }));
    if (gone) { items.push(item(`n${gone.v}`, cell(gone.p, NODE), C.dim, { glow: 0.5 })); tags.push(tag(`n${gone.v}`, front(gone.p, 0.4), String(gone.v), { color: C.dim })); }
    out.push({ caption, line, items, tags });
  };
  const row = (vals) => vals.map((v, i) => ({ v, p: [i * 1.9 - 2.5, 0, 0] }));
  let N = row([3, 8, 5]);
  draw('A singly linked list: each node holds a value and a pointer to the next node.', -1, N, [[3, 8], [8, 5], [5, null]], { headAt: 3 });
  const six = { v: 6, p: [1.35, -1.5, 0] };
  draw('Insert 6 after 8. First, create the new node anywhere in memory.', 0, [...N, six], [[3, 8], [8, 5], [5, null]], { headAt: 3, hi: { 6: C.amber } });
  draw('Point 6 at whatever 8 points to (5).', 1, [...N, six], [[3, 8], [8, 5], [6, 5], [5, null]], { headAt: 3, hi: { 6: C.amber } });
  draw('Swing 8’s pointer to 6. Done: O(1), nothing had to shift.', 2, [...N, six], [[3, 8], [8, 6], [6, 5], [5, null]], { headAt: 3, hi: { 6: C.amber, 8: C.pink } });
  N = row([3, 8, 6, 5]).map((n) => ({ ...n, p: [n.p[0] - 0.3, 0, 0] }));
  draw('Same list, laid out in order: 3 → 8 → 6 → 5.', -1, N, [[3, 8], [8, 6], [6, 5], [5, null]], { headAt: 3 });
  draw('Delete the head: move head to the second node.', 3, N.slice(1), [[8, 6], [6, 5], [5, null]], { headAt: 8, gone: { v: 3, p: [N[0].p[0], -1.5, 0] } });
  draw('Node 3 is unreachable now and gets freed. The trade-off: no random access, finding the k-th node is O(k).', -1, N.slice(1), [[8, 6], [6, 5], [5, null]], { headAt: 8 });
  return out;
}

// ------------------------------------------------------------------ min-heap (tree view + array view)
function minHeap() {
  const out = [], A = [];
  let uid = 0, side = null;
  const W = 7.2, parent = (i) => Math.floor((i - 1) / 2);
  const pos = (i) => { const d = Math.floor(Math.log2(i + 1)), p = i - (2 ** d - 1); return [((p + 0.5) / 2 ** d - 0.5) * W, 3.6 - d * 1.35, 0]; };
  const apos = (i) => [(i - 3) * 1.0, -0.9, 0];
  const draw = (caption, line, hl = {}) => {
    const items = [], tags = [tag('treeLbl', [-4.3, 3.6, 0], 'tree', { color: C.dim }), tag('arrLbl', [-4.3, -0.9, 0], 'array', { color: C.dim })];
    A.forEach((e, i) => {
      const col = hl[i] || C.cyan, p = pos(i), a = apos(i);
      items.push(item(`t${e.id}`, node(p, 0.36), col, { label: `heap[${i}]`, info: `value ${e.v}` }));
      items.push(item(`a${e.id}`, cell(a), col, { glow: 0.85 }));
      if (i > 0) items.push(item(`te${i}`, edge(pos(parent(i)), p, { trim: 0.42 }), C.dim, { glow: 0.9 }));
      tags.push(tag(`tv${e.id}`, front(p, 0.42), String(e.v), { big: true }), tag(`av${e.id}`, front(a), String(e.v)), tag(`ai${i}`, add(a, [0, -0.75, 0]), String(i), { color: C.dim }));
    });
    if (side) { const p = [4.4, 3.6, 0]; items.push(item(`t${side.id}`, node(p, 0.36), C.pink)); tags.push(tag(`tv${side.id}`, front(p, 0.42), String(side.v), { big: true }), tag('ret', add(p, [0, 0.7, 0]), 'min', { color: C.pink })); }
    out.push({ caption, line, items, tags });
  };
  const swap = (i, j) => { [A[i], A[j]] = [A[j], A[i]]; };
  const insert = (v) => {
    side = null;
    A.push({ id: uid++, v });
    let i = A.length - 1;
    draw(`insert(${v}): add it at the next free slot (end of the array).`, 0, { [i]: C.amber });
    while (i > 0 && A[parent(i)].v > A[i].v) {
      const p = parent(i);
      draw(`${A[i].v} < parent ${A[p].v} → swap (sift up).`, 1, { [i]: C.amber, [p]: C.pink });
      swap(i, p); i = p;
      draw(`Swapped. ${A[i].v} moved up to index ${i}.`, 2, { [i]: C.amber });
    }
    if (A.length > 1) draw(`Parent is smaller, stop. Every parent ≤ its children again.`, 3, { [i]: C.green });
  };
  draw('A min-heap: a complete binary tree stored in a plain array. Parent of i is (i − 1) / 2.', -1);
  [6, 4, 8, 2, 5, 1].forEach(insert);
  draw('The minimum is always at the root, index 0.', -1, { 0: C.green });
  // extract-min
  side = A[0];
  const last = A.pop();
  A[0] = last;
  draw(`extract_min(): take the root (${side.v}) and move the last element (${last.v}) to the top.`, 4, { 0: C.amber });
  let i = 0;
  for (;;) {
    const l = 2 * i + 1, r = l + 1;
    let m = i;
    if (l < A.length && A[l].v < A[m].v) m = l;
    if (r < A.length && A[r].v < A[m].v) m = r;
    if (m === i) break;
    draw(`${A[i].v} is bigger than child ${A[m].v} → swap with the smaller child (sift down).`, 5, { [i]: C.amber, [m]: C.pink });
    swap(i, m); i = m;
    draw('Swapped.', 5, { [i]: C.amber });
  }
  draw('Heap restored in O(log n). Priority queues, Dijkstra and top-k all run on this.', -1, { 0: C.green });
  return out;
}

// ------------------------------------------------------------------ bubble sort
function bubbleSort() {
  const out = [], vals = [5, 2, 8, 1, 9, 3], n = vals.length;
  const a = vals.map((v, k) => ({ id: `b${k}`, v }));
  const x = (i) => (i - (n - 1) / 2) * 1.0, H = 0.42;
  let sortedFrom = n;
  const draw = (caption, line, hl = {}, j = -1) => {
    const items = [item('floor', plate([0, -0.05, 0], [n + 0.6, 0.04, 1.0]), C.dim)], tags = [];
    a.forEach((e, i) => {
      const col = hl[i] || (i >= sortedFrom ? C.green : C.cyan);
      items.push(item(e.id, bar(x(i), e.v * H), col, { label: `a[${i}]`, info: `value ${e.v}`, glow: hl[i] ? 1.25 : 1 }));
      tags.push(tag(e.id, [x(i), e.v * H + 0.35, 0], String(e.v), { big: true }));
    });
    if (j >= 0) tags.push(tag('j', [x(j), -0.45, 0], 'j', { color: C.amber }), tag('j1', [x(j + 1), -0.45, 0], 'j+1', { color: C.amber }));
    out.push({ caption, line, items, tags });
  };
  draw('Bubble sort: walk the array, swap neighbours that are out of order. Big values bubble to the end.', -1);
  for (let i = 0; i < n - 1; i++) {
    let swapped = false;
    for (let j = 0; j < n - 1 - i; j++) {
      draw(`Compare a[${j}] = ${a[j].v} with a[${j + 1}] = ${a[j + 1].v}.`, 2, { [j]: C.amber, [j + 1]: C.amber }, j);
      if (a[j].v > a[j + 1].v) {
        [a[j], a[j + 1]] = [a[j + 1], a[j]]; swapped = true;
        draw(`${a[j + 1].v} > ${a[j].v} → swap.`, 3, { [j]: C.pink, [j + 1]: C.pink }, j);
      }
    }
    sortedFrom = n - 1 - i;
    if (!swapped) { sortedFrom = 0; draw('A full pass with no swaps: the array is sorted, stop early.', 4); break; }
    draw(`Pass ${i + 1} done: ${a[n - 1 - i].v} is in its final place.`, 4);
  }
  sortedFrom = 0;
  draw('Sorted. Bubble sort is O(n²) comparisons, which is why real code uses merge sort or quicksort.', -1);
  return out;
}

// ------------------------------------------------------------------ binary search
function binarySearch() {
  const out = [], a = [2, 5, 8, 12, 16, 23, 38, 45, 56, 72, 91], target = 45, n = a.length;
  const x = (i) => (i - (n - 1) / 2) * 1.0;
  const draw = (caption, line, lo, hi, mid = -1, found = false) => {
    const items = [], tags = [tag('target', [x(0) - 0.4, 1.5, 0], `looking for ${target}`, { color: C.pink, big: true })];
    a.forEach((v, i) => {
      const live = i >= lo && i <= hi, c = [x(i), live ? 0 : -0.35, 0];
      const col = i === mid ? (found ? C.green : C.amber) : live ? C.cyan : C.dim;
      items.push(item(`c${i}`, cell(c), col, { label: `a[${i}]`, info: `value ${v}`, glow: live ? (i === mid ? 1.3 : 1) : 0.35 }));
      tags.push(tag(`v${i}`, front(c), String(v), { big: live, color: live ? undefined : C.dim }), tag(`i${i}`, [x(i), -0.85, 0], String(i), { color: C.dim }));
    });
    if (lo <= hi) {
      items.push(item('range', rail(x(lo) - 0.45, x(hi) + 0.45, 0.6), C.amber));
      tags.push(tag('lo', [x(lo), 0.95, 0], 'lo', { color: C.amber }), tag('hi', [x(hi), 0.95, 0], 'hi', { color: C.amber }));
    }
    if (mid >= 0) tags.push(tag('mid', [x(mid), -1.3, 0], 'mid', { color: found ? C.green : C.amber }));
    out.push({ caption, line, items, tags });
  };
  let lo = 0, hi = n - 1;
  draw(`A sorted array of ${n} numbers. Find ${target} without checking every one.`, 0, lo, hi);
  for (let step = 1; lo <= hi; step++) {
    const mid = (lo + hi) >> 1;
    draw(`Step ${step}: mid = (${lo} + ${hi}) // 2 = ${mid}, and a[${mid}] = ${a[mid]}.`, 2, lo, hi, mid);
    if (a[mid] === target) { draw(`a[${mid}] = ${target}. Found it in ${step} steps instead of ${mid + 1}.`, 3, lo, hi, mid, true); break; }
    if (a[mid] < target) { lo = mid + 1; draw(`${a[mid]} < ${target}, so everything left of mid is too small. Drop it.`, 4, lo, hi); }
    else { hi = mid - 1; draw(`${a[mid]} > ${target}, so everything right of mid is too big. Drop it.`, 5, lo, hi); }
  }
  draw('Each step halves the range: about log₂(n) steps. A million items take about 20.', -1, 0, n - 1, a.indexOf(target), true);
  return out;
}

// ------------------------------------------------------------------ binary search tree insert
function bst() {
  const out = [], keys = [50, 30, 70, 20, 40, 60, 35];
  const T = [];               // { k, p, parent }
  const childPos = (par, left, d) => add(par.p, [(left ? -1 : 1) * 2.6 / 2 ** d, -1.35, 0]);
  const draw = (caption, line, hl = {}, floating = null) => {
    const items = [], tags = [];
    for (const nd of T) {
      items.push(item(`n${nd.k}`, node(nd.p, 0.34), hl[nd.k] || C.cyan, { label: `key ${nd.k}`, info: 'left subtree smaller, right subtree larger' }));
      tags.push(tag(`n${nd.k}`, front(nd.p, 0.4), String(nd.k), { big: true }));
      if (nd.parent) items.push(item(`e${nd.k}`, edge(nd.parent.p, nd.p, { trim: 0.4 }), C.dim));
    }
    if (floating) {
      items.push(item(`n${floating.k}`, node(floating.p, 0.34), C.amber, { glow: 1.25 }));
      tags.push(tag(`n${floating.k}`, front(floating.p, 0.4), String(floating.k), { big: true }));
    }
    out.push({ caption, line, items, tags });
  };
  draw('A binary search tree: smaller keys go left, larger keys go right.', -1);
  for (const k of keys) {
    if (!T.length) { T.push({ k, p: [0, 3.4, 0], parent: null, d: 0 }); draw(`insert(${k}): the tree is empty, ${k} becomes the root.`, 1, { [k]: C.green }); continue; }
    let cur = T[0];
    for (;;) {
      const left = k < cur.k;
      draw(`${k} ${left ? '<' : '>'} ${cur.k} → go ${left ? 'left' : 'right'}.`, left ? 2 : 3, { [cur.k]: C.pink }, { k, p: add(cur.p, [left ? -0.85 : 0.85, 0.75, 0]) });
      const next = T.find((t) => t.parent === cur && (t.k < cur.k) === left);
      if (next) { cur = next; continue; }
      T.push({ k, p: childPos(cur, left, cur.d + 1), parent: cur, d: cur.d + 1 });
      draw(`Empty spot. ${k} becomes the ${left ? 'left' : 'right'} child of ${cur.k}.`, 1, { [k]: C.green });
      break;
    }
  }
  const order = [...T].sort((p, q) => p.p[0] - q.p[0]).map((t) => t.k).join('  ');
  draw(`Read it left to right (in-order): ${order}. Sorted. Search and insert take O(height).`, -1);
  return out;
}

// ------------------------------------------------------------------ breadth-first search
function bfs() {
  const out = [];
  const P = { A: [0, 3.2, 0], B: [-2.2, 1.9, 0], C: [2.2, 1.9, 0], D: [-3.3, 0.5, 0], E: [-1.1, 0.5, 0], F: [1.1, 0.5, 0], G: [3.3, 0.5, 0] };
  const E = [['A', 'B'], ['A', 'C'], ['B', 'D'], ['B', 'E'], ['C', 'F'], ['E', 'F'], ['C', 'G'], ['F', 'G']];
  const nb = (u) => E.filter((e) => e.includes(u)).map((e) => (e[0] === u ? e[1] : e[0])).sort();
  const visited = new Set(), dist = {}, Q = [], tree = new Set();
  let cur = null, done = new Set();
  const qx = (i) => -3 + i * 0.95;
  const draw = (caption, line) => {
    const items = [], tags = [tag('qLbl', [-4.3, -1.1, 0], 'queue', { color: C.dim })];
    for (const [a, b] of E) {
      const k = `${a}${b}`, on = tree.has(k);
      items.push(item(`e${k}`, edge(P[a], P[b], { trim: 0.38 }), on ? C.green : C.dim, { glow: on ? 1 : 0.7 }));
    }
    for (const [u, p] of Object.entries(P)) {
      const col = u === cur ? C.pink : done.has(u) ? C.green : visited.has(u) ? C.amber : C.dim;
      items.push(item(`n${u}`, node(p, 0.36), col, { label: `node ${u}`, info: visited.has(u) ? `distance ${dist[u]} from A` : 'not reached yet' }));
      tags.push(tag(`n${u}`, front(p, 0.42), u, { big: true }));
      if (u in dist) tags.push(tag(`d${u}`, add(p, [0, -0.62, 0]), `d=${dist[u]}`, { color: C.dim }));
    }
    Q.forEach((u, i) => { const c = [qx(i), -1.1, 0]; items.push(item(`q${u}`, cell(c, [0.75, 0.75, 0.75]), C.amber)); tags.push(tag(`q${u}`, front(c, 0.42), u)); });
    out.push({ caption, line, items, tags });
  };
  draw('Breadth-first search explores a graph in rings: everything 1 step away, then 2 steps, and so on.', -1);
  visited.add('A'); dist.A = 0; Q.push('A');
  draw('Start at A: mark it visited and put it in the queue.', 0);
  while (Q.length) {
    cur = Q.shift();
    draw(`Take ${cur} from the front of the queue.`, 2);
    for (const v of nb(cur)) {
      if (visited.has(v)) continue;
      visited.add(v); dist[v] = dist[cur] + 1; Q.push(v);
      tree.add([cur, v].sort().join(''));
      draw(`${v} is new: visit it (distance ${dist[v]}) and add it to the back of the queue.`, 5);
    }
    done.add(cur);
    cur = null;
  }
  draw('Queue empty, done. The green edges form the shortest paths from A in an unweighted graph.', -1);
  return out;
}

const still = { still: true, tilt: 6, viewYaw: 0 };
export const DSA = [
  { name: 'Stack', category: 'Data Structures', color: C.cyan, fact: 'LIFO · push / pop / peek in O(1)', ...still, frames: stack,
    code: ['push(x):  top = top + 1; a[top] = x', 'pop():    x = a[top]; top = top - 1; return x', 'peek():   return a[top]'] },
  { name: 'Queue', category: 'Data Structures', color: C.cyan, fact: 'FIFO · enqueue at the rear, dequeue at the front', ...still, frames: queue,
    code: ['enqueue(x):  add x at the rear', 'dequeue():   remove and return the front', 'front():     look at the front'] },
  { name: 'Linked List', category: 'Data Structures', color: C.cyan, fact: 'nodes + pointers · O(1) insert, O(n) lookup', ...still, frames: linkedList,
    code: ['new = Node(6)', 'new.next = prev.next', 'prev.next = new', 'head = head.next'] },
  { name: 'Min-Heap', category: 'Data Structures', color: C.cyan, fact: 'a tree living in an array · O(log n) insert / extract', ...still, frames: minHeap,
    code: ['a.append(x); i = len(a) - 1', 'while i > 0 and a[parent(i)] > a[i]:', '    swap(a[i], a[parent(i)])', '    i = parent(i)', 'x = a[0]; a[0] = a.pop()', 'sift_down(0)  # swap with the smaller child'] },
  { name: 'Bubble Sort', category: 'Algorithms', color: C.amber, fact: 'O(n²) · simple, stable, slow', ...still, frames: bubbleSort,
    code: ['for i in range(n - 1):', '    for j in range(n - 1 - i):', '        if a[j] > a[j + 1]:', '            a[j], a[j + 1] = a[j + 1], a[j]', '    # a[n-1-i] is now in place'] },
  { name: 'Binary Search', category: 'Algorithms', color: C.amber, fact: 'O(log n) · needs a sorted array', ...still, frames: binarySearch,
    code: ['lo, hi = 0, n - 1', 'while lo <= hi:', '    mid = (lo + hi) // 2', '    if a[mid] == target: return mid', '    if a[mid] < target: lo = mid + 1', '    else: hi = mid - 1'] },
  { name: 'BST Insert', category: 'Algorithms', color: C.amber, fact: 'O(height) · smaller left, larger right', ...still, frames: bst,
    code: ['def insert(node, x):', '    if node is None: return Node(x)', '    if x < node.key: node.left = insert(node.left, x)', '    else: node.right = insert(node.right, x)'] },
  { name: 'Breadth-First Search', category: 'Algorithms', color: C.amber, fact: 'O(V + E) · shortest paths when edges are unweighted', ...still, frames: bfs,
    code: ['queue = deque([start]); visited = {start}', 'while queue:', '    u = queue.popleft()', '    for v in neighbors(u):', '        if v not in visited:', '            visited.add(v); queue.append(v)'] },
];
