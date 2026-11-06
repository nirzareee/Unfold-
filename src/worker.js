// Builds scene frames off the main thread so stepping never stutters the render loop. Buffers are transferred, not copied.
// The handler is installed before the catalog loads (some scenes fetch data first), so no message is missed.
const catalog = import('./models/catalog.js');

self.onmessage = async (e) => {
  const { id, index, step, n } = e.data;
  try {
    const { buildModel } = await catalog;
    const { def, ...m } = buildModel(index, n, 0, step);
    self.postMessage({ ...m, id, index, step }, [m.target.buffer, m.offset.buffer, m.tint.buffer, m.pick.buffer]);
  } catch (err) {
    self.postMessage({ id, index, step, error: String((err && err.stack) || err) });
  }
};
