// docforensics -- document forensics models, in the browser.
//
// ONE RUNTIME, MANY MODELS
// docforensics is a family of separate lines. `layout` detects regions on
// damaged pages and is the line shipping today; `restore` is in training. They
// are different architectures rather than heads on a shared backbone, so each
// line gets its own entry point here and its own repository of weights.
//
// Nothing is vendored. `load()` takes a model name, fetches the weights and
// their manifest from Hugging Face, and reads the class list out of the
// manifest instead of holding a copy. A hardcoded class list is a second
// source of truth that goes stale the first time a line gains a region type,
// and it goes stale silently, which is the worst way for a decode to be wrong.
//
// Outputs are YOLOv8-shaped by design, so an existing YOLOv8 decode works
// unchanged. Detection is decoded inside the graph; NMS runs here, because
// ONNX NonMaxSuppression is one of the least portable operators in the spec
// and several onnxruntime-web builds handle it only on specific opsets.

const HF = 'https://huggingface.co/ginexys';

/** Known models, by line. Adding a scale is one line here. */
export const MODELS = {
  'docforensics-layout-n': HF + '/docforensics-layout/resolve/main/docforensics-layout-n',
  'docforensics-layout-s': HF + '/docforensics-layout/resolve/main/docforensics-layout-s',
};

// Exported for callers that want the layout class list without loading a
// model. `detect()` reads the live list from the manifest, never from here.
export const CLASSES = ['text', 'heading', 'list', 'table', 'picture', 'caption',
  'formula', 'header', 'footer', 'footnote', 'seal', 'form', 'field',
  'checkbox', 'signature'];

export const SIGNALS = ['skew', 'blur', 'noise', 'bleed', 'warp', 'tears',
  'handwriting', 'native'];

const SIDE = 640;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/**
 * Load a model.
 *
 * @param {string} model  a name from MODELS, or a URL or path to your own
 *                        .onnx. Either way the manifest is read from the same
 *                        location with a .json extension.
 * @param {object} ort    the onnxruntime-web module
 */
export async function load(model, ort) {
  const base = MODELS[model]
    ?? (/^https?:|^\.{0,2}\//.test(model) ? model.replace(/\.onnx$/, '') : null);
  if (!base) {
    throw new Error(
      'unknown model ' + JSON.stringify(model) + '. Pass one of '
      + Object.keys(MODELS).join(', ') + ', or a URL to your own .onnx.');
  }

  const res = await fetch(base + '.json');
  if (!res.ok) throw new Error('manifest ' + base + '.json: HTTP ' + res.status);
  const manifest = await res.json();

  const session = await ort.InferenceSession.create(base + '.onnx', {
    executionProviders: ['wasm'],
  });

  // Refuse a graph that disagrees with its own manifest rather than guessing.
  // A wrong class order produces confident nonsense, which is worse than an
  // error, because nothing downstream can tell it apart from a real answer.
  const outputs = new Set(session.outputNames);
  for (const name of manifest.outputs ?? []) {
    if (!outputs.has(name)) {
      throw new Error(manifest.id + ': manifest names output ' + JSON.stringify(name)
        + ', graph has ' + session.outputNames.join(', '));
    }
  }

  return { session, ort, manifest };
}

/** Letterbox to 640, pad 114. Must match training preprocessing exactly. */
function preprocess(canvas) {
  const c = document.createElement('canvas');
  c.width = c.height = SIDE;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.fillStyle = 'rgb(114,114,114)';
  g.fillRect(0, 0, SIDE, SIDE);
  const s = SIDE / Math.max(canvas.width, canvas.height);
  g.drawImage(canvas, 0, 0, canvas.width * s, canvas.height * s);
  const { data } = g.getImageData(0, 0, SIDE, SIDE);
  const out = new Float32Array(3 * SIDE * SIDE);
  const px = SIDE * SIDE;
  for (let i = 0; i < px; i++) {
    for (let ch = 0; ch < 3; ch++) {
      out[ch * px + i] = (data[i * 4 + ch] / 255 - MEAN[ch]) / STD[ch];
    }
  }
  return { tensor: out, scale: s };
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const w = x2 - x1, h = y2 - y1;
  if (w <= 0 || h <= 0) return 0;
  const i = w * h;
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i);
}

/**
 * Detect regions on a canvas or image element.
 *
 * @returns {{regions: Array, signals: object}}
 *   `regions` carry bbox in SOURCE-image pixels, label and score.
 *   `signals` are eight page-condition scalars in [0,1].
 */
export async function detect({ session, ort, manifest }, canvas,
                             { conf = 0.18, iouThreshold = 0.5 } = {}) {
  if (manifest?.line && manifest.line !== 'layout') {
    throw new Error('detect() is the layout line; ' + manifest.id + ' is ' + manifest.line);
  }
  const classes = manifest?.classes ?? CLASSES;
  const signalOrder = manifest?.forensics?.signalOrder ?? SIGNALS;

  const { tensor, scale } = preprocess(canvas);
  const feeds = {
    images: new ort.Tensor('float32', tensor, [1, 3, SIDE, SIDE]),
    evidence: new ort.Tensor('float32', new Float32Array(4), [1, 4]),
  };
  const out = await session.run(feeds);
  const det = out.detections.data;
  const A = out.detections.dims[2];          // 8500 anchors
  const nc = classes.length;

  // The row is cx, cy, w, h, then one score per class. A graph carrying a
  // different row count than the manifest claims would shift every label after
  // the first, so it is checked rather than assumed.
  const rows = out.detections.dims[1];
  if (rows !== 4 + nc) {
    throw new Error((manifest?.id ?? 'model') + ': graph has ' + rows
      + ' detection rows, manifest lists ' + nc + ' classes (expected ' + (4 + nc) + ')');
  }

  const cand = [];
  for (let i = 0; i < A; i++) {
    let best = 0, bestC = 0;
    for (let c = 0; c < nc; c++) {
      const v = det[(4 + c) * A + i];
      if (v > best) { best = v; bestC = c; }
    }
    if (best <= conf) continue;
    const cx = det[i], cy = det[A + i], w = det[2 * A + i], h = det[3 * A + i];
    cand.push({ box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                score: best, label: classes[bestC] });
  }

  // Per-class NMS.
  cand.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of cand) {
    if (!kept.some(k => k.label === c.label && iou(k.box, c.box) > iouThreshold)) {
      kept.push(c);
    }
  }

  const regions = kept.map(k => ({
    label: k.label,
    score: k.score,
    bbox: {
      x: k.box[0] / scale, y: k.box[1] / scale,
      w: (k.box[2] - k.box[0]) / scale, h: (k.box[3] - k.box[1]) / scale,
    },
  }));

  const sig = out.forensic_signals.data;
  const signals = Object.fromEntries(signalOrder.map((n, i) => [n, sig[i]]));
  return { regions, signals };
}
