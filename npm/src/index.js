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
//
// BROWSER AND SERVER, ONE PATH
// Nothing here touches `document` or `window` unless you hand it a canvas.
// `load()` fetches the weights into a buffer rather than passing onnxruntime a
// URL, because onnxruntime-node reads a string as a FILE PATH and reports a
// perfectly good HTTPS URL as "File doesn't exist". `detect()` takes a canvas
// or an ImageData-shaped `{data, width, height}`, so a server can decode a
// page with sharp or pdfium and pass the pixels straight in.

const HF = 'https://huggingface.co/ginexys';

// Known models, by line. Adding a scale is one line here -- and only once the
// weights are actually on Hugging Face. A name listed before its files exist
// is the library telling you to call something it cannot load, and the error
// you get is a bare HTTP 404 on a URL you never typed.
export const MODELS = {
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
 * @param {object} ort    onnxruntime-web or onnxruntime-node
 * @param {object} [opts]
 * @param {string[]} [opts.executionProviders]  passed straight to onnxruntime.
 *   Left unset by default so each runtime picks its own: wasm in the browser,
 *   cpu under Node. Naming 'wasm' here used to be hardcoded, which made the
 *   package unusable from onnxruntime-node for no reason.
 */
export async function load(model, ort, opts = {}) {
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

  // Fetched into memory rather than handed over as a URL: onnxruntime-node
  // treats a string as a path on disk and will not fetch anything.
  const wRes = await fetch(base + '.onnx');
  if (!wRes.ok) throw new Error('weights ' + base + '.onnx: HTTP ' + wRes.status);
  const bytes = new Uint8Array(await wRes.arrayBuffer());
  if (manifest.sizeBytes && bytes.length !== manifest.sizeBytes) {
    throw new Error(manifest.id + ': downloaded ' + bytes.length
      + ' bytes, manifest says ' + manifest.sizeBytes + '. Truncated download.');
  }

  const session = await ort.InferenceSession.create(
    bytes, opts.executionProviders ? { executionProviders: opts.executionProviders } : {});

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

/**
 * Pull RGBA pixels out of whatever was passed in.
 *
 * A canvas or an OffscreenCanvas goes through getImageData. Anything carrying
 * `{data, width, height}` -- an ImageData, or the object sharp and pdfium hand
 * you -- is used directly, which is the whole server-side path. Three-channel
 * buffers are widened to four rather than rejected, because "raw RGB" is what
 * most decoders give you when you ask them not to bother with alpha.
 */
function pixels(src) {
  if (src && typeof src.getContext === 'function') {
    const g = src.getContext('2d', { willReadFrequently: true });
    const d = g.getImageData(0, 0, src.width, src.height);
    return { data: d.data, width: d.width, height: d.height };
  }
  if (src && src.data && src.width && src.height) {
    const n = src.width * src.height;
    const ch = src.data.length / n;
    if (ch === 4) return { data: src.data, width: src.width, height: src.height };
    if (ch === 3) {
      const out = new Uint8ClampedArray(n * 4);
      for (let i = 0; i < n; i++) {
        out[i * 4] = src.data[i * 3];
        out[i * 4 + 1] = src.data[i * 3 + 1];
        out[i * 4 + 2] = src.data[i * 3 + 2];
        out[i * 4 + 3] = 255;
      }
      return { data: out, width: src.width, height: src.height };
    }
    throw new Error('expected 3 or 4 channels, got ' + ch);
  }
  throw new Error('detect() takes a canvas or {data, width, height}, got '
    + Object.prototype.toString.call(src));
}

/**
 * Letterbox to 640, pad 114. Must match training preprocessing exactly.
 *
 * The resize is a box filter -- every source pixel landing in a destination
 * pixel is averaged. Training used OpenCV INTER_AREA, which is the same
 * operation, and the browser's drawImage does the same thing when downscaling.
 * Nearest-neighbour would be four lines shorter and would drop most of the
 * thin strokes on a scanned page, which is exactly the signal the detector
 * reads. Pages are always downscaled to 640 here, so the box filter is the
 * only case that needs to be right.
 */
function preprocess(src) {
  const { data, width: sw, height: sh } = pixels(src);
  const scale = SIDE / Math.max(sw, sh);
  // Truncated, not rounded: OpenCV's resize takes an integer size and the
  // training pipeline passes int(w * scale). Rounding here puts the page one
  // column wider than the model was trained to see.
  const dw = Math.max(1, Math.trunc(sw * scale));
  const dh = Math.max(1, Math.trunc(sh * scale));

  const out = new Float32Array(3 * SIDE * SIDE);
  const px = SIDE * SIDE;
  // Pad value first, so anything the page does not cover is already 114.
  for (let ch = 0; ch < 3; ch++) {
    out.fill((114 / 255 - MEAN[ch]) / STD[ch], ch * px, (ch + 1) * px);
  }

  // Fractional-weight area average. A destination pixel covers the source
  // rectangle [x*xr, (x+1)*xr), whose first and last columns are usually only
  // partly inside it. Giving those partial columns full weight -- the obvious
  // implementation -- moved 20% of the input tensor by more than 0.05 against
  // the OpenCV INTER_AREA reference the model was trained on, and changed the
  // region count on a tax form from 76 to 81. The fractions matter.
  const xr = sw / dw, yr = sh / dh;
  for (let y = 0; y < dh; y++) {
    const fy0 = y * yr, fy1 = fy0 + yr;
    const y0 = Math.floor(fy0), y1 = Math.min(sh, Math.ceil(fy1));
    for (let x = 0; x < dw; x++) {
      const fx0 = x * xr, fx1 = fx0 + xr;
      const x0 = Math.floor(fx0), x1 = Math.min(sw, Math.ceil(fx1));
      let r = 0, g = 0, b = 0, wsum = 0;
      for (let sy = y0; sy < y1; sy++) {
        const wy = Math.min(sy + 1, fy1) - Math.max(sy, fy0);
        if (wy <= 0) continue;
        let i = (sy * sw + x0) * 4;
        for (let sx = x0; sx < x1; sx++, i += 4) {
          const w = wy * (Math.min(sx + 1, fx1) - Math.max(sx, fx0));
          if (w <= 0) continue;
          r += data[i] * w; g += data[i + 1] * w; b += data[i + 2] * w; wsum += w;
        }
      }
      const o = y * SIDE + x;
      out[o] = (r / wsum / 255 - MEAN[0]) / STD[0];
      out[px + o] = (g / wsum / 255 - MEAN[1]) / STD[1];
      out[2 * px + o] = (b / wsum / 255 - MEAN[2]) / STD[2];
    }
  }
  return { tensor: out, scale };
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
 * Detect regions on a page.
 *
 * @param {object} model   what load() returned
 * @param {*} input        a canvas, or {data, width, height} with RGB(A)
 *                         pixels -- an ImageData, or the output of sharp's
 *                         .raw() or any PDF rasteriser.
 *
 * @returns {{regions: Array, signals: object}}
 *   `regions` carry bbox in SOURCE-image pixels, label and score.
 *   `signals` are eight page-condition scalars in [0,1].
 */
export async function detect({ session, ort, manifest }, input,
                             { conf = 0.18, iouThreshold = 0.5 } = {}) {
  if (manifest?.line && manifest.line !== 'layout') {
    throw new Error('detect() is the layout line; ' + manifest.id + ' is ' + manifest.line);
  }
  const classes = manifest?.classes ?? CLASSES;
  const signalOrder = manifest?.forensics?.signalOrder ?? SIGNALS;

  const { tensor, scale } = preprocess(input);
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
