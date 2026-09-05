# Quickstart

## Python

```bash
pip install onnxruntime opencv-python numpy
python examples/detect.py your-page.png out.png
```

Prints the regions found and the eight page-condition signals, and writes an
annotated image.

## Browser

Serve the directory and open `examples/browser.html`:

```bash
python -m http.server 8000
```

The model loads once (8.8 MB) and then runs per page with no network traffic.

## Browser (npm)

```bash
npm install @canwork/docforensics onnxruntime-web
```

```js
import { load, detect } from '@canwork/docforensics';
import * as ort from 'onnxruntime-web';

// Weights and manifest are fetched from Hugging Face on first load and
// then cached by the browser. Pass a URL instead to use your own copy.
const model = await load('docforensics-layout-s', ort);
const { regions, signals } = await detect(model, canvas);
```

## Node

Same package, same two calls. `detect()` takes a canvas *or* raw pixels, so on
a server you hand it whatever your decoder produced — no canvas library, no
DOM shim.

```bash
npm install @canwork/docforensics onnxruntime-node sharp
```

```js
import { load, detect } from '@canwork/docforensics';
import ort from 'onnxruntime-node';
import sharp from 'sharp';

const model = await load('docforensics-layout-s', ort);

const { data, info } = await sharp('page.png')
  .removeAlpha().raw().toBuffer({ resolveWithObject: true });

const { regions, signals } = await detect(model, {
  data, width: info.width, height: info.height,
});
```

Node runs on the CPU execution provider and is roughly three times faster than
the browser's WebAssembly build — about 90 ms a page against 290 ms on an M-series
laptop. The results are identical: the same page gives the same 74 regions in
the browser, under Node, and from `examples/detect.py`.

Pass `executionProviders` if you want to override the runtime's own default:

```js
await load('docforensics-layout-s', ort, { executionProviders: ['cpu'] });
```

## Reading the output

`regions` carry a label from the fifteen classes, a score, and a bbox in
source-image pixels.

`signals` are eight scalars in [0,1]:

| signal | meaning |
|---|---|
| `skew` | rotation, 0.5 is level |
| `blur` | out of focus or low resolution |
| `noise` | sensor or scan grain |
| `bleed` | text showing through from the reverse side |
| `warp` | perspective distortion, a photographed page |
| `tears` | physical damage |
| `handwriting` | handwritten marks present |
| `native` | 1.0 means born-digital, 0.0 means a scan |

`native` is the one worth knowing about. A scanned page carrying an invisible
OCR text layer looks born-digital to most pipelines. This flags it.

## Choosing a confidence threshold

0.18 is the default and suits degraded pages, where scores run lower. Raise it
toward 0.3 on clean documents if you are getting spurious small regions.
