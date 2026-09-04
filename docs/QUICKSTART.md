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

## Node

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
