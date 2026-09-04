---
license: other
license_name: docforensics-dual
license_link: LICENSE.md
tags:
  - document-layout-analysis
  - object-detection
  - onnx
  - document-ai
  - ocr
  - forms
library_name: onnx
pipeline_tag: object-detection
---

# docforensics-layout-s

A 8.8 MB document layout detector for **damaged, scanned and photographed pages**.
Runs in a browser through onnxruntime-web, or anywhere ONNX runs. No GPU.

It detects fifteen region types including form fields and checkboxes, and it
reports eight page-condition signals (blur, skew, tearing, bleed-through,
handwriting, and whether the page is a scan carrying an invisible OCR layer)
that no comparable detector produces.

## What it is good at

On held-out degraded pages it reaches **0.449 mAP@50-95 where YOLOv8n-DocLayNet
reaches 0.194**, at 71% of that model's size. Across 120 held-out degraded
pages the baseline returned no regions at all on 14 of them; this model returned
more regions on 104.

| | this model | YOLOv8n-DocLayNet |
|---|---:|---:|
| size | **8.77 MB** | 12.27 MB |
| degraded mAP@50-95 | **0.449** | 0.194 |
| clean mAP@50-95 | 0.477 | **0.660** |
| form fields | **AP 0.670** | no such class |
| page-condition signals | **8** | none |

## What it is not good at

**Clean, born-digital pages rendered at high quality.** YOLOv8n-DocLayNet is
better there, 0.660 against 0.477, and if that is your input you should use it.
The deficit is uniform across all ten shared classes, which reflects parameter
count rather than a class-specific weakness.

**Picture-heavy magazine layout.** On a 1959 illustrated magazine page the
baseline found 17 regions to this model's 10, separating advertisement panels
and captions more finely.

**Checkboxes at 640 px.** A checkbox is roughly 6.5 px at this input size, below
the finest anchor stride, so AP is 0.153. Read them from the PDF's AcroForm
widgets instead when the page is born-digital.

## Use it

```python
import cv2, numpy as np, onnxruntime as ort

CLASSES = ["text","heading","list","table","picture","caption","formula",
           "header","footer","footnote","seal","form","field","checkbox","signature"]

sess = ort.InferenceSession("docforensics-layout-s.onnx",
                            providers=["CPUExecutionProvider"])

def letterbox(img, side=640):
    h, w = img.shape[:2]
    s = side / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.full((side, side, 3), 114, np.uint8)
    canvas[:small.shape[0], :small.shape[1]] = small
    return canvas

img = letterbox(cv2.imread("page.png"))
x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None]

det, signals, maps = sess.run(None, {"images": x.astype(np.float32),
                                     "evidence": np.zeros((1, 4), np.float32)})

# det is [1, 19, 8500]: rows 0-3 are cx,cy,w,h in 640px space, rows 4-18 are
# per-class scores. Apply your own confidence threshold and NMS.
boxes, scores = det[0, :4].T, det[0, 4:].T
keep = scores.max(1) > 0.18
print(f"{keep.sum()} candidate regions before NMS")

# signals is [1, 8]: skew, blur, noise, bleed, warp, tears, handwriting, native
print(dict(zip(["skew","blur","noise","bleed","warp","tears","hand","native"],
               signals[0].round(3))))
```

Outputs are YOLOv8-shaped on purpose, so any existing YOLOv8 decode and NMS
works unchanged.

## Inputs

`images` is `[1, 3, 640, 640]`, RGB, ImageNet-normalised, letterboxed with fill
value 114. `evidence` is `[1, 4]` and may be all zeros; it carries optional
deterministic page flags and is reserved for future use.

## How it was trained

68k DocLayNet pages under a synthetic degradation pipeline covering skew, blur,
sensor noise, bleed-through, perspective warp, physical tearing and aged-paper
fading, plus 2,184 form pages labelled directly from AcroForm widget geometry
and 6,821 pages labelled by a deterministic vector extractor. Training data is
CDLA-Permissive-1.0 (DocLayNet) and US federal public domain (IRS forms, NASA
technical reports). No proprietary or unlicensed material was used.

## Licence

Dual-licensed, AGPL-3.0 or commercial. See `LICENSE.md`.

Published by **Ginexys**, a d/b/a of **Canworks, LLC**.
Commercial licensing: contact@ginexys.com.
Licence terms and compliance: legal@ginexys.com.

The weights are separable from the surrounding pipeline and were trained
entirely on permissively licensed and public-domain data.

## Citation

See `CITATION.cff`.

A technical report documents the evaluation methodology, including a measured
decorrelation between detection mAP and downstream extraction quality. It is
not published yet; write to contact@ginexys.com if you need it before then.
