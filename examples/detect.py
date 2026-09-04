#!/usr/bin/env python
"""
detect.py -- run docforensics on a page image and draw the regions.

    python detect.py page.png [out.png]

Needs: onnxruntime, opencv-python, numpy.
"""
import sys
import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["text", "heading", "list", "table", "picture", "caption", "formula",
           "header", "footer", "footnote", "seal", "form", "field", "checkbox",
           "signature"]
SIGNALS = ["skew", "blur", "noise", "bleed", "warp", "tears", "handwriting", "native"]
COLOURS = {"text": (75, 180, 60), "heading": (75, 25, 230), "list": (25, 225, 255),
           "table": (200, 130, 0), "picture": (48, 130, 245), "caption": (180, 30, 145),
           "field": (90, 220, 60), "checkbox": (230, 0, 0), "form": (0, 140, 255)}
CONF, IOU = 0.18, 0.5


def letterbox(img, side=640):
    """Match training preprocessing exactly: fit long side, pad with 114."""
    h, w = img.shape[:2]
    s = side / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.full((side, side, 3), 114, np.uint8)
    canvas[:small.shape[0], :small.shape[1]] = small
    return canvas


def nms(boxes, scores, thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = order[1:][inter / (a[i] + a[order[1:]] - inter) <= thr]
    return keep


def detect(path, model="docforensics-layout-s.onnx"):
    raw = cv2.imread(path)
    if raw is None:
        sys.exit(f"cannot read {path}")
    img = letterbox(raw)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])
    x = x.transpose(2, 0, 1)[None].astype(np.float32)

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    det, signals, _maps = sess.run(
        None, {"images": x, "evidence": np.zeros((1, 4), np.float32)})

    # det is [1, 4+15, 8500] in YOLOv8 row order: cx, cy, w, h, then per-class
    # scores. Any YOLOv8 decode works unchanged.
    d = det[0].T
    cx, cy, bw, bh = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    scores, labels = d[:, 4:].max(1), d[:, 4:].argmax(1)

    out = []
    for c in np.unique(labels[scores > CONF]):
        m = (labels == c) & (scores > CONF)
        for i in nms(xyxy[m], scores[m], IOU):
            out.append((xyxy[m][i], scores[m][i], CLASSES[c]))
    return img, out, dict(zip(SIGNALS, signals[0].round(3).tolist()))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    img, regions, signals = detect(sys.argv[1])
    for box, score, name in regions:
        c = COLOURS.get(name, (150, 150, 150))
        cv2.rectangle(img, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), c, 2)
        cv2.putText(img, f"{name} {score:.2f}", (int(box[0]), max(10, int(box[1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
    print(f"{len(regions)} regions")
    for k, v in signals.items():
        print(f"  {k:12s} {v:.3f}")
    dst = sys.argv[2] if len(sys.argv) > 2 else "out.png"
    cv2.imwrite(dst, img)
    print(f"wrote {dst}")
