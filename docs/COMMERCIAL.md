# Commercial licensing

## Who this is for

You need document layout analysis on pages that are scanned, photographed,
faxed, microfilmed or otherwise damaged, and you are shipping a closed-source
product.

Typical cases:

- **Digitisation and archives.** Microfilm, bound books, historical records.
- **Legal e-discovery.** Scanned exhibits, photocopies, faxes.
- **Claims processing.** Photographs of forms taken on a phone.
- **Records management.** Replacing a per-page cloud OCR bill with a
  client-side model.

## Why not the free track

AGPL-3.0 requires you to publish your source if you run a modified version as a
network service. Most commercial products cannot accept that. The commercial
licence removes it.

## What is different from the alternatives

**Against cloud OCR APIs.** This runs on the client. No page ever leaves the
device, which matters for legal, medical and financial documents, and there is
no per-page cost.

**Against larger layout models.** 8.8 MB loads in a browser. A 300 MB
transformer does not.

**Against YOLOv8n-DocLayNet.** Better on damaged pages by a factor of 2.3,
detects form fields, and reports page condition. Worse on clean rendered pages,
which is stated plainly rather than hidden.

## The page-condition signals

No comparable detector reports these. Eight scalars per page: skew, blur,
sensor noise, bleed-through, perspective warp, physical tearing, handwriting
presence, and whether the page is a genuine digital document or a scan carrying
an invisible OCR layer.

That last signal has an audit use. A scanned page with a synthetic text layer
looks born-digital to most pipelines, and this model flags it.

## Getting started

Evaluate under AGPL first. The model, weights and examples are public, and no
licence is needed to try them.

When you are ready to ship:

- **contact@ginexys.com** for pricing, evaluation and tier selection
- **legal@ginexys.com** for licence terms, redistribution and compliance

Ginexys, a d/b/a of Canworks, LLC. https://ginexys.com
