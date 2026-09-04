# Licence

`docforensics-layout-s` is published by **Ginexys**, a d/b/a of
**Canworks, LLC**. https://ginexys.com

`docforensics-layout-s` is dual-licensed. Choose the track that fits your use.

## Track 1 — AGPL-3.0 (free)

Use, modify and redistribute freely under the GNU Affero General Public License
v3.0. The obligation that matters: **if you run a modified version as a network
service, you must publish your source to its users.** That covers a hosted API,
a SaaS product, or an internal tool exposed over a network.

Suitable for research, evaluation, open-source projects, and personal use.

## Track 2 — Commercial licence (paid)

A commercial licence removes the AGPL copyleft obligation. You embed the model
in a closed-source product and publish nothing.

| tier | for | includes |
|---|---|---|
| Weights | teams wanting only the ONNX model | weights, manifest, no copyleft |
| Product | embedding in a commercial application | weights, worker integration, updates |
| Enterprise | volume or on-premise deployment | the above, plus support and a custom-training option |

To licence commercially, or to ask which tier applies to your use:

- **contact@ginexys.com** for pricing and evaluation
- **legal@ginexys.com** for licence terms, redistribution and compliance

## Copyright

Copyright (c) 2026 Canworks, LLC. All rights reserved except as granted
above.

Ginexys is a trading name of Canworks, LLC. Licences are granted by, and
enforceable against, Canworks, LLC.

## What the licence covers

**The weights** (`docforensics-layout-s.onnx`) were trained entirely on
CDLA-Permissive-1.0 material (DocLayNet, from IBM's primary distribution) and
US federal public-domain works (IRS forms, NASA technical reports, USPTO
publications) under 17 U.S.C. 105. No proprietary, scraped or unlicensed
material contributed to them, and no output of another model was distilled into
them.

**The surrounding pipeline** in the pdf-processor repository is AGPL-3.0
independently of this model.

## Attribution for training data

- DocLayNet, IBM Research, CDLA-Permissive-1.0
- IRS forms, US Internal Revenue Service, 17 U.S.C. 105
- NASA technical reports, NTRS, 17 U.S.C. 105
- Official Gazette of the USPTO, 17 U.S.C. 105, digitised by the Internet Archive
