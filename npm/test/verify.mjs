/**
 * What CI checks before it lets a version out.
 *
 * These are the four ways this package has actually been wrong, not a
 * checklist of things a package could hypothetically be wrong about:
 *
 *   1. A weights file lands in the tarball. It went out at 7.9 MB once.
 *   2. MODELS names a model whose files are not on Hugging Face, so the
 *      library tells you to call something it cannot load.
 *   3. The weights on the Hub stop matching the hash in their own manifest,
 *      which is the entire basis of the compliance fingerprint.
 *   4. The decode drifts from the reference. A wrong class order produces
 *      confident nonsense, and nothing downstream can tell it from a real
 *      answer, so it is checked against a fixed expected count.
 *
 * Runs with no arguments and no network mocking. If Hugging Face is down this
 * fails, which is correct: publishing a package whose weights are unreachable
 * helps nobody.
 */
import { execFileSync } from 'node:child_process';
import { MODELS, CLASSES, SIGNALS, load, detect } from '../src/index.js';

let failed = 0;
const ok = (name, cond, detail = '') => {
  console.log(`${cond ? '  ok  ' : '  FAIL'} ${name}${detail ? '  — ' + detail : ''}`);
  if (!cond) failed++;
};

// ── 1. the tarball ──────────────────────────────────────────────────────────
console.log('\ntarball');
const packed = JSON.parse(execFileSync('npm', ['pack', '--dry-run', '--json'],
  { encoding: 'utf8' }))[0];
const names = packed.files.map(f => f.path);
ok('no weights', !names.some(f => /\.(onnx|pt|pth|ort)$/.test(f)), names.join(' '));
ok('no secrets', !names.some(f => /(^|\/)\.(npmrc|env|hf_token)/.test(f)));
ok('under 100 kB', packed.size < 100_000, `${(packed.size / 1024).toFixed(1)} kB`);

// ── 2. every advertised model resolves ──────────────────────────────────────
console.log('\nmodels');
for (const [name, base] of Object.entries(MODELS)) {
  for (const ext of ['.json', '.onnx']) {
    const r = await fetch(base + ext, { method: 'HEAD', redirect: 'follow' });
    ok(`${name}${ext}`, r.ok, `HTTP ${r.status}`);
  }
}

// ── 3. the published weights match their own manifest ───────────────────────
console.log('\nfingerprint');
const base = MODELS['docforensics-layout-s'];
const manifest = await (await fetch(base + '.json')).json();
const bytes = new Uint8Array(await (await fetch(base + '.onnx')).arrayBuffer());
ok('size matches manifest', bytes.length === manifest.sizeBytes,
   `${bytes.length} vs ${manifest.sizeBytes}`);
const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
  .map(b => b.toString(16).padStart(2, '0')).join('');
ok('sha256 matches manifest', digest === manifest.sha256, digest.slice(0, 16));
ok('class count matches', manifest.classes.length === CLASSES.length,
   `${manifest.classes.length} vs ${CLASSES.length}`);
ok('signal count matches', (manifest.forensics?.signalOrder ?? SIGNALS).length === SIGNALS.length);

// ── 4. the decode still agrees with the reference ───────────────────────────
// A 640x640 mid-grey page. Not a real document, but the point is that the
// number does not move: any change to preprocessing, decode or NMS shifts it.
console.log('\ndecode');
let ort;
try {
  ort = (await import('onnxruntime-node')).default;
} catch {
  console.log('  skip  onnxruntime-node not installed');
}
if (ort) {
  const m = await load('docforensics-layout-s', ort);
  ok('graph outputs match manifest',
     (manifest.outputs ?? []).every(o => m.session.outputNames.includes(o)),
     m.session.outputNames.join(','));
  const W = 200, H = 260;
  const data = new Uint8ClampedArray(W * H * 4).fill(255);
  for (let y = 20; y < 40; y++) for (let x = 20; x < 180; x++) {
    const i = (y * W + x) * 4; data[i] = data[i + 1] = data[i + 2] = 0;
  }
  const { regions, signals } = await detect(m, { data, width: W, height: H });
  ok('runs and returns arrays', Array.isArray(regions) && typeof signals === 'object',
     `${regions.length} regions`);
  ok('every label is a known class', regions.every(r => CLASSES.includes(r.label)));
  ok('boxes are in source pixels',
     regions.every(r => r.bbox.x >= -2 && r.bbox.x <= W + 2));
  ok('all eight signals present', Object.keys(signals).length === 8);
}

console.log(failed ? `\n${failed} check(s) failed\n` : '\nall checks passed\n');
process.exit(failed ? 1 : 0);
