// Does the committed browser WASM still correspond to the Rust source?
//
// `src/wasm/showboat_physics_bg.wasm` is a committed build artifact. Every
// physics decision in the app runs through it, and nothing in the repo checks
// that it was built from the Rust beside it. Someone can edit
// `physics-core/src/collisions.rs`, run the test suite, watch 395 tests pass —
// because they all exercise the OLD binary — and ship a change that has no
// effect. That is the failure this guards.
//
// WHAT IT CAN AND CANNOT PROVE
//
// It cannot prove the artifact is the compilation of the source: that needs a
// Rust toolchain, and `wasm-pack` output is not bit-reproducible across
// toolchain versions anyway. What it proves is the thing that actually goes
// wrong — DRIFT. A lockfile records the hash of every Rust input and the hash
// of the artifact at the moment they were last known to agree. Afterwards:
//
//   - Rust source changed, artifact not rebuilt  -> fails, names the files
//   - artifact replaced, source unchanged        -> fails
//   - both changed and `--update` was run        -> passes, and the lockfile
//                                                   diff is in the commit for
//                                                   a reviewer to see
//
// It NEVER writes the artifact. `--update` rewrites only the lockfile, and only
// when a human runs it after a real `npm run build:wasm`.
//
//   node scripts/wasm-provenance.mjs            # check (CI, and `npm test`)
//   node scripts/wasm-provenance.mjs --update   # after rebuilding the WASM

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import { join, dirname, relative } from "node:path";
import { fileURLToPath } from "node:url";

const APP = dirname(dirname(fileURLToPath(import.meta.url)));
const LOCK = join(APP, "physics-core", "wasm-provenance.json");
const ARTIFACT = join(APP, "src", "wasm", "showboat_physics_bg.wasm");
const GLUE = join(APP, "src", "wasm", "showboat_physics.js");

const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");

/** Every file that can change what the WASM contains. Sorted, so it is stable. */
const rustInputs = () => {
  const core = join(APP, "physics-core");
  const src = join(core, "src");
  const files = readdirSync(src)
    .filter((f) => f.endsWith(".rs"))
    .map((f) => join(src, f));
  files.push(join(core, "Cargo.toml"));
  const lock = join(core, "Cargo.lock");
  if (existsSync(lock)) files.push(lock);
  return files.sort();
};

const snapshot = () => {
  const inputs = {};
  for (const f of rustInputs()) inputs[relative(APP, f)] = sha256(readFileSync(f));
  const out = {
    // Documented so the reader knows what this file is before reading the code.
    _: "Hashes of the Rust sources and the committed WASM at the moment they were last known to agree. See scripts/wasm-provenance.mjs.",
    rustInputs: inputs,
    artifact: {
      path: relative(APP, ARTIFACT),
      sha256: sha256(readFileSync(ARTIFACT)),
      bytes: readFileSync(ARTIFACT).byteLength,
    },
  };
  if (existsSync(GLUE)) {
    out.glue = { path: relative(APP, GLUE), sha256: sha256(readFileSync(GLUE)) };
  }
  return out;
};

const update = () => {
  const snap = snapshot();
  // Preserve whatever provenance note a human wrote about the build.
  if (existsSync(LOCK)) {
    const prev = JSON.parse(readFileSync(LOCK, "utf8"));
    if (prev.builtWith) snap.builtWith = prev.builtWith;
  }
  writeFileSync(LOCK, JSON.stringify(snap, null, 2) + "\n");
  console.log(`[wasm-provenance] lockfile updated: ${relative(APP, LOCK)}`);
  console.log(`[wasm-provenance] artifact ${snap.artifact.sha256.slice(0, 16)}… (${snap.artifact.bytes} bytes)`);
};

const check = () => {
  if (!existsSync(LOCK)) {
    console.error(
      `[wasm-provenance] no lockfile at ${relative(APP, LOCK)}. Run:\n` +
        `  node scripts/wasm-provenance.mjs --update`,
    );
    process.exit(1);
  }
  const want = JSON.parse(readFileSync(LOCK, "utf8"));
  const have = snapshot();
  const problems = [];

  for (const [path, hash] of Object.entries(want.rustInputs)) {
    const now = have.rustInputs[path];
    if (now === undefined) problems.push(`${path}: recorded, but the file is gone`);
    else if (now !== hash) problems.push(`${path}: changed since the WASM was built`);
  }
  for (const path of Object.keys(have.rustInputs)) {
    if (!(path in want.rustInputs)) problems.push(`${path}: new Rust file, not in the lockfile`);
  }
  if (have.artifact.sha256 !== want.artifact.sha256) {
    problems.push(
      `${want.artifact.path}: sha256 ${have.artifact.sha256.slice(0, 16)}… != recorded ${want.artifact.sha256.slice(0, 16)}…`,
    );
  }
  if (want.glue && have.glue && have.glue.sha256 !== want.glue.sha256) {
    problems.push(`${want.glue.path}: wasm-bindgen glue changed`);
  }

  if (problems.length > 0) {
    console.error("[wasm-provenance] the committed WASM and the Rust source have drifted:\n");
    for (const p of problems) console.error(`  - ${p}`);
    console.error(
      `\nThe committed artifact is NOT regenerated automatically — that would hide the\n` +
        `problem rather than report it. Rebuild it deliberately and record the result:\n\n` +
        `  npm run build:wasm      # needs a Rust toolchain + wasm-pack\n` +
        `  npm run test:wasm       # cargo test, native\n` +
        `  node scripts/wasm-provenance.mjs --update\n`,
    );
    process.exit(1);
  }

  console.log(
    `[wasm-provenance] OK — ${Object.keys(want.rustInputs).length} Rust inputs and ` +
      `${want.artifact.path} (${want.artifact.bytes} bytes, sha256 ${want.artifact.sha256.slice(0, 16)}…) ` +
      `are unchanged since they were last built together.`,
  );
};

process.argv.includes("--update") ? update() : check();
