// The build gate has to actually gate. This runs the real
// `scripts/verify-build-model.mjs` (the same script `npm run build` invokes)
// as a subprocess against fixture directories, and asserts it exits non-zero
// with a readable reason for every way the shipped model can be wrong.
//
// Without this, "the build fails loudly on a bad artifact" would be a claim
// rather than a checked property.

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { execFileSync } from "node:child_process";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../../..");
const SCRIPT = join(APP_ROOT, "scripts/verify-build-model.mjs");
const SOURCE = join(APP_ROOT, "public/model/ranker");

let work: string;

/** Run the gate against `dir`. Returns exit code + combined output. */
function runGate(dir: string): { code: number; out: string } {
  try {
    const out = execFileSync(process.execPath, [SCRIPT, "--dist", dir], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    return { code: 0, out };
  } catch (e) {
    const err = e as { status?: number; stdout?: string; stderr?: string };
    return { code: err.status ?? 1, out: `${err.stdout ?? ""}${err.stderr ?? ""}` };
  }
}

function freshCopy(name: string): string {
  const dir = join(work, name);
  cpSync(SOURCE, dir, { recursive: true });
  return dir;
}

describe("build gate: verify-build-model.mjs rejects a bad shipped model", () => {
  beforeAll(() => {
    work = mkdtempSync(join(tmpdir(), "showboat-buildgate-"));
  });
  afterAll(() => rmSync(work, { recursive: true, force: true }));

  it("accepts the real staged model", () => {
    const r = runGate(freshCopy("good"));
    expect(r.code).toBe(0);
    expect(r.out).toContain("OK");
  });

  it("rejects a missing manifest", () => {
    const dir = freshCopy("no-manifest");
    unlinkSync(join(dir, "manifest.json"));
    const r = runGate(dir);
    expect(r.code).not.toBe(0);
    expect(r.out).toContain("BUILD REJECTED");
    expect(r.out).toContain("stage-production-model");
  });

  it("rejects a missing artifact", () => {
    const dir = freshCopy("no-artifact");
    const m = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
    unlinkSync(join(dir, m.artifact));
    const r = runGate(dir);
    expect(r.code).not.toBe(0);
    expect(r.out).toMatch(/does not exist/);
  });

  it("rejects a truncated artifact", () => {
    const dir = freshCopy("truncated");
    const m = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
    const bytes = readFileSync(join(dir, m.artifact));
    writeFileSync(join(dir, m.artifact), bytes.subarray(0, 1000));
    const r = runGate(dir);
    expect(r.code).not.toBe(0);
    expect(r.out).toMatch(/bytes/);
  });

  it("rejects a same-length but corrupted artifact (hash check)", () => {
    const dir = freshCopy("corrupted");
    const m = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
    const bytes = Buffer.from(readFileSync(join(dir, m.artifact)));
    bytes[bytes.length - 1] ^= 0xff;
    writeFileSync(join(dir, m.artifact), bytes);
    const r = runGate(dir);
    expect(r.code).not.toBe(0);
    expect(r.out).toMatch(/sha256/);
  });

  it("rejects a manifest whose hash no longer matches the reviewed training manifest", () => {
    const dir = freshCopy("wrong-provenance");
    const m = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
    const bytes = Buffer.from(readFileSync(join(dir, m.artifact)));
    bytes[0] = bytes[0]; // unchanged; we forge the manifest instead
    m.onnx_sha256 = "0".repeat(64);
    writeFileSync(join(dir, "manifest.json"), JSON.stringify(m));
    const r = runGate(dir);
    expect(r.code).not.toBe(0);
    expect(r.out).toMatch(/sha256/);
  });

  it("rejects a schema-version mismatch", () => {
    const dir = freshCopy("wrong-schema");
    const m = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
    m.schema_version = "showboat-ranker-v1";
    writeFileSync(join(dir, "manifest.json"), JSON.stringify(m));
    const r = runGate(dir);
    expect(r.code).not.toBe(0);
    expect(r.out).toMatch(/schema_version/);
  });

  it("rejects a dimension mismatch", () => {
    const dir = freshCopy("wrong-dim");
    const m = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
    m.total_dim = 67;
    writeFileSync(join(dir, "manifest.json"), JSON.stringify(m));
    const r = runGate(dir);
    expect(r.code).not.toBe(0);
    expect(r.out).toMatch(/total_dim/);
  });
});
