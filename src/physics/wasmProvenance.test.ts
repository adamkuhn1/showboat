// The WASM provenance guard, driven.
//
// `scripts/wasm-provenance.mjs` exists to catch one specific way this repo can
// lie to itself: a Rust change that is never compiled, so all 395 tests keep
// passing against the OLD committed binary and the change ships as a no-op.
//
// A guard nobody has watched fail is not a guard, so this drives it: it runs
// the real script against a real copy of the tree, mutates one Rust file, and
// requires a non-zero exit. It also requires the script to leave the artifact
// alone, which is the property that stops it "fixing" drift by hiding it.

import { describe, it, expect } from "vitest";
import { execFileSync } from "node:child_process";
import { cpSync, mkdtempSync, readFileSync, appendFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const APP = join(dirname(fileURLToPath(import.meta.url)), "../..");
const SCRIPT = "scripts/wasm-provenance.mjs";

/** A throwaway copy of just the parts the script reads. */
const sandbox = (): string => {
  const dir = mkdtempSync(join(tmpdir(), "showboat-wasm-prov-"));
  cpSync(join(APP, "physics-core"), join(dir, "physics-core"), { recursive: true });
  cpSync(join(APP, "src/wasm"), join(dir, "src/wasm"), { recursive: true });
  cpSync(join(APP, SCRIPT), join(dir, SCRIPT), { recursive: false });
  return dir;
};

const run = (dir: string, args: string[] = []): { code: number; out: string } => {
  try {
    const out = execFileSync("node", [SCRIPT, ...args], { cwd: dir, encoding: "utf8", stdio: "pipe" });
    return { code: 0, out };
  } catch (e) {
    const err = e as { status?: number; stdout?: string; stderr?: string };
    return { code: err.status ?? 1, out: `${err.stdout ?? ""}${err.stderr ?? ""}` };
  }
};

describe("the committed WASM is checked against the Rust it came from", () => {
  it("passes on the tree as committed", () => {
    // The real check, in the real app directory. If this fails, either someone
    // edited Rust without rebuilding or someone replaced the artifact.
    const r = run(APP);
    expect(r.out).toContain("[wasm-provenance] OK");
    expect(r.code).toBe(0);
  });

  it("fails when a Rust source changes and the WASM does not", () => {
    const dir = sandbox();
    try {
      expect(run(dir).code).toBe(0);
      // The exact mistake this exists for: a real edit to the physics.
      appendFileSync(join(dir, "physics-core/src/collisions.rs"), "\n// a real change\n");
      const r = run(dir);
      expect(r.code).toBe(1);
      expect(r.out).toContain("drifted");
      expect(r.out).toContain("physics-core/src/collisions.rs");
      expect(r.out).toContain("changed since the WASM was built");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("fails when the artifact is replaced and the Rust is not", () => {
    const dir = sandbox();
    try {
      const wasm = join(dir, "src/wasm/showboat_physics_bg.wasm");
      const bytes = new Uint8Array(readFileSync(wasm));
      // Flip a byte well past the header, so it is still a plausible file.
      bytes[bytes.length - 1] ^= 0xff;
      writeFileSync(wasm, bytes);
      const r = run(dir);
      expect(r.code).toBe(1);
      expect(r.out).toContain("showboat_physics_bg.wasm");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("notices a new Rust file that no build has seen", () => {
    const dir = sandbox();
    try {
      writeFileSync(join(dir, "physics-core/src/spin.rs"), "// new module\n");
      const r = run(dir);
      expect(r.code).toBe(1);
      expect(r.out).toContain("physics-core/src/spin.rs");
      expect(r.out).toContain("not in the lockfile");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("never rewrites the artifact — not even to make itself pass", () => {
    const dir = sandbox();
    try {
      const wasm = join(dir, "src/wasm/showboat_physics_bg.wasm");
      const before = readFileSync(wasm);
      appendFileSync(join(dir, "physics-core/src/engine.rs"), "\n// drift\n");
      expect(run(dir).code).toBe(1);
      expect(Buffer.compare(before, readFileSync(wasm))).toBe(0);
      // And `--update` records the new state without touching the binary.
      expect(run(dir, ["--update"]).code).toBe(0);
      expect(Buffer.compare(before, readFileSync(wasm))).toBe(0);
      expect(run(dir).code).toBe(0);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("says how to fix it, including the cargo steps this machine cannot run", () => {
    const dir = sandbox();
    try {
      appendFileSync(join(dir, "physics-core/src/motion.rs"), "\n// drift\n");
      const r = run(dir);
      expect(r.out).toContain("npm run build:wasm");
      expect(r.out).toContain("npm run test:wasm");
      expect(r.out).toContain("--update");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("records that the baseline was taken without a Rust toolchain", () => {
    // Honesty about what the lockfile does and does not assert. If someone
    // later verifies the artifact really compiles from this source, they flip
    // this flag and this test tells them to update the note with it.
    const lock = JSON.parse(
      readFileSync(join(APP, "physics-core/wasm-provenance.json"), "utf8"),
    ) as { builtWith?: { toolchainVerified?: boolean; note?: string } };
    expect(lock.builtWith).toBeDefined();
    if (lock.builtWith!.toolchainVerified === false) {
      expect(lock.builtWith!.note).toContain("NOT that the artifact was observed to compile");
    }
  });
});
