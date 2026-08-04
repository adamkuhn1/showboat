// Human-readable dataset report, generated FROM the manifest + shards —
// never hand-duplicated stats (see task brief: "Generate human-readable
// report from manifest, not hand-duplicated stats"). Prints per-candidate-
// kind coverage (nominal legal-pot rate, foul/scratch rates, mean feature
// sanity) alongside the provenance/version/split/hash fields already in
// dataset_manifest.json, and writes the same content to REPORT.md next to
// the dataset for later inspection.
//
// Usage: RANKER_PROFILE=pilot npx tsx training/ranker/phase2c/report.ts

import { readFileSync, readdirSync, existsSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

interface KindStats {
  n: number;
  states: Set<string>;
  legalPotSum: number;
  anyPotSum: number;
  foulSum: number;
  scratchSum: number;
  perturbationsSum: number;
}

function main() {
  const profileName = process.env.RANKER_PROFILE ?? "smoke";
  const dir = join(__dirname, "data", profileName);
  const manifestPath = join(dir, "dataset_manifest.json");
  if (!existsSync(manifestPath)) {
    process.stderr.write(`[report] no dataset_manifest.json in ${dir} — run generation first\n`);
    process.exit(1);
  }
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));

  const shardFiles = readdirSync(dir).filter((f) => f.endsWith(".ndjson") && !f.endsWith(".tmp"));
  const byKind = new Map<string, KindStats>();
  for (const shardFile of shardFiles) {
    const lines = readFileSync(join(dir, shardFile), "utf8").trim().split("\n").filter(Boolean);
    for (const line of lines) {
      const row = JSON.parse(line);
      let s = byKind.get(row.candidate_kind);
      if (!s) {
        s = { n: 0, states: new Set(), legalPotSum: 0, anyPotSum: 0, foulSum: 0, scratchSum: 0, perturbationsSum: 0 };
        byKind.set(row.candidate_kind, s);
      }
      s.n++;
      s.states.add(row.state_id);
      s.legalPotSum += row.raw_counts.legal_pot;
      s.anyPotSum += row.raw_counts.any_pot;
      s.foulSum += row.raw_counts.foul;
      s.scratchSum += row.raw_counts.scratch;
      s.perturbationsSum += row.n_perturbations;
    }
  }

  const lines: string[] = [];
  lines.push(`# Phase 2C dataset report — profile=${profileName}`);
  lines.push("");
  lines.push(`Generated from: \`${manifestPath}\``);
  lines.push("");
  lines.push("## Provenance");
  lines.push(`- dataset_schema_version: ${manifest.dataset_schema_version}`);
  lines.push(`- feature_schema_version: ${manifest.feature_schema_version}`);
  lines.push(`- generator_version (git commit): ${manifest.generator_version}`);
  lines.push(`- physics_version (wasm sha256): ${manifest.physics_version}`);
  lines.push(`- master_seed: ${manifest.master_seed}`);
  lines.push(`- dataset_hash: ${manifest.dataset_hash}`);
  lines.push(`- generated_at: ${manifest.generated_at}`);
  lines.push("");
  lines.push("## Scale");
  lines.push(`- games_played: ${manifest.games_played}`);
  lines.push(`- controlled_states_requested: ${manifest.controlled_states_requested}`);
  lines.push(`- states_used (>=1 candidate): ${manifest.states_used}`);
  lines.push(`- n_rows: ${manifest.n_rows}`);
  lines.push(`- n_shards: ${manifest.n_shards}`);
  lines.push(`- perturbations_per_candidate: ${manifest.perturbations_per_candidate}`);
  lines.push("");
  lines.push("## Splits (family-aware, see split.ts)");
  lines.push(`- rows_by_split: ${JSON.stringify(manifest.rows_by_split)}`);
  lines.push(`- family_count_by_split: ${JSON.stringify(manifest.family_count_by_split)}`);
  lines.push(`- state_count_by_split: ${JSON.stringify(manifest.state_count_by_split)}`);
  lines.push(`- leakage_check: ${manifest.leakage_check.ok ? "PASS" : "FAIL — " + manifest.leakage_check.issues.join("; ")}`);
  lines.push("");
  lines.push("## State origin mix");
  lines.push(`- rows_by_state_origin: ${JSON.stringify(manifest.rows_by_state_origin)}`);
  lines.push("");
  lines.push("## Candidate-kind coverage (computed from shards, not hand-entered)");
  lines.push("");
  lines.push("| kind | rows | distinct states | nominal legal-pot rate | any-pot rate | foul rate | scratch rate |");
  lines.push("|---|---|---|---|---|---|---|");
  for (const [kind, s] of [...byKind.entries()].sort()) {
    const legalRate = s.perturbationsSum > 0 ? (s.legalPotSum / s.perturbationsSum).toFixed(3) : "n/a";
    const anyRate = s.perturbationsSum > 0 ? (s.anyPotSum / s.perturbationsSum).toFixed(3) : "n/a";
    const foulRate = s.perturbationsSum > 0 ? (s.foulSum / s.perturbationsSum).toFixed(3) : "n/a";
    const scratchRate = s.perturbationsSum > 0 ? (s.scratchSum / s.perturbationsSum).toFixed(3) : "n/a";
    lines.push(`| ${kind} | ${s.n} | ${s.states.size} | ${legalRate} | ${anyRate} | ${foulRate} | ${scratchRate} |`);
  }
  lines.push("");
  lines.push(`Overall mean_legal_pot_rate (from manifest): ${manifest.mean_legal_pot_rate}`);
  lines.push("");

  const report = lines.join("\n") + "\n";
  process.stdout.write(report);
  const reportPath = join(dir, "REPORT.md");
  writeFileSync(reportPath, report);
  process.stderr.write(`[report] wrote ${reportPath}\n`);
}

main();
