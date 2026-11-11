# 05: Patience reruns (MESA only)

**One-line finding**: at matched patience, MESA outperforms Moody on
every condition for every model (From-Scratch, Next-Token, BIOT) — a
consistent cohort-size effect, never reversed. See
`patience_comparison_table.md`.

**This family is MESA-only.** Moody's own numbers live in the
owner-only Moody area and are never copied into git — see `../06_moody/README.md`.
This family exists specifically to hold the MESA half of any
patience-matched comparison, pulled out of the Moody notes files
where they used to live mixed together with Moody's own data.

## What's here

- `mesa_eeg3_p50_reference.md` — MESA EEG3-only at patience 50 (isolates
  the patience effect from the channel effect).
- `mesa_fromscratch_p20.md`, `mesa_nexttoken_p20.md` — the new
  matched-patience-20 MESA batches (20 runs each).
- `patience_comparison_table.md` — the canonical MESA-vs-Moody diff
  table (Moody's notes keep their own copy plus a link here).
- `per_run_metrics/` — `metrics.json` only, 41 run directories (1
  EEG3 p50-ref + 20 From-Scratch p20 + 20 Next-Token p20).

## Tables moved out of the Moody notes, this reorganization

From `RETRAIN_PHASE1_RESULTS.md` (owner-only, Moody area):
- The "Patience-matched MESA EEG3-only reference" table → `mesa_eeg3_p50_reference.md`.
- The "MESA (reference)" column in the main results table and the
  final per-model summary table → values now live here; the Moody
  file keeps only the Moody column plus a link to this folder.

From `MATCHED_PATIENCE20_RESULTS.md` (owner-only, Moody area):
- The entire "MESA From-Scratch p20" and "MESA Next-Token p20" result
  tables → `mesa_fromscratch_p20.md`, `mesa_nexttoken_p20.md`.
- The "BIOT MESA p20" per-fold table → merged into
  `patience_comparison_table.md`.
- The Moody file keeps its own Moody numbers and the MESA−Moody diff
  (a legitimate comparison fact, not pure-MESA data), with a link here
  for the MESA half.

## Regeneration

See each file's own "Regeneration" section.
