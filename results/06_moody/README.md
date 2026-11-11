# 06: Moody / PhysioNet 2026 (I0006)

**No numbers live here, by design.** Moody is BDSP-licensed data; all
Moody results, notes, and derived artifacts stay owner-only
(`chmod 600`/`700`) under
`/scratch/project_2019517/moody2026_pilot/`, on scratch, never copied
into this git repo or anywhere group- or world-readable.

Notes files (owner-only, outside git):
- `ZEROSHOT_RESULTS.md` — MESA-pretrained encoders evaluated zero-shot
  on Moody, no Moody-specific training.
- `RETRAIN_PHASE1_RESULTS.md` — From-Scratch and BIOT retrained on
  Moody; Next-Token pretraining/fine-tuning; the token-cache
  degenerate-codebook diagnosis and fix; the flat-window sensitivity
  check; the encoder/embeddings provenance audit.
- `MATCHED_PATIENCE20_RESULTS.md` — the Moody half of the
  matched-patience-20 comparison (MESA half: `../05_patience_reruns/`).

For any MESA-vs-Moody comparison, the MESA numbers are in
`../05_patience_reruns/patience_comparison_table.md`; the Moody notes
link there rather than duplicating the MESA values.

If you need the actual Moody numbers and have the appropriate access,
read the files above directly on the cluster.
