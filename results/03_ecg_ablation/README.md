# 03: ECG ablation (encoder pretraining-data ablation study)

**One-line finding**: see `ECG_ABLATION_RESULTS.md` for the full
conclusion — the ablation varies what the *encoder* was pretrained on
(ablations A-E, plus an old-350-subject-era encoder reference), then
evaluates the resulting embeddings on ECG-only (and a couple of
cross-check conditions) fine-tuning.

## What's here

- `ECG_ABLATION_RESULTS.md`, `ECG_ABLATION_DETAILED.md` — the summary
  and full detail.
- `per_run_metrics/` — `metrics.json` only for the 10 ablation
  fine-tuning runs: ablations A/B/C/D/E and the old-350-subject-encoder
  reference, each on ECG_ONLY (6 dirs), plus the matching EEG_ONLY
  (ablation A, ablation E, old-encoder — 3 dirs) and EEG_ECG (ablation
  A — 1 dir) cross-checks that exist on disk.

## Regeneration

Two stages: (1) pretrain the ablation encoder itself — see
`sleepfm-data/checkpoints/encoders/fromscratch_ablation_{A-E}_fullcohort`
and `fromscratch_old350enc_ablation` for the existing checkpoints
(`best.pt`+`config.json`, ~75M each, cheap to keep); each encoder's own
pretraining took hours originally. (2) Fine-tune via
`finetune_sleep_staging.py` with `model_path` pointed at the ablation
encoder, same as any other condition, ~10-20 min/fold.

Note: this family's ablation encoders' *embeddings* (the big,
expensive-to-regenerate artifact, not the small `.pt` checkpoint) were
already deleted in an earlier session to free scratch space (~146GB) —
the exact regeneration command is preserved in the Moody area's
`ABLATION_EMBEDDINGS_DELETION_MANIFEST.md` (owner-only, since that
deletion/manifest was done as part of Moody retraining work, but the
command itself applies equally to these MESA ablation encoders).
