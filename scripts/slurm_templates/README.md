# SLURM templates (copied from scratch, originals untouched)

148 templates, copied verbatim from
`/scratch/project_2019517/claude_arm_build/*.slurm` (which is on the
180-day Lustre purge clock and not otherwise backed up). Scanned for
record IDs, tokens, keys, passwords, and secrets before copying —
clean (see the reorganization task's Step 1/3 report). The originals
on scratch were left in place, not deleted.

Two templates are referenced by name in current notes:
`finetune_sleep_staging_fullcohort_2056.slurm` (the MESA benchmark's
main template) and `preprocess_mesa.slurm` — but the latter no longer
exists among these 148; `PIPELINE_VALIDATION_FINDINGS.md` itself
describes it as "the older" preprocessing script, consistent with it
having already been superseded (likely by `preprocess_mesa_timetest.slurm`
or a non-template conversion path) before this snapshot was taken.

## By task type

**Data download/conversion**: `download_caisr`, `download_i0006`,
`download_i0006_large`, `moody_convert_1135`, `preprocess_mesa_timetest`.

**Encoder pretraining**: `pretrain_fromscratch_*` (fullcohort/smoketest/
timingtest/seed123/ablation A-E/reproduction/lrfix-sanitycheck/moody
variants), `pretrain_nexttoken_*`, `pretrain_spectral_*` (fullcohort/
smoketest/timingtest/resumetest), `moody_nexttoken_pretrain`,
`moody_nexttoken_smoketest`.

**Spectral/token cache building**: `build_spectral_cache_*`,
`build_token_cache_*`, `moody_build_spectral_cache`,
`moody_build_token_cache`.

**Embeddings generation**: `generate_embeddings_*` (20 files — fullcohort,
per-encoder, per-ablation, single-channel, seed123, gapfill, timetest),
`moody_generate_embeddings*`.

**Fine-tuning, SleepFM architecture (From-Scratch/Next-Token/Spectral)**:
`finetune_sleep_staging_fullcohort*`, `finetune_spectral_nexttoken_fullcohort`,
`full_cohort_*_run_template`, `finetune_historical_reproduction_*`,
`finetune_eegonly_fold5v1_fold0_checkpointpt`, `real_run_EEG_ONLY_fold5_fold0`,
`moody_finetune_*_run_template`, `mesa_moody_finetune_p20_run_template`,
`mesa_eeg3_p50ref`.

**BIOT**: `biot_fullcohort*`, `biot_smoketest*`, `biot_labram_arm_verify`,
`finetune_biot_resume`, `evaluate_biot_salvage`, `moody_biot_*`,
`reeval_biot_ecg_only`.

**LaBraM**: `labram_fullcohort`, `labram_smoketest`,
`labram_arm_verify_retry`, `labram_resume_ecg_only_fold4`,
`finetune_labram_resume`, `evaluate_labram_salvage`.

**SensorLM**: `sensorlm_fullcohort`, `sensorlm_smoketest`,
`finetune_sensorlm_resume`.

**Zero-shot / regression evaluation**: `moody_zeroshot_eval`,
`mesa_regression_eval`, `regression_test_b_eval`.

**Infra/diagnostic tests (not science, safe to ignore for reproducing
results)**: `runtracking_test_*` (13), `smoketest_biot_timeout*`/`_failure`/
`_run_tracking` (7), `test_collision_fix`, `test_compute_metrics_fix`,
`test_hdf5_dir_override`, `diag_nvidia_smi`, `check_kaggle_auth*`,
`check_kaggle_reachability`, `smoketest_val_fix`,
`smoketest_ft_sleep_staging_*`.

## Currently in active use (as of 2026-11-11)

`mesa_moody_finetune_p20_run_template.slurm`,
`moody_finetune_nexttoken_run_template.slurm`,
`moody_build_spectral_cache.slurm`, `moody_build_token_cache.slurm`,
`moody_nexttoken_pretrain.slurm`, `moody_nexttoken_smoketest.slurm`,
`moody_biot_wave2.slurm`, `mesa_eeg3_p50ref.slurm`,
`finetune_sleep_staging_fullcohort_2056.slurm`. Everything else is
either superseded (an earlier iteration of the same task) or a
one-off diagnostic.
