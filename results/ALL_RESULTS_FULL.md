# SleepFM on MESA: Full-Cohort Results (1944 subjects, fold5_v1)

---

**How this differs from `ALL_RESULTS.md`**: that file covers the original
350-subject Puhti-era pipeline (270/30/50 train/val/test, a single fixed
split, fold_0 only). This file covers the full 1944-subject cohort using
`fold5_v1` — genuine subject-level 5-fold cross-validation (~388 test
subjects/fold, disjoint train/val/test per fold, disjoint test sets across
folds, all independently re-verified — see the pretraining/fine-tuning audit
trail in `experiments_full_cohort/full_cohort/`). All numbers below are
mean ± std across the 5 folds, independently recomputed from raw prediction
pickles (not just read from `metrics.json`) for every entry in this file.

A controlled investigation (see `experiments_full_cohort/full_cohort/
CV_COMPARISON_fold5_vs_fold10.md` and the historical-reproduction
experiments in the archive) found the ~0.51 full-cohort EEG_ONLY score is
**not** a bug, encoder regression, or pipeline regression relative to the
historical 0.6582 — a fresh, from-random-init encoder trained today on the
identical 350-subject data/split/pipeline reproduces 0.6592, matching the
historical number almost exactly. The gap is a property of evaluating on a
much larger, harder, more representative population (fold5_v1's ~388
test subjects/fold vs. the historical 50-subject fixed test set), not an
artifact of anything in this pipeline.

---

## Section 1: SleepFM From-Scratch — Full-Cohort Modality Ablation

From-scratch encoder pretrained on the full 1944-subject cohort (EEG_ONLY-atomic
leave-one-out contrastive pretraining, 4 modality groups: EEG_ONLY/RESP/EKG/EMG).
Fine-tuned end-to-end on fold5_v1, evaluated per-fold on each fold's held-out
test subjects.

| Modality | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| EEG only | 0.5162 ± 0.0080 | 0.5633 ± 0.0128 | 0.7380 | 0.3512 | 0.5604 | 0.3813 | 0.5500 |
| ECG only | 0.3943 ± 0.0109 | 0.4587 ± 0.0059 | 0.6756 | 0.2839 | 0.4111 | 0.2553 | 0.3455 |
| EEG+ECG | 0.5162 ± 0.0090 | 0.5612 ± 0.0084 | 0.7395 | 0.3594 | 0.5522 | 0.3811 | 0.5487 |

---

## Section 2: SleepFM Spectral — Full-Cohort (EEG_ONLY encoder)

Encoder pretrained via EEG spectral band-power reconstruction on the full
1944-subject cohort. **EEG_ONLY only** — this encoder's cache
(`spectral_cache_eegonly_fullcohort`) never scans EKG/RESP/EMG channels, so
ECG_ONLY and EEG_ECG are not available (confirmed against the actual
cache-construction code).

| Modality | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| EEG only | 0.5042 ± 0.0128 | 0.5533 ± 0.0135 | 0.7130 | 0.3313 | 0.5495 | 0.3810 | 0.5460 |

---

## Section 3: SleepFM Next-Token — Full-Cohort (EEG_ONLY encoder)

Encoder pretrained via next-window token prediction (512-cluster k-means
codebook) on the full 1944-subject cohort. Inherits Spectral's cache
entirely — **EEG_ONLY only**, same limitation as Spectral, confirmed against
the actual config (`token_codebook_path`, `spectral_signals_path_*` all
point at `spectral_cache_eegonly_fullcohort`).

| Modality | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| EEG only | 0.5235 ± 0.0207 | 0.5683 ± 0.0170 | 0.7351 | 0.3552 | 0.5609 | 0.3859 | 0.5804 |

---

## Section 4: Main Comparison — EEG_ONLY, full cohort, fold5_v1

All three completed pretraining methods, same encoder architecture, same fold5_v1 split.

| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| SleepFM Next-Token | 0.5235 ± 0.0207 | 0.5683 ± 0.0170 | 0.7351 | 0.3552 | 0.5609 | 0.3859 | 0.5804 |
| SleepFM From-Scratch | 0.5162 ± 0.0080 | 0.5633 ± 0.0128 | 0.7380 | 0.3512 | 0.5604 | 0.3813 | 0.5500 |
| SleepFM Spectral | 0.5042 ± 0.0128 | 0.5533 ± 0.0135 | 0.7130 | 0.3313 | 0.5495 | 0.3810 | 0.5460 |

---

## Section 5: Still to come

BIOT, LaBraM, SensorLM full-cohort fold5_v1 runs have not been run yet.
This file will be extended with matching sections (and Section 4 updated)
as those complete, mirroring `ALL_RESULTS.md`'s structure.

---

## Provenance

- Detailed per-run artifacts (metrics.json, classification_report.txt,
  per_subject_results.csv, config.json) for every experiment summarized
  here: `experiments_full_cohort/full_cohort/`
- Every macro F1 / per-class F1 number in this file was independently
  recomputed from raw `test_all_outputs.pickle` / `test_all_targets.pickle`
  / `test_all_masks.pickle` (plain `sklearn.f1_score`, not
  `compute_metrics.py`) and matched `metrics.json` exactly for all
  25 underlying fold-level runs.
- Diagnostic/one-off experiments (checkpoint.pt vs best.pt A/B test, old-encoder
  vs full-cohort-encoder comparison, historical-reproduction experiments) are
  NOT included in the summary tables above — they remain in
  `experiments_full_cohort/full_cohort/` for provenance but are not part of
  the standard model-comparison ladder.
